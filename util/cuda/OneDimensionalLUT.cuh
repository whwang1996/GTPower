#pragma once
#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "CheckCudaRuntime.cuh"
#include <string>
#include "Enums.hh"
#include "Types.hh"
#include "Types.cuh"
#include "Defines.cuh"
#include "Managed.cuh"
#include "CudaMemStats.hh"
#include "Log.hh"

// We do not use a base class here, since it will cause overhead on GPU, for more details, see https://zhuanlan.zhihu.com/p/372619272 
// (ASPLOS-2021-Judging a Type by Its Pointer: Optimizing GPU Virtual Functions)
namespace utils::cuda {
class OneDimensionalLUT: public utils::cuda::Managed {
public:
  explicit OneDimensionalLUT():
    is_scalar_(false),
    scalar_value_(0.0),
    idx1_dim_(0),
    device_allocation_(nullptr),
    idx1_input_transitions_(nullptr),
    lookup_table_(nullptr) {};  // TODO delete this empty constructor

  explicit OneDimensionalLUT(
    LookupTableIdx idx1_dim, const SlewVal* const idx1_input_transitions, 
    const EnergyVal* const lookup_table,
    bool is_scalar = false,
    EnergyVal scalar_value = 0.0
  ) :
  is_scalar_(is_scalar),
  scalar_value_(scalar_value),
  idx1_dim_(idx1_dim),
  device_allocation_(nullptr),
  idx1_input_transitions_(nullptr),
  lookup_table_(nullptr) {
    if (is_scalar_) {
      return;
    }
    validateTableInputs(idx1_dim, idx1_input_transitions, lookup_table);
    const size_t idx1_bytes = sizeof(SlewVal) * idx1_dim;
    const size_t table_offset = alignOffset(idx1_bytes, alignof(EnergyVal));
    const size_t table_bytes = sizeof(EnergyVal) * idx1_dim;
    const size_t total_bytes = table_offset + table_bytes;
    std::vector<char> host_buffer(total_bytes);
    std::memcpy(host_buffer.data(), idx1_input_transitions, idx1_bytes);
    std::memcpy(host_buffer.data() + table_offset, lookup_table, table_bytes);

    CHECK_CUDA_RUNTIME(cudaMalloc(&device_allocation_, total_bytes));
    char* device_bytes = static_cast<char*>(device_allocation_);
    idx1_input_transitions_ = reinterpret_cast<SlewVal*>(device_bytes);
    lookup_table_ = reinterpret_cast<EnergyVal*>(device_bytes + table_offset);
    CHECK_CUDA_RUNTIME(cudaMemcpy(device_allocation_, host_buffer.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDA_MEM_STATS.add(CudaMemStats::Category::lut_table, total_bytes);
  }

  __device__ __host__ EnergyVal lookUpValue(SlewVal x0) const {
    if (is_scalar_) {
      return scalar_value_;
    }

    const auto& x = idx1_input_transitions_;
    assert(idx1_dim_ > 0 && x != nullptr && lookup_table_ != nullptr);
    if (idx1_dim_ == 1) {
      return lookup_table_[0];
    }

    LookupTableIdx x1_idx = -1;
    LookupTableIdx x2_idx = -1;
    findSegment(x, idx1_dim_, x0, x1_idx, x2_idx);

    // ------calculation------
    assert(x1_idx >= 0 && x1_idx < idx1_dim_ && x2_idx >= 0 && x2_idx < idx1_dim_ && x1_idx < x2_idx);
    const SlewVal x_diff = x[x2_idx] - x[x1_idx];
    assert(x_diff != 0.0);
    SlewVal x01 = (x0 - x[x1_idx]) / x_diff;
    EnergyVal res = lookup_table_[x1_idx] + x01 * (lookup_table_[x2_idx] - lookup_table_[x1_idx]);

    return res;
    // ------end of calculation------
  }

  __host__ __device__ void print() const {
    if (is_scalar_) {
      printf("Is scalar, scalar value: %e\n", scalar_value_);
    } else {
      printf("index1: \n");
      for (int i = 0; i < idx1_dim_; ++i) {
        printf("%e ", idx1_input_transitions_[i]);
      }
      printf("\nnldm table: \n");
      for (int i = 0; i < idx1_dim_; ++i) {
        printf("%e ", lookup_table_[i]);
      }
      printf("\n");
    }
  }

  ~OneDimensionalLUT() {
    if (device_allocation_) {
      cudaFree(device_allocation_);
      device_allocation_ = nullptr;
      idx1_input_transitions_ = nullptr;
      lookup_table_ = nullptr;
      return;
    }

    cudaFree(idx1_input_transitions_);
    idx1_input_transitions_ = nullptr;
    cudaFree(lookup_table_);
    lookup_table_ = nullptr;

    // printf("Destructor of OneDimensionalLUT\n");
  }

private:
  static void failInvalidTable(const char* message) {
    LOG_ERROR << "Invalid OneDimensionalLUT: " << message;
  }

  static void validateTableInputs(
    LookupTableIdx idx1_dim,
    const SlewVal* idx1_input_transitions,
    const EnergyVal* lookup_table
  ) {
    if (idx1_dim <= 0) {
      failInvalidTable("idx1_dim must be positive for non-scalar tables");
    }
    if (idx1_input_transitions == nullptr) {
      failInvalidTable("idx1_input_transitions must not be null for non-scalar tables");
    }
    if (lookup_table == nullptr) {
      failInvalidTable("lookup_table must not be null for non-scalar tables");
    }
    validateStrictlyIncreasing(idx1_input_transitions, idx1_dim, "idx1_input_transitions");
  }

  static void validateStrictlyIncreasing(
    const SlewVal* axis,
    LookupTableIdx dim,
    const char* axis_name
  ) {
    for (LookupTableIdx i = 1; i < dim; ++i) {
      if (!(axis[i - 1] < axis[i])) {
        LOG_ERROR << "Invalid OneDimensionalLUT: " << axis_name << " must be strictly increasing";
      }
    }
  }

  __host__ __device__ static void findSegment(
    const SlewVal* axis,
    LookupTableIdx dim,
    SlewVal value,
    LookupTableIdx& lower_idx,
    LookupTableIdx& upper_idx
  ) {
    assert(dim >= 2);
    if (value <= axis[0]) {
      lower_idx = 0;
      upper_idx = 1;
      return;
    }
    if (value >= axis[dim - 1]) {
      lower_idx = dim - 2;
      upper_idx = dim - 1;
      return;
    }
    for (LookupTableIdx i = 1; i < dim; ++i) {
      if (value <= axis[i]) {
        lower_idx = i - 1;
        upper_idx = i;
        return;
      }
    }
    lower_idx = dim - 2;
    upper_idx = dim - 1;
  }

  static size_t alignOffset(size_t offset, size_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
  }

  bool is_scalar_;
  EnergyVal scalar_value_;
  LookupTableIdx idx1_dim_;
  void* device_allocation_;
  SlewVal* idx1_input_transitions_;  // index1, i.e. outter index, vertical, only on device side
  EnergyVal* lookup_table_; // only on device side
};

class OneDimensionalLUTPair: public utils::cuda::Managed {
public:
  explicit OneDimensionalLUTPair() :
    rise_table_(nullptr),
    fall_table_(nullptr) {
    initWhenState();
  }

  explicit OneDimensionalLUTPair(
    OneDimensionalLUT* fall_table,
    OneDimensionalLUT* rise_table,
    const std::vector<VcdEventVal>& when_state
  ) :
    fall_table_(fall_table),
    rise_table_(rise_table) {
      setWhenState(when_state);
    }

  void setTable(OneDimensionalLUT* table, RISEFALL rise_fall) {
    if (rise_fall == RISE) {
      rise_table_ = table;
    } else if (rise_fall == FALL) {
      fall_table_ = table;
    }
  }

  void setWhenState(const std::vector<VcdEventVal>& when_state) {
    assert(when_state.size() <= MAX_N_PIN);
    initWhenState();
    for (size_t idx = 0; idx < when_state.size(); ++idx) {
      when_state_[idx] = when_state[idx];
    }
  }

  __device__ bool matchPinStates(const VcdEventVal* pin_states, NPinVal n_pin) {
    bool matched = true;
    for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
      if (when_state_[pin_idx] != INVALID_VCD_EVENT_VAL && when_state_[pin_idx] != ((pin_states[pin_idx] == 2 || pin_states[pin_idx] == 0) ? 0 : 1)) {
        matched = false;
        break;
      }
    }

    return matched;
  }

  __host__ __device__ EnergyVal lookUp(SlewVal tramsition_time, RISEFALL rise_fall) const {
    assert(rise_fall == FALL || rise_fall == RISE);
    if (rise_fall == FALL) {
      assert(fall_table_ != nullptr);
      return fall_table_->lookUpValue(tramsition_time);
    } else if (rise_fall == RISE) {
      assert(rise_table_ != nullptr);
      return rise_table_->lookUpValue(tramsition_time);
    } else {
      return -1;
    }
  }

  __host__ __device__ void print() const {
    if (rise_table_) {
      printf("rise table: \n");
      rise_table_->print();
    }

    if (fall_table_) {
      printf("fall table: \n");
      fall_table_->print();
    }
  }

  ~OneDimensionalLUTPair() {
    delete fall_table_;
    fall_table_ = nullptr;
    delete rise_table_;
    rise_table_ = nullptr;
    // printf("Destructor of OneDimensionalLUTPair\n");
  }

private:
  void initWhenState() {
    for (size_t idx = 0; idx < MAX_N_PIN; ++idx) {
      when_state_[idx] = INVALID_VCD_EVENT_VAL;
    }
  }

  OneDimensionalLUT* fall_table_;
  OneDimensionalLUT* rise_table_;
  VcdEventVal when_state_[MAX_N_PIN];
};

} // end of namespace utils::cuda
