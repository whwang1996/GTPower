#pragma once
#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>
#include <string>
#include "Enums.hh"
#include "Types.hh"
#include "Types.cuh"
#include "Defines.cuh"
#include "Managed.cuh"
#include "TableModel.hh"
#include "Managed.cuh"
#include "CudaMemStats.hh"
#include "Log.hh"

// We do not use a base class here, since it will cause overhead on GPU, for more details, see https://zhuanlan.zhihu.com/p/372619272 
// (ASPLOS-2021-Judging a Type by Its Pointer: Optimizing GPU Virtual Functions)
namespace utils::cuda {
class TwoDimensionalLUT: public utils::cuda::Managed {
public:
  explicit TwoDimensionalLUT():
    is_scalar_(false),
    scalar_value_(0.0),
    idx1_dim_(0),
    idx2_dim_(0),
    device_allocation_(nullptr),
    idx1_input_transitions_(nullptr),
    idx2_output_capacitances_(nullptr),
    lookup_table_(nullptr) {};  // TODO delete this empty constructor

  explicit TwoDimensionalLUT(
    LookupTableIdx idx1_dim, const SlewVal* const idx1_input_transitions, 
    LookupTableIdx idx2_dim, const CapacitanceVal* const idx2_output_capacitances, 
    const EnergyVal* const lookup_table,
    bool is_scalar = false,
    EnergyVal scalar_value = 0.0
  ) :
  is_scalar_(is_scalar),
  scalar_value_(scalar_value),
  idx1_dim_(idx1_dim),
  idx2_dim_(idx2_dim),
  device_allocation_(nullptr),
  idx1_input_transitions_(nullptr),
  idx2_output_capacitances_(nullptr),
  lookup_table_(nullptr) {
    if (is_scalar_) {
      return;
    }
    copyDeviceArrays(idx1_input_transitions, idx2_output_capacitances, lookup_table);
  }

  explicit TwoDimensionalLUT(
    LookupTableIdx idx1_dim, const SlewVal* const idx1_input_transitions, 
    LookupTableIdx idx2_dim, const CapacitanceVal* const idx2_output_capacitances, 
    const sta::FloatTable* const lookup_table,
    bool is_scalar = false,
    EnergyVal scalar_value = 0.0
  ):
  is_scalar_(is_scalar),
  scalar_value_(scalar_value),
  idx1_dim_(idx1_dim),
  idx2_dim_(idx2_dim),
  device_allocation_(nullptr),
  idx1_input_transitions_(nullptr),
  idx2_output_capacitances_(nullptr),
  lookup_table_(nullptr) {
    if (is_scalar_) {
      return;
    }
    if (lookup_table == nullptr) {
      failInvalidTable("lookup_table must not be null for non-scalar tables");
    }
    if (idx1_dim_ <= 0 || idx2_dim_ <= 0) {
      failInvalidTable("dimensions must be positive for non-scalar tables");
    }
    if (idx1_dim_ != lookup_table->size()) {
      failInvalidTable("idx1_dim does not match lookup_table row count");
    }
    std::vector<EnergyVal> tmp_table;
    for (size_t i = 0; i < lookup_table->size(); ++i) {
      const auto* row = lookup_table->at(i);
      if (row == nullptr) {
        failInvalidTable("lookup_table row must not be null");
      }
      if (idx2_dim_ != row->size()) {
        failInvalidTable("idx2_dim does not match lookup_table column count");
      }
      for (size_t j = 0; j < row->size(); ++j) {
        tmp_table.push_back(row->at(j));
      }
    }
    copyDeviceArrays(idx1_input_transitions, idx2_output_capacitances, tmp_table.data());
  }

  // for one dimensional LUT
  explicit TwoDimensionalLUT(
    LookupTableIdx idx2_dim, const CapacitanceVal* const idx2_output_capacitances, 
    const EnergyVal* const lookup_table
  ):
  is_scalar_(false),
  scalar_value_(0.0),
  idx1_dim_(1),
  idx2_dim_(idx2_dim),
  device_allocation_(nullptr),
  idx1_input_transitions_(nullptr),
  idx2_output_capacitances_(nullptr),
  lookup_table_(nullptr) {
    const SlewVal idx1_input_transitions[1] = {0.0};
    copyDeviceArrays(idx1_input_transitions, idx2_output_capacitances, lookup_table);
  }

  __device__ __host__ EnergyVal lookUpValue(SlewVal x0, CapacitanceVal y0) const {
    if (is_scalar_) {
      return scalar_value_;
    }

    const auto& x = idx1_input_transitions_;
    const auto& y = idx2_output_capacitances_;
    assert(idx1_dim_ > 0 && idx2_dim_ > 0 && x != nullptr && y != nullptr && lookup_table_ != nullptr);

    if (idx1_dim_ == 1 && idx2_dim_ == 1) {
      return lookup_table_[0];
    }

    if (idx1_dim_ == 1) {
      LookupTableIdx y1_idx = -1;
      LookupTableIdx y2_idx = -1;
      findSegment(y, idx2_dim_, y0, y1_idx, y2_idx);
      return interpolate(y0, y[y1_idx], y[y2_idx], lookup_table_[y1_idx], lookup_table_[y2_idx]);
    }

    if (idx2_dim_ == 1) {
      LookupTableIdx x1_idx = -1;
      LookupTableIdx x2_idx = -1;
      findSegment(x, idx1_dim_, x0, x1_idx, x2_idx);
      return interpolate(x0, x[x1_idx], x[x2_idx], lookup_table_[x1_idx * idx2_dim_], lookup_table_[x2_idx * idx2_dim_]);
    }

    LookupTableIdx x1_idx = -1;
    LookupTableIdx x2_idx = -1;
    findSegment(x, idx1_dim_, x0, x1_idx, x2_idx);

    LookupTableIdx y1_idx = -1;
    LookupTableIdx y2_idx = -1;
    findSegment(y, idx2_dim_, y0, y1_idx, y2_idx);

    // printf("x1_idx: %hd, x2_idx: %hd, y1_idx: %hd, y2_idx: %hd\n", x1_idx, x2_idx, y1_idx, y2_idx);
    // ------calculation------
    assert(x1_idx >= 0 && x1_idx < idx1_dim_ && x2_idx >= 0 && x2_idx < idx1_dim_ && x1_idx < x2_idx
      && y1_idx >= 0 && y1_idx < idx2_dim_ && y2_idx >= 0 && y2_idx < idx2_dim_ && y1_idx < y2_idx);
    const SlewVal x_diff = x[x2_idx] - x[x1_idx];
    const CapacitanceVal y_diff = y[y2_idx] - y[y1_idx];
    assert(x_diff != 0.0 && y_diff != 0.0);
    SlewVal x01 = (x0 - x[x1_idx]) / x_diff;
    SlewVal x20 = (x[x2_idx] - x0) / x_diff;
    SlewVal y01 = (y0 - y[y1_idx]) / y_diff;
    SlewVal y20 = (y[y2_idx] - y0) / y_diff;

    EnergyVal res = x20 * y20 * lookup_table_[x1_idx * idx2_dim_ + y1_idx] + x20 * y01 * lookup_table_[x1_idx * idx2_dim_ + y2_idx] + 
      x01 * y20 * lookup_table_[x2_idx * idx2_dim_ + y1_idx] + x01 * y01 * lookup_table_[x2_idx * idx2_dim_ + y2_idx];
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
      printf("\nindex2: \n");
      for (int i = 0; i < idx2_dim_; ++i) {
        printf("%e ", idx2_output_capacitances_[i]);
      }
      printf("\nnldm table: \n");
      for (int i = 0; i < idx1_dim_; ++i) {
        for (int j = 0; j < idx2_dim_; ++j) {
          printf("%e ", lookup_table_[i * idx2_dim_ + j]);
        }
        printf("\n");
      }
    }
  }

  ~TwoDimensionalLUT() {
    if (device_allocation_) {
      cudaFree(device_allocation_);
      device_allocation_ = nullptr;
      idx1_input_transitions_ = nullptr;
      idx2_output_capacitances_ = nullptr;
      lookup_table_ = nullptr;
      return;
    }

    cudaFree(idx1_input_transitions_);
    idx1_input_transitions_ = nullptr;
    cudaFree(idx2_output_capacitances_);
    idx2_output_capacitances_ = nullptr;
    cudaFree(lookup_table_);
    lookup_table_ = nullptr;

    // printf("Destructor of TwoDimensionalLUT\n");
  }

private:
  static void failInvalidTable(const char* message) {
    LOG_ERROR << "Invalid TwoDimensionalLUT: " << message;
  }

  static size_t alignOffset(size_t offset, size_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
  }

  template <typename AxisVal>
  static void validateStrictlyIncreasing(
    const AxisVal* axis,
    LookupTableIdx dim,
    const char* axis_name
  ) {
    for (LookupTableIdx i = 1; i < dim; ++i) {
      if (!(axis[i - 1] < axis[i])) {
        LOG_ERROR << "Invalid TwoDimensionalLUT: " << axis_name << " must be strictly increasing";
      }
    }
  }

  void copyDeviceArrays(
    const SlewVal* idx1_input_transitions,
    const CapacitanceVal* idx2_output_capacitances,
    const EnergyVal* lookup_table
  ) {
    if (idx1_dim_ <= 0 || idx2_dim_ <= 0) {
      failInvalidTable("dimensions must be positive for non-scalar tables");
    }
    if (idx1_input_transitions == nullptr) {
      failInvalidTable("idx1_input_transitions must not be null for non-scalar tables");
    }
    if (idx2_output_capacitances == nullptr) {
      failInvalidTable("idx2_output_capacitances must not be null for non-scalar tables");
    }
    if (lookup_table == nullptr) {
      failInvalidTable("lookup_table must not be null for non-scalar tables");
    }
    validateStrictlyIncreasing(idx1_input_transitions, idx1_dim_, "idx1_input_transitions");
    validateStrictlyIncreasing(idx2_output_capacitances, idx2_dim_, "idx2_output_capacitances");

    const size_t idx1_bytes = sizeof(SlewVal) * idx1_dim_;
    const size_t idx2_offset = alignOffset(idx1_bytes, alignof(CapacitanceVal));
    const size_t idx2_bytes = sizeof(CapacitanceVal) * idx2_dim_;
    const size_t table_offset = alignOffset(idx2_offset + idx2_bytes, alignof(EnergyVal));
    const size_t table_bytes = sizeof(EnergyVal) * idx1_dim_ * idx2_dim_;
    const size_t total_bytes = table_offset + table_bytes;

    std::vector<char> host_buffer(total_bytes);
    std::memcpy(host_buffer.data(), idx1_input_transitions, idx1_bytes);
    std::memcpy(host_buffer.data() + idx2_offset, idx2_output_capacitances, idx2_bytes);
    std::memcpy(host_buffer.data() + table_offset, lookup_table, table_bytes);

    CHECK_CUDA_RUNTIME(cudaMalloc(&device_allocation_, total_bytes));
    char* device_bytes = static_cast<char*>(device_allocation_);
    idx1_input_transitions_ = reinterpret_cast<SlewVal*>(device_bytes);
    idx2_output_capacitances_ = reinterpret_cast<CapacitanceVal*>(device_bytes + idx2_offset);
    lookup_table_ = reinterpret_cast<EnergyVal*>(device_bytes + table_offset);
    CHECK_CUDA_RUNTIME(cudaMemcpy(device_allocation_, host_buffer.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDA_MEM_STATS.add(CudaMemStats::Category::lut_table, total_bytes);
  }

  template <typename AxisVal>
  __host__ __device__ static void findSegment(
    const AxisVal* axis,
    LookupTableIdx dim,
    AxisVal value,
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

  template <typename AxisVal>
  __host__ __device__ static EnergyVal interpolate(
    AxisVal value,
    AxisVal lower_axis,
    AxisVal upper_axis,
    EnergyVal lower_value,
    EnergyVal upper_value
  ) {
    const AxisVal axis_diff = upper_axis - lower_axis;
    assert(axis_diff != 0.0);
    const AxisVal ratio = (value - lower_axis) / axis_diff;
    return lower_value + ratio * (upper_value - lower_value);
  }

  bool is_scalar_;
  EnergyVal scalar_value_;
  LookupTableIdx idx1_dim_;
  LookupTableIdx idx2_dim_;
  void* device_allocation_;
  SlewVal* idx1_input_transitions_;  // index1, i.e. outter index, vertical, managed memory
  CapacitanceVal* idx2_output_capacitances_;  // index2, i.e. inner index, horizontal, managed memory
  EnergyVal* lookup_table_; // managed memory
};

class TwoDimensionalLUTPair: public utils::cuda::Managed {
public:
  explicit TwoDimensionalLUTPair() :
    rise_table_(nullptr),
    fall_table_(nullptr) {
    initWhenState();
  }

  explicit TwoDimensionalLUTPair(
    TwoDimensionalLUT* fall_table,
    TwoDimensionalLUT* rise_table,
    const std::vector<VcdEventVal>& when_state
  ) :
    fall_table_(fall_table),
    rise_table_(rise_table) {
      setWhenState(when_state);
    }

  void setTable(TwoDimensionalLUT* table, RISEFALL rise_fall) {
    if (rise_fall == RISE) {
      rise_table_ = table;
    } else if (rise_fall == FALL) {
      fall_table_ = table;
    }
  }

  void setWhenState(const std::vector<VcdEventVal>& when_state, bool satisfiable = true) {
    assert(when_state.size() <= MAX_N_PIN);
    when_satisfiable_ = satisfiable;
    initWhenState();
    for (size_t idx = 0; idx < when_state.size(); ++idx) {
      when_state_[idx] = when_state[idx];
    }
  }

  __device__ bool matchPinStates(const VcdEventVal* pin_states, NPinVal n_pin) {
    if (!when_satisfiable_) {
      return false;
    }
    bool matched = true;
    for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
      if (when_state_[pin_idx] != INVALID_VCD_EVENT_VAL && when_state_[pin_idx] != ((pin_states[pin_idx] == 2 || pin_states[pin_idx] == 0) ? 0 : 1)) {
        matched = false;
        break;
      }
    }

    return matched;
  }

  __host__ __device__ EnergyVal lookUp(SlewVal tramsition_time, CapacitanceVal capacitance, RISEFALL rise_fall) const {
    assert(rise_fall == FALL || rise_fall == RISE);
    if (rise_fall == FALL) {
      assert(fall_table_ != nullptr);
      return fall_table_->lookUpValue(tramsition_time, capacitance);
    } else if (rise_fall == RISE) {
      assert(rise_table_ != nullptr);
      return rise_table_->lookUpValue(tramsition_time, capacitance);
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

  ~TwoDimensionalLUTPair() {
    delete fall_table_;
    fall_table_ = nullptr;
    delete rise_table_;
    rise_table_ = nullptr;
    // printf("Destructor of TwoDimensionalLUTPair\n");
  }

private:
  void initWhenState() {
    for (size_t idx = 0; idx < MAX_N_PIN; ++idx) {
      when_state_[idx] = INVALID_VCD_EVENT_VAL;
    }
  }

  TwoDimensionalLUT* fall_table_; // managed memory
  TwoDimensionalLUT* rise_table_; // managed memory
  bool when_satisfiable_ = true;
  VcdEventVal when_state_[MAX_N_PIN];
};

} // end of namespace utils::cuda
