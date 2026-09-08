#pragma once
#include <cassert>
#include <unordered_map>

#include "Types.hh"
#include "Defines.hh"
#include "Defines.cuh"
#include "Managed.cuh"
#include "CheckCudaRuntime.cuh"
#include "OneDimensionalLUT.cuh"
#include "TwoDimensionalLUT.cuh"
#include "CudaMemStats.hh"

namespace sta{
  class Instance;
}

namespace sta::power {
inline std::unordered_map<const LibertyCell*, PowerVal*> cell_to_leakage_powers;
inline std::unordered_map<const LibertyCell*, utils::cuda::OneDimensionalLUTPair***> cell_to_input_pin_internal_power_LUTs;
inline std::unordered_map<const LibertyCell*, NStateVal*> cell_to_n_input_pin_internal_power_LUTs_indexed_by_order;
inline std::unordered_map<const LibertyCell*, utils::cuda::OneDimensionalLUTPair***> cell_to_input_pin_internal_power_LUTs_indexed_by_order;
inline std::unordered_map<const LibertyCell*, utils::cuda::TwoDimensionalLUTPair****> cell_to_output_pin_internal_power_LUTs;
inline std::unordered_map<const LibertyCell*, NStateVal**> cell_to_n_output_pin_internal_power_LUTs_indexed_by_order;
inline std::unordered_map<const LibertyCell*, utils::cuda::TwoDimensionalLUTPair****> cell_to_output_pin_internal_power_LUTs_indexed_by_order;

struct Gate {
  const Instance *inst;  // only for host side debug
  bool is_logic;

  PowerVal *leakage_powers;  // array of leakage power values indexed by binary encoding of when states
  bool default_leakage_exists;
  PowerVal default_leakage_power_val;

  NPinVal n_pin;
  NPinVal n_input_pin;
  NPinVal n_output_pin;
  // NStateVal n_state;
  NEeventVal *pin_waveform_starts;  // array of start pointer of each pin
  NEeventVal *pin_waveform_ends;  // array of end pointer of each pin
  NEeventVal *pin_waveform_sizes;  // only on host side, for scheduling kernel
  VcdEventVal *pin_default_states;
  NPeriodVal n_cycle_per_thread;
  VoltageVal *pin_voltages;
  SlewVal *pin_rise_slews;
  SlewVal *pin_fall_slews;
  CapacitanceVal *pin_load_capacitances;
  bool* pin_is_clocks;
  // cell arc delay information, indexed by output pin, input pin, and four arc types.
  DelayVal* cell_arc_delays;

  //----------kernel scheduling----------
  NThreadVal start_thread_for_event_partition;
  NThreadVal end_thread_for_event_partition;
  NBlockVal start_block_for_event_partition;
  NBlockVal end_block_for_event_partition;

  NThreadVal start_thread_for_cycle_partition;
  NThreadVal end_thread_for_cycle_partition;
  NBlockVal start_block_for_cycle_partition;
  NBlockVal end_block_for_cycle_partition;
  // NThreadVal accu_thread_pin_count;
  //----------end of kernel scheduling----------

  //----------result----------
  PowerVal* per_tile_switching_res;
  PowerVal* per_tile_glitch_switching_res;
  PowerVal* per_tile_leakage_res;
  PowerVal* per_tile_internal_res;
  PowerVal* per_tile_glitch_internal_res;
  //----------end of result----------

  // internal power for input pins
  // 2-D array of LUT pointers, the outter loop is input pin, the inner loop is LUTs indexed by binary encoding of when states
  utils::cuda::OneDimensionalLUTPair*** input_pin_internal_power_LUTs;
  // utils::cuda::OneDimensionalLUTPair** input_pin_default_internal_power_LUTs;
  NStateVal* n_input_pin_internal_power_LUTs_indexed_by_order;
  utils::cuda::OneDimensionalLUTPair*** input_pin_internal_power_LUTs_indexed_by_order;

  // internal power for output pins
  // 3-D array of LUT pointers, the outter loop is output pin, the middle loop is related pins, and the inner loop is LUTs indexed by binary encoding of when states
  utils::cuda::TwoDimensionalLUTPair**** output_pin_internal_power_LUTs;
  // utils::cuda::TwoDimensionalLUTPair*** output_pin_default_internal_power_LUTs;
  NStateVal** n_output_pin_internal_power_LUTs_indexed_by_order;  // the outter loop is output pin, the inner loop is related pin
  utils::cuda::TwoDimensionalLUTPair**** output_pin_internal_power_LUTs_indexed_by_order;

  explicit Gate(const Instance *_inst, const LibertyCell *corner_cell, bool _is_logic,
    const std::vector<PowerVal>& _leakage_powers, bool _default_leakage_exists, PowerVal _default_leakage_power_val,
    const std::vector<utils::cuda::OneDimensionalLUTPair**>& h_input_ports_internal_power_LUTs, const std::vector<NStateVal>& h_input_ports_n_internal_power_LUTs_index_by_order, const std::vector<utils::cuda::OneDimensionalLUTPair**>& h_input_ports_internal_power_LUTs_index_by_order,
    const std::vector<utils::cuda::TwoDimensionalLUTPair***>& h_output_ports_internal_power_LUTs, const std::vector<NStateVal*>& h_output_ports_n_internal_power_LUTs_index_by_order, const std::vector<utils::cuda::TwoDimensionalLUTPair***>& h_output_ports_internal_power_LUTs_index_by_order,
    NPinVal _n_pin, NPinVal _n_input_pin, NPinVal _n_output_pin, NPinVal _output_pin_idx
  ) :
    inst(_inst),
    is_logic(_is_logic),
    default_leakage_exists(_default_leakage_exists),
    default_leakage_power_val(_default_leakage_power_val),
    n_pin(_n_pin),
    n_input_pin(_n_input_pin),
    n_output_pin(_n_output_pin),
    pin_default_states(nullptr),
    n_cycle_per_thread(0),
    cell_arc_delays(nullptr),
    start_thread_for_event_partition(0),
    end_thread_for_event_partition(0),
    start_block_for_event_partition(0),
    end_block_for_event_partition(0),
    start_thread_for_cycle_partition(0),
    end_thread_for_cycle_partition(0),
    start_block_for_cycle_partition(0),
    end_block_for_cycle_partition(0),
    per_tile_switching_res(nullptr),
    per_tile_glitch_switching_res(nullptr),
    per_tile_leakage_res(nullptr),
    per_tile_internal_res(nullptr),
    per_tile_glitch_internal_res(nullptr)
  {
    // assert(n_input_pin >= 1 && n_output_pin >= 1 && n_pin >= 1);  // that might fails
    assert(n_input_pin + n_output_pin == n_pin);
    assert((default_leakage_exists && default_leakage_power_val != INVALID_LEAKAGE_POWER_VAL) || (!default_leakage_exists && default_leakage_power_val == INVALID_LEAKAGE_POWER_VAL));

    // ---------------------------internal power of input ports------------------------------
    if (h_input_ports_internal_power_LUTs.empty()) {
      input_pin_internal_power_LUTs = nullptr;
    } else if (cell_to_input_pin_internal_power_LUTs.count(corner_cell)) {
      input_pin_internal_power_LUTs = cell_to_input_pin_internal_power_LUTs.at(corner_cell);
    } else {
      assert(h_input_ports_internal_power_LUTs.size() == n_input_pin);
      CHECK_CUDA_RUNTIME(cudaMalloc(&input_pin_internal_power_LUTs, sizeof(utils::cuda::OneDimensionalLUTPair**) * h_input_ports_internal_power_LUTs.size()));
      CHECK_CUDA_RUNTIME(cudaMemcpy(input_pin_internal_power_LUTs, h_input_ports_internal_power_LUTs.data(), sizeof(utils::cuda::OneDimensionalLUTPair**) * h_input_ports_internal_power_LUTs.size(), cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::bsim_lut_index, sizeof(utils::cuda::OneDimensionalLUTPair**) * h_input_ports_internal_power_LUTs.size());
      cell_to_input_pin_internal_power_LUTs[corner_cell] = input_pin_internal_power_LUTs;
    }
    if (h_input_ports_internal_power_LUTs_index_by_order.empty()) {
      n_input_pin_internal_power_LUTs_indexed_by_order = nullptr;
      input_pin_internal_power_LUTs_indexed_by_order = nullptr;
    } else if (cell_to_n_input_pin_internal_power_LUTs_indexed_by_order.count(corner_cell) && cell_to_input_pin_internal_power_LUTs_indexed_by_order.count(corner_cell)) {
      n_input_pin_internal_power_LUTs_indexed_by_order = cell_to_n_input_pin_internal_power_LUTs_indexed_by_order.at(corner_cell);
      input_pin_internal_power_LUTs_indexed_by_order = cell_to_input_pin_internal_power_LUTs_indexed_by_order.at(corner_cell);
    } else {
      assert(h_input_ports_n_internal_power_LUTs_index_by_order.size() == n_input_pin);
      assert(h_input_ports_internal_power_LUTs_index_by_order.size() == n_input_pin);
      CHECK_CUDA_RUNTIME(cudaMalloc(&n_input_pin_internal_power_LUTs_indexed_by_order, sizeof(NStateVal) * n_input_pin));
      CHECK_CUDA_RUNTIME(cudaMemcpy(n_input_pin_internal_power_LUTs_indexed_by_order, h_input_ports_n_internal_power_LUTs_index_by_order.data(), sizeof(NStateVal) * n_input_pin, cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::fallback_lut_index, sizeof(NStateVal) * n_input_pin);
      CHECK_CUDA_RUNTIME(cudaMalloc(&input_pin_internal_power_LUTs_indexed_by_order, sizeof(utils::cuda::OneDimensionalLUTPair**) * n_input_pin));
      CHECK_CUDA_RUNTIME(cudaMemcpy(input_pin_internal_power_LUTs_indexed_by_order, h_input_ports_internal_power_LUTs_index_by_order.data(), sizeof(utils::cuda::OneDimensionalLUTPair**) * n_input_pin, cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::fallback_lut_index, sizeof(utils::cuda::OneDimensionalLUTPair**) * n_input_pin);
      cell_to_n_input_pin_internal_power_LUTs_indexed_by_order[corner_cell] = n_input_pin_internal_power_LUTs_indexed_by_order;
      cell_to_input_pin_internal_power_LUTs_indexed_by_order[corner_cell] = input_pin_internal_power_LUTs_indexed_by_order;
    }
    // ---------------------------end of internal power of input ports------------------------------

    // ---------------------------internal power of output ports------------------------------
    if (h_output_ports_internal_power_LUTs.empty()) {
      output_pin_internal_power_LUTs = nullptr;
    } else if (cell_to_output_pin_internal_power_LUTs.count(corner_cell)) {
      output_pin_internal_power_LUTs = cell_to_output_pin_internal_power_LUTs.at(corner_cell);
    } else {
      assert(h_output_ports_internal_power_LUTs.size() == n_output_pin);
      CHECK_CUDA_RUNTIME(cudaMalloc(&output_pin_internal_power_LUTs, sizeof(utils::cuda::TwoDimensionalLUTPair***) * h_output_ports_internal_power_LUTs.size()));
      CHECK_CUDA_RUNTIME(cudaMemcpy(output_pin_internal_power_LUTs, h_output_ports_internal_power_LUTs.data(), sizeof(utils::cuda::TwoDimensionalLUTPair***) * h_output_ports_internal_power_LUTs.size(), cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::bsim_lut_index, sizeof(utils::cuda::TwoDimensionalLUTPair***) * h_output_ports_internal_power_LUTs.size());
      cell_to_output_pin_internal_power_LUTs[corner_cell] = output_pin_internal_power_LUTs;
    }
    if (h_output_ports_internal_power_LUTs_index_by_order.empty()) {
      n_output_pin_internal_power_LUTs_indexed_by_order = nullptr;
      output_pin_internal_power_LUTs_indexed_by_order = nullptr;
    } else if (cell_to_n_output_pin_internal_power_LUTs_indexed_by_order.count(corner_cell) && cell_to_output_pin_internal_power_LUTs_indexed_by_order.count(corner_cell)) {
      n_output_pin_internal_power_LUTs_indexed_by_order = cell_to_n_output_pin_internal_power_LUTs_indexed_by_order.at(corner_cell);
      output_pin_internal_power_LUTs_indexed_by_order = cell_to_output_pin_internal_power_LUTs_indexed_by_order.at(corner_cell);
    } else {
      assert(h_output_ports_n_internal_power_LUTs_index_by_order.size() == n_output_pin);
      assert(h_output_ports_internal_power_LUTs_index_by_order.size() == n_output_pin);
      CHECK_CUDA_RUNTIME(cudaMalloc(&n_output_pin_internal_power_LUTs_indexed_by_order, sizeof(NStateVal*) * n_output_pin));
      CHECK_CUDA_RUNTIME(cudaMemcpy(n_output_pin_internal_power_LUTs_indexed_by_order, h_output_ports_n_internal_power_LUTs_index_by_order.data(), sizeof(NStateVal*) * n_output_pin, cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::fallback_lut_index, sizeof(NStateVal*) * n_output_pin);
      CHECK_CUDA_RUNTIME(cudaMalloc(&output_pin_internal_power_LUTs_indexed_by_order, sizeof(utils::cuda::TwoDimensionalLUTPair***) * n_output_pin));
      CHECK_CUDA_RUNTIME(cudaMemcpy(output_pin_internal_power_LUTs_indexed_by_order, h_output_ports_internal_power_LUTs_index_by_order.data(), sizeof(utils::cuda::TwoDimensionalLUTPair***) * n_output_pin, cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::fallback_lut_index, sizeof(utils::cuda::TwoDimensionalLUTPair***) * n_output_pin);
      cell_to_n_output_pin_internal_power_LUTs_indexed_by_order[corner_cell] = n_output_pin_internal_power_LUTs_indexed_by_order;
      cell_to_output_pin_internal_power_LUTs_indexed_by_order[corner_cell] = output_pin_internal_power_LUTs_indexed_by_order;
    }
    // ---------------------------end of internal power of output ports------------------------------

    if (_leakage_powers.empty()) {
      leakage_powers = nullptr;
    } else if (cell_to_leakage_powers.count(corner_cell)) {
      leakage_powers = cell_to_leakage_powers.at(corner_cell);
    } else {
      CHECK_CUDA_RUNTIME(cudaMalloc(&leakage_powers, sizeof(PowerVal) * _leakage_powers.size()));
      CHECK_CUDA_RUNTIME(cudaMemcpy(leakage_powers, _leakage_powers.data(), sizeof(PowerVal) * _leakage_powers.size(), cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(utils::cuda::CudaMemStats::Category::bsim_leakage_index, sizeof(PowerVal) * _leakage_powers.size());
      cell_to_leakage_powers[corner_cell] = leakage_powers;
    }
    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_waveform_starts, sizeof(NEeventVal) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_waveform_starts, _pin_waveform_starts, sizeof(NEeventVal) * n_pin, cudaMemcpyHostToDevice));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_waveform_ends, sizeof(NEeventVal) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_waveform_ends, _pin_waveform_ends, sizeof(NEeventVal) * n_pin, cudaMemcpyHostToDevice));

    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_voltages, sizeof(VoltageVal) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_voltages, _pin_voltages, sizeof(VoltageVal) * n_pin, cudaMemcpyHostToDevice));

    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_rise_slews, sizeof(SlewVal) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_rise_slews, _pin_rise_slews, sizeof(SlewVal) * n_pin, cudaMemcpyHostToDevice));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_fall_slews, sizeof(SlewVal) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_fall_slews, _pin_fall_slews, sizeof(SlewVal) * n_pin, cudaMemcpyHostToDevice));

    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_load_capacitances, sizeof(CapacitanceVal) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_load_capacitances, _pin_load_capacitances, sizeof(CapacitanceVal) * n_pin, cudaMemcpyHostToDevice));

    // CHECK_CUDA_RUNTIME(cudaMalloc(&pin_is_clocks, sizeof(bool) * n_pin));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_is_clocks, _pin_is_clocks, sizeof(bool) * n_pin, cudaMemcpyHostToDevice));

    pin_waveform_sizes = new NEeventVal[n_pin];
    // for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    //   pin_waveform_sizes[pin_idx] = _pin_waveform_ends[pin_idx] - _pin_waveform_starts[pin_idx];
    // }
  }

  void setAllPinsBlockThreadRange(
    NThreadVal _start_thread_for_event_partition, NThreadVal _end_thread_for_event_partition,
    NBlockVal _start_block_for_event_partition, NBlockVal _end_block_for_event_partition, 
    NThreadVal _start_thread_for_cycle_partition, NThreadVal _end_thread_for_cycle_partition,
    NBlockVal _start_block_for_cycle_partition, NBlockVal _end_block_for_cycle_partition
    // NThreadVal _accu_thread_pin_count
  ) {
    start_thread_for_event_partition = _start_thread_for_event_partition;
    end_thread_for_event_partition = _end_thread_for_event_partition;
    start_block_for_event_partition = _start_block_for_event_partition;
    end_block_for_event_partition = _end_block_for_event_partition;
    start_thread_for_cycle_partition = _start_thread_for_cycle_partition;
    end_thread_for_cycle_partition = _end_thread_for_cycle_partition;
    start_block_for_cycle_partition = _start_block_for_cycle_partition;
    end_block_for_cycle_partition = _end_block_for_cycle_partition;
    // accu_thread_pin_count = _accu_thread_pin_count;
    // CHECK_CUDA_RUNTIME(cudaFree(per_tile_leakage_res));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&per_tile_leakage_res, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaMemset(per_tile_leakage_res, 0, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaFree(per_tile_internal_res));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&per_tile_internal_res, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaMemset(per_tile_internal_res, 0, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaFree(per_tile_glitch_internal_res));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&per_tile_glitch_internal_res, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaMemset(per_tile_glitch_internal_res, 0, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaFree(per_tile_switching_res));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&per_tile_switching_res, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaMemset(per_tile_switching_res, 0, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaFree(per_tile_glitch_switching_res));
    // CHECK_CUDA_RUNTIME(cudaMalloc(&per_tile_glitch_switching_res, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
    // CHECK_CUDA_RUNTIME(cudaMemset(per_tile_glitch_switching_res, 0, sizeof(PowerVal) * (all_pins_end_thread - all_pins_start_thread)));
  }

  __device__ NThreadVal getStartThreadIdx(char cuda_thread_partition_basis) const {
    if (cuda_thread_partition_basis == 'e') {
      return start_thread_for_event_partition;
    } else if (cuda_thread_partition_basis == 'c') {
      return start_thread_for_cycle_partition;
    } else {
      assert(false);
    }
  }

  __device__ NThreadVal getEndThreadIdx(char cuda_thread_partition_basis) const {
    if (cuda_thread_partition_basis == 'e') {
      return end_thread_for_event_partition;
    } else if (cuda_thread_partition_basis == 'c') {
      return end_thread_for_cycle_partition;
    } else {
      assert(false);
    }
  }

  __device__ NBlockVal getStartBlockIdx(char cuda_thread_partition_basis) const {
    if (cuda_thread_partition_basis == 'e') {
      return start_block_for_event_partition;
    } else if (cuda_thread_partition_basis == 'c') {
      return start_block_for_cycle_partition;
    } else {
      assert(false);
    }
  }

  __device__ NBlockVal getEndBlockIdx(char cuda_thread_partition_basis) const {
    if (cuda_thread_partition_basis == 'e') {
      return end_block_for_event_partition;
    } else if (cuda_thread_partition_basis == 'c') {
      return end_block_for_cycle_partition;
    } else {
      assert(false);
    }
  }

  void setPerTileResultPointer(
    PowerVal* global_per_tile_leakage_res,
    PowerVal* global_per_tile_internal_res,
    PowerVal* global_per_tile_glitch_internal_res,
    PowerVal* global_per_tile_switching_res,
    PowerVal* global_per_tile_glitch_switching_res,
    size_t cur_gate_power_res_offset
  ) {
    per_tile_leakage_res = global_per_tile_leakage_res + cur_gate_power_res_offset;
    per_tile_internal_res = global_per_tile_internal_res + cur_gate_power_res_offset;
    per_tile_glitch_internal_res = global_per_tile_glitch_internal_res + cur_gate_power_res_offset;
    per_tile_switching_res = global_per_tile_switching_res + cur_gate_power_res_offset;
    per_tile_glitch_switching_res = global_per_tile_glitch_switching_res + cur_gate_power_res_offset;
  }

  void renewWaveformPointers(
    NEeventVal *d_global_pin_waveform_starts, NEeventVal *d_global_pin_waveform_ends,
    const std::vector<NEeventVal>& h_global_pin_waveform_starts_vec, const std::vector<NEeventVal>& h_global_pin_waveform_ends_vec, 
    size_t accu_pin_count
  ) {
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_waveform_starts, _pin_waveform_starts, sizeof(NEeventVal) * n_pin, cudaMemcpyHostToDevice));
    // CHECK_CUDA_RUNTIME(cudaMemcpy(pin_waveform_ends, _pin_waveform_ends, sizeof(NEeventVal) * n_pin, cudaMemcpyHostToDevice));

    pin_waveform_starts = d_global_pin_waveform_starts + accu_pin_count;
    pin_waveform_ends = d_global_pin_waveform_ends + accu_pin_count;
    for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
      pin_waveform_sizes[pin_idx] = h_global_pin_waveform_ends_vec.at(pin_idx + accu_pin_count) - h_global_pin_waveform_starts_vec.at(pin_idx + accu_pin_count);
    }
  }

  void setGateInfoByGlobalPointer(NEeventVal* d_global_pin_waveform_starts, NEeventVal* d_global_pin_waveform_ends, 
    const std::vector<NEeventVal>& h_global_pin_waveform_starts_vec, const std::vector<NEeventVal>& h_global_pin_waveform_ends_vec, 
    VcdEventVal* d_global_pin_default_states,
    VoltageVal* d_global_pin_voltages,
    SlewVal* d_global_pin_rise_slews, SlewVal* d_global_pin_fall_slews,
    CapacitanceVal* d_global_pin_load_capacitances, bool* d_global_pin_is_clocks,
    DelayVal* d_global_cell_arc_delays,
    size_t accu_pin_count,
    size_t accu_cell_arc_delay_count) 
  {
    renewWaveformPointers(d_global_pin_waveform_starts, d_global_pin_waveform_ends, h_global_pin_waveform_starts_vec, h_global_pin_waveform_ends_vec, accu_pin_count);
    pin_default_states = d_global_pin_default_states + accu_pin_count;
    pin_voltages = d_global_pin_voltages + accu_pin_count;
    pin_rise_slews = d_global_pin_rise_slews + accu_pin_count;
    pin_fall_slews = d_global_pin_fall_slews + accu_pin_count;
    pin_load_capacitances = d_global_pin_load_capacitances + accu_pin_count;
    pin_is_clocks = d_global_pin_is_clocks + accu_pin_count;
    cell_arc_delays = d_global_cell_arc_delays == nullptr ? nullptr : d_global_cell_arc_delays + accu_cell_arc_delay_count;
  }

  __device__ DelayVal cellArcDelay(NPinVal output_pin_local_idx, NPinVal input_pin_idx, RISEFALL from_rf, RISEFALL to_rf) const {
    const size_t arc_idx = 2 * (1 - from_rf) + (1 - to_rf);
    const size_t pin_pair_idx = static_cast<size_t>(output_pin_local_idx) * n_input_pin + input_pin_idx;
    return cell_arc_delays[pin_pair_idx * 4 + arc_idx];
  }

  void setNCyclePerThread(const NPeriodVal _n_cycle_per_thread) {
    n_cycle_per_thread = _n_cycle_per_thread;
  }

  void setNCyclePerThreadByEventCount(
    const NPeriodVal n_cycle,
    const NThreadVal threads_target,   // 稀疏时希望大约这么多线程, n_thread_per_block * 4
    const NEeventVal E_target
  ) {
    NEeventVal n_event = std::accumulate(pin_waveform_sizes, pin_waveform_sizes + n_pin, static_cast<NEeventVal>(0));
    if (n_cycle <= 0 || n_event < 0) {
      LOG_ERROR << "n_cycle <= 0 || n_event < 0, n_cycle: " << n_cycle << " n_event: " << n_event;
    }

    NPeriodVal n_cycle_target = (n_cycle + threads_target - 1) / threads_target;
    n_cycle_target = std::max(n_cycle_target, static_cast<NPeriodVal>(1)); // 防止极端

    double rho = static_cast<double>(n_event) / static_cast<double>(n_cycle);
    double rho_floor = static_cast<double>(E_target) / static_cast<double>(n_cycle_target); // 这就是 cycle 基线 B
    double rho_eff = rho + rho_floor;

    NPeriodVal n = static_cast<NPeriodVal>(llround(static_cast<double>(E_target) / rho_eff));
    n = std::min(n, n_cycle);
    n = std::max(n, static_cast<NPeriodVal>(1));

    n_cycle_per_thread = n;
  }

  __device__ EnergyVal getSingleRiseTransitionEnergy(NPinVal output_pin_idx) const {
    return pin_load_capacitances[output_pin_idx] * pin_voltages[output_pin_idx] * pin_voltages[output_pin_idx];
  }

  __device__ SlewVal getPinSlew(NPinVal pin_idx, RISEFALL rise_fall) const {
    return rise_fall == RISE ? pin_rise_slews[pin_idx] : pin_fall_slews[pin_idx];
  }

  __host__ ~Gate() {  // TODO add free lut pointers
    delete[] pin_waveform_sizes;
    pin_waveform_sizes = nullptr;
  }
};

} // end of namespace sta::power
