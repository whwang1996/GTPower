#include <vector>
#include <cstring>  // strcmp
#include <string>
#include <numeric>  // std::accumulate
#include <fstream>  // std::ofstream
#include <thread>
#include <omp.h>
#include <random>   // std::random_device, std::mt19937
#include <array>
#include <algorithm>
#include <memory>

#include "CudaPower.hh"

#include "Corner.hh"
#include "Liberty.hh"
#include "LeakagePower.hh"
#include "InternalPower.hh"
#include "PortDirection.hh"
#include "GraphDelayCalc.hh"
#include "Vcd.hh"
#include "PowerUtils.hh"
#include "Log.hh"
#include "Types.hh"
#include "Types.cuh"
#include "Graph.hh"
#include "GlobalConfig.hh"
#include "CheckCudaRuntime.cuh"
#include "DeviceSetting.cuh"
#include "Managed.cuh"
#include "Gate.cuh"
#include "Event.cuh"
#include "Defines.hh"
#include "CudaUtils.cuh"
#include "CudaPowerUtils.cuh"
#include "Defines.cuh"
#include "OneDimensionalLUT.cuh"
#include "TwoDimensionalLUT.cuh"
#include "VcdHelper.hh"
#include "ScopedTimer.hh"
#include "ThreadPowerResult.cuh"
#include "CudaPowerThreadAllocStats.hh"
#include "CudaMemStats.hh"

using namespace utils::cuda;
using namespace utils::cuda::power;

using CudaMemCategory = utils::cuda::CudaMemStats::Category;

// #define DISABLE_BSIM

namespace sta {
namespace power {
__device__ __forceinline__ VcdEventVal
getPinDefaultState(const Gate& gate, NPinVal pin_idx)
{
  if (gate.pin_default_states == nullptr) {
    return 2;
  }

  const VcdEventVal pin_default_state = gate.pin_default_states[pin_idx];
  return pin_default_state == 0 || pin_default_state == 1
    ? pin_default_state
    : 2;
}

__device__ PowerVal 
findLeakageVal(
  const Gate* gate, const Gate& sh_cur_gate,
  const VcdEventVal* pin_states
) {
  if (gate->leakage_powers != nullptr) {  // has leakage power table, n_pin <= G_CONFIG.nums.max_n_pin_for_leakage_power
#ifndef DISABLE_BSIM
    return gate->leakage_powers[getPinStatesIndex(pin_states, sh_cur_gate.n_pin)];
#else
    NStateVal n_state = 1 << sh_cur_gate.n_pin;
    for (NStateVal i_state = 0; i_state < n_state; ++i_state) {
      if (i_state == getPinStatesIndex(pin_states, sh_cur_gate.n_pin)) {
        return gate->leakage_powers[i_state];
      }
    }
#endif
  } else if (gate->default_leakage_exists) {  // try to use default leakage power value instead
    return gate->default_leakage_power_val;
  } else {
    return 0.0;
  }
}

__device__ void
calculatePerCycleLeakagePower(
  const Gate* gate, 
  PowerVal leakage_power_val,
  VcdEventTime prev_time,
  VcdEventTime cur_time,
  EventTimeVal vcd_time_scale,
  PeriodVal clk_period,
  PowerVal *per_cycle_leakage_powers_
)
{
  NPeriodVal cur_cycle_idx = clkedWaveformIdx(prev_time, vcd_time_scale, clk_period);
  while (cur_cycle_idx * clk_period <= cur_time * vcd_time_scale) {
    EventTimeVal left = my_max(cur_cycle_idx * clk_period, prev_time * vcd_time_scale);
    EventTimeVal right = my_min((cur_cycle_idx + 1) * clk_period, cur_time * vcd_time_scale);
    EventTimeVal duration = right - left;
    atomicAdd(&per_cycle_leakage_powers_[cur_cycle_idx], leakage_power_val * duration / clk_period);

    ++cur_cycle_idx;
  }
}

__device__ EnergyVal
getInputPinInternalEnergyVal(
  const Gate* gate, const Gate& sh_cur_gate,
  const VcdEventVal* pin_states,
  NPinVal toggle_pin_idx,
  RISEFALL rise_fall
) {
#ifndef DISABLE_BSIM
  const NStateVal state_idx = getPinStatesIndex(pin_states, sh_cur_gate.n_pin);
  // printf("toggle_pin_idx: %hd, rise_fall: %s slew: %e\n", toggle_pin_idx, rise_fall == RISE ? "rise" : "fall", gate->getPinSlew(toggle_pin_idx, rise_fall));
  if (gate->input_pin_internal_power_LUTs[toggle_pin_idx] != nullptr && gate->input_pin_internal_power_LUTs[toggle_pin_idx][state_idx] != nullptr) {  // try to use lut indexed by state
    return gate->input_pin_internal_power_LUTs[toggle_pin_idx][state_idx]->lookUp(sh_cur_gate.getPinSlew(toggle_pin_idx, rise_fall), rise_fall);
  }
#endif
  // then try to iterate all luts
  for (NStateVal lut_idx = 0; lut_idx < gate->n_input_pin_internal_power_LUTs_indexed_by_order[toggle_pin_idx]; ++lut_idx) {
    if (gate->input_pin_internal_power_LUTs_indexed_by_order[toggle_pin_idx][lut_idx]->matchPinStates(pin_states, gate->n_pin)) {
      return gate->input_pin_internal_power_LUTs_indexed_by_order[toggle_pin_idx][lut_idx]->lookUp(sh_cur_gate.getPinSlew(toggle_pin_idx, rise_fall), rise_fall);
    }
  }

  return 0.0;
}

__device__ EnergyVal
getDefaultOutputPinInternalEnergyVal(
  const Gate* gate, const Gate& sh_cur_gate,
  NPinVal toggle_pin_idx,
  RISEFALL to_rf
) {
  const NPinVal output_pin_local_idx = toggle_pin_idx - sh_cur_gate.n_input_pin;
  int n_table = 0;
  EnergyVal total_energy = 0.0;

  if (gate->n_output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx] != nullptr) {
    for (NPinVal related_pin_idx = 0; related_pin_idx < sh_cur_gate.n_input_pin; ++related_pin_idx) {
      for (NStateVal lut_idx = 0; lut_idx < gate->n_output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx][related_pin_idx]; ++lut_idx) {
        for (RISEFALL from_rf = RISE; from_rf <= FALL; from_rf=(RISEFALL)(from_rf + 1)) {
          if (isValidDelay(gate->cellArcDelay(output_pin_local_idx, related_pin_idx, from_rf, to_rf)) 
            && isValidSlew(sh_cur_gate.getPinSlew(related_pin_idx, from_rf))) {
            SlewVal input_slew = sh_cur_gate.getPinSlew(related_pin_idx, from_rf);
            EnergyVal cur_table_energy = gate->output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx][related_pin_idx][lut_idx]->lookUp(input_slew, sh_cur_gate.pin_load_capacitances[toggle_pin_idx], to_rf);
            total_energy += cur_table_energy;
            ++n_table;
          }
        }
      }
    }
  }

  if (n_table == 0) {
    return 0;
  } else {
    return total_energy / n_table;
  }
}

__device__ EnergyVal
getOutputPinInternalEnergyVal(
  const Gate* gate, const Gate& sh_cur_gate,
  const Event *events,
  VcdEventTime cur_time,
  NPinVal toggle_pin_idx,
  RISEFALL to_rf,
  const VcdEventTime max_time,
  EventTimeVal vcd_time_scale,
  PeriodVal clk_period,
  const VcdEventVal* prev_pin_states
) {
  const NPinVal output_pin_local_idx = toggle_pin_idx - sh_cur_gate.n_input_pin;
  // ----------find related pin-----------
  VcdEventTime min_diff_time = max_time;
  NPinVal related_pin_idx = INVALID_PIN_IDX;
  NEeventVal related_pin_event_idx = INVALID_WAVEFORM_PTR;
  for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_input_pin; ++pin_idx) {
    if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR) {  // no events of pin
      continue;
    }
    NEeventVal cur_pin_related_event_idx = my_max(
      sh_cur_gate.pin_waveform_starts[pin_idx],
      getEventIdxByTime(events, sh_cur_gate.pin_waveform_starts[pin_idx], sh_cur_gate.pin_waveform_ends[pin_idx], cur_time, false) - 1 // minus one to get the larsest event idx that of which the time is smaller than cur_time
    );

    NEeventVal cur_pin_related_prev_event_idx = my_max(
      sh_cur_gate.pin_waveform_starts[pin_idx],
      cur_pin_related_event_idx - 1
    );
    RISEFALL from_rf = getRiseFallEdge(events[cur_pin_related_prev_event_idx].val, events[cur_pin_related_event_idx].val);
    DelayVal cur_cell_arc_delay = gate->cellArcDelay(output_pin_local_idx, pin_idx, from_rf, to_rf);
    // printf("cur_time: %lld, pin_idx: %hd, sh_cur_gate.pin_waveform_starts[pin_idx]: %lld, cur_pin_related_event_idx: %lld, from_rf: %d, to_rf: %d, cur_cell_arc_delay: %e\n", cur_time, pin_idx, sh_cur_gate.pin_waveform_starts[pin_idx], cur_pin_related_event_idx, from_rf, to_rf, cur_cell_arc_delay);
    if (!isValidDelay(cur_cell_arc_delay)) {
      continue;
    }

    const double cur_cell_arc_delay_in_ticks = static_cast<double>(cur_cell_arc_delay) / static_cast<double>(vcd_time_scale);
    const VcdEventTime cur_cell_arc_delay_ticks = static_cast<VcdEventTime>(
      cur_cell_arc_delay_in_ticks >= 0.0 ? cur_cell_arc_delay_in_ticks + 0.5 : cur_cell_arc_delay_in_ticks - 0.5);
    const VcdEventTime cur_pin_arrival_time = events[cur_pin_related_event_idx].time + cur_cell_arc_delay_ticks;
    const VcdEventTime cur_pin_time_delta = cur_time - cur_pin_arrival_time;
    const VcdEventTime cur_pin_diff_time = cur_pin_time_delta >= 0 ? cur_pin_time_delta : -cur_pin_time_delta;
    if (sh_cur_gate.is_logic && clkedWaveformIdx(events[cur_pin_related_event_idx].time, vcd_time_scale, clk_period) != clkedWaveformIdx(cur_time, vcd_time_scale, clk_period)) {
      continue;
    }

    const VcdEventTime tie_diff = cur_pin_diff_time >= min_diff_time
      ? cur_pin_diff_time - min_diff_time
      : min_diff_time - cur_pin_diff_time;
    const bool near_tie_with_same_input_time = related_pin_idx != INVALID_PIN_IDX
      && events[cur_pin_related_event_idx].time == events[related_pin_event_idx].time
      && tie_diff <= 1;
    if (cur_pin_diff_time < min_diff_time 
        || (near_tie_with_same_input_time && cur_pin_related_event_idx > related_pin_event_idx)) {
      min_diff_time = cur_pin_diff_time;
      related_pin_idx = pin_idx;
      related_pin_event_idx = cur_pin_related_event_idx;
    }
  }
  // printf("toggle_pin_idx: %hd, related_pin_idx: %hd, min_diff_time: %e\n", toggle_pin_idx, related_pin_idx, min_diff_time);
  if (related_pin_idx == INVALID_PIN_IDX) {
    return getDefaultOutputPinInternalEnergyVal(gate, sh_cur_gate, toggle_pin_idx, to_rf);
  }

  NEeventVal related_pin_prev_event_idx = my_max(
    sh_cur_gate.pin_waveform_starts[related_pin_idx],
    related_pin_event_idx - 1
  );
#ifndef DISABLE_BSIM
  const NStateVal state_idx = getPinStatesIndex(prev_pin_states, sh_cur_gate.n_pin);
#endif
  // ----------end of find related pin-----------

  RISEFALL from_rf = getRiseFallEdge(events[related_pin_prev_event_idx].val, events[related_pin_event_idx].val);
  SlewVal input_slew = sh_cur_gate.getPinSlew(related_pin_idx, from_rf);
  // printf("getOutputPinInternalEnergyVal, toggle_pin_idx: %hd, related_pin_idx: %hd, input_slew: %e, load_cap: %e, gate->output_pin_internal_power_LUTs[output_pin_local_idx]: %s, gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx]: %s, gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx][state_idx]: %s \n", 
  //   toggle_pin_idx, related_pin_idx, input_slew, gate->pin_load_capacitances[toggle_pin_idx],
  //   gate->output_pin_internal_power_LUTs[output_pin_local_idx] ? "exist" : "not exist", gate->output_pin_internal_power_LUTs[output_pin_local_idx] && gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx] ? "exist" : "not exist", 
  //   gate->output_pin_internal_power_LUTs[output_pin_local_idx] && gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx] && gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx][state_idx] ? "exist" : "not exist");
#ifndef DISABLE_BSIM
  if (gate->output_pin_internal_power_LUTs[output_pin_local_idx] != nullptr && gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx] != nullptr 
      && gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx][state_idx] != nullptr) {
    return gate->output_pin_internal_power_LUTs[output_pin_local_idx][related_pin_idx][state_idx]->lookUp(input_slew, sh_cur_gate.pin_load_capacitances[toggle_pin_idx], to_rf);
  } 
#endif
  // then try to iterate all luts
  if (gate->n_output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx] != nullptr) {
    for (NStateVal lut_idx = 0; lut_idx < gate->n_output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx][related_pin_idx]; ++lut_idx) {
      if (gate->output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx][related_pin_idx][lut_idx]->matchPinStates(prev_pin_states, sh_cur_gate.n_pin)) {
        return gate->output_pin_internal_power_LUTs_indexed_by_order[output_pin_local_idx][related_pin_idx][lut_idx]->lookUp(input_slew, sh_cur_gate.pin_load_capacitances[toggle_pin_idx], to_rf);
      }
    }
  }
  
  return getDefaultOutputPinInternalEnergyVal(gate, sh_cur_gate, toggle_pin_idx, to_rf);
}

__device__ NToggleVal
getGlitchScalingRatioClockCycleBasedOnDeviceOnly(
  const Gate* cur_gate, const Gate& sh_cur_gate,
  const Event *events,
  NPeriodVal* pin_cur_period_idxes, NEeventInOnePeriodVal* pin_n_event_in_cur_period,
  NEeventVal cur_event_idx, NPinVal pin_idx,
  PeriodVal clk_period, EventTimeVal vcd_time_scale
) {
  if (cur_gate->pin_is_clocks[pin_idx]) {
    return INVALID_N_TOGGLE_VAL;
  }

  NPeriodVal cur_event_period_idx = clkedWaveformIdx(events[cur_event_idx].time, vcd_time_scale, clk_period);
  if (cur_event_period_idx == pin_cur_period_idxes[pin_idx]) {
    pin_n_event_in_cur_period[pin_idx] += 1;
  } else {
    pin_cur_period_idxes[pin_idx] = cur_event_period_idx;
    pin_n_event_in_cur_period[pin_idx] = 1;
  }

  VcdEventTime prev_glitch_pulse_width = INVALID_PULSE_WIDTH;
  if (cur_event_idx - 1 >= sh_cur_gate.pin_waveform_starts[pin_idx] && cur_event_period_idx == clkedWaveformIdx(events[cur_event_idx - 1].time, vcd_time_scale, clk_period)) {
    prev_glitch_pulse_width = events[cur_event_idx].time - events[cur_event_idx - 1].time;
    // printf("cur_event_idx: %lld, cur_event_period_idx: %lld, events[cur_event_idx].time: %lld, events[cur_event_idx - 1].time: %lld,  prev event is glitch: %s\n", 
    //   cur_event_idx, cur_event_period_idx, events[cur_event_idx].time, events[cur_event_idx - 1].time, events[cur_event_idx - 1].is_glitch ? "true" : "false");
  }

  VcdEventTime next_glitch_pulse_width = INVALID_PULSE_WIDTH;
  if (cur_event_idx + 1 < sh_cur_gate.pin_waveform_ends[pin_idx] && cur_event_period_idx == clkedWaveformIdx(events[cur_event_idx + 1].time, vcd_time_scale, clk_period)) {  // next event
    if (cur_event_idx + 2 < sh_cur_gate.pin_waveform_ends[pin_idx] && cur_event_period_idx == clkedWaveformIdx(events[cur_event_idx + 2].time, vcd_time_scale, clk_period)
        || pin_n_event_in_cur_period[pin_idx] % 2 == 1) {
      next_glitch_pulse_width = events[cur_event_idx + 1].time - events[cur_event_idx].time;
    }
  } else if (pin_n_event_in_cur_period[pin_idx] % 2 == 1) {  // is the last event of current period
    return INVALID_N_TOGGLE_VAL;
  }

  VcdEventTime selected_glitch_pulse_width = my_max(prev_glitch_pulse_width, next_glitch_pulse_width);
  if (selected_glitch_pulse_width != INVALID_PULSE_WIDTH) {
    const SlewVal sum_slew = sh_cur_gate.pin_rise_slews[pin_idx] + sh_cur_gate.pin_fall_slews[pin_idx];
    NToggleVal scaling_ratio = getTimeBasedGlitchScalingRatio(selected_glitch_pulse_width * vcd_time_scale, sum_slew);
    return scaling_ratio;
  } else {
    return INVALID_N_TOGGLE_VAL;
  }
}

__global__ static void kernel1AllPowerCalculationTimeRangePartitionedByCycleUnit (
    char cuda_thread_partition_basis, const int n_event_per_thread, const int n_cycle_per_thread, const int n_thread_per_block, 
    const NPeriodVal interval_start_period_idx, const NPeriodVal interval_end_period_idx, const NEeventVal vcd_time_unit_per_cycle, const VcdEventTime max_time, 
    PeriodVal clk_period, EventTimeVal vcd_time_scale,
    Gate *gates, const Event *events, const NGateVal *block_corr_gate_idxes,
    // Return Values
    PowerVal *per_cycle_leakage_powers, PowerVal *per_cycle_internal_powers, PowerVal *per_cycle_glitch_internal_powers,
    PowerVal *per_cycle_switching_powers, PowerVal *per_cycle_glitch_switching_powers,
    int power_res_length_for_each_gate
  ) {
  // -----------------preparing------------------
  const NThreadVal thread_idx_in_block = threadIdx.x;
  const NBlockVal idx_of_block = blockIdx.x;
  const NGateVal cur_gate_idx = block_corr_gate_idxes[idx_of_block];
  const Gate* cur_gate = &gates[cur_gate_idx];
  const Gate& sh_cur_gate = gates[cur_gate_idx];

  assert(idx_of_block < cur_gate->getEndBlockIdx(cuda_thread_partition_basis));
  const NThreadVal thread_idx_in_gate = (idx_of_block - cur_gate->getStartBlockIdx(cuda_thread_partition_basis)) * n_thread_per_block + thread_idx_in_block;
  NPeriodVal cur_thread_start_period_idx = -1, cur_thread_end_period_idx = -1;
  if (cuda_thread_partition_basis == 'e') {
    const NEeventVal cur_thread_start_event_count = thread_idx_in_gate * n_event_per_thread;
    const NEeventVal cur_thread_end_event_count = (thread_idx_in_gate + 1) * n_event_per_thread;
    cur_thread_start_period_idx = getCycleIndexByAccuEventCount(cur_gate, events, cur_thread_start_event_count, interval_start_period_idx, interval_end_period_idx, vcd_time_unit_per_cycle);
    cur_thread_end_period_idx = getCycleIndexByAccuEventCount(cur_gate, events, cur_thread_end_event_count, interval_start_period_idx, interval_end_period_idx, vcd_time_unit_per_cycle);
  } else if (cuda_thread_partition_basis == 'c') {
    cur_thread_start_period_idx = thread_idx_in_gate * sh_cur_gate.n_cycle_per_thread + interval_start_period_idx;
    cur_thread_end_period_idx = (thread_idx_in_gate + 1) * sh_cur_gate.n_cycle_per_thread + interval_start_period_idx;
  } else {
    assert(false);
  }

  const VcdEventTime cur_thread_start_time = cur_thread_start_period_idx * vcd_time_unit_per_cycle;
  const VcdEventTime cur_thread_end_time = cur_thread_end_period_idx * vcd_time_unit_per_cycle + (cur_thread_end_period_idx == interval_end_period_idx ? 1 : 0);  // add 1 to cover the event occurs on the last time
  // [cur_thread_start_time, cur_thread_end_time)
  if (thread_idx_in_gate >= (cur_gate->getEndThreadIdx(cuda_thread_partition_basis) - cur_gate->getStartThreadIdx(cuda_thread_partition_basis)) || cur_thread_start_period_idx == cur_thread_end_period_idx) {
    return;
  }
  if (cur_thread_start_period_idx == cur_thread_end_period_idx) {
    printf("ERROR: cur_gate_idx: %lld thread_idx_in_gate: %lld, cur_thread_start_period_idx: %lld == cur_thread_end_period_idx: %lld\n", cur_gate_idx, thread_idx_in_gate, cur_thread_start_period_idx, cur_thread_end_period_idx);
    assert(false);
  }
  // printf("cur_gate_idx: %lld, cur_thread_start_event_count: %lld, cur_thread_end_event_count: %lld, cur_thread_start_time: %lld, cur_thread_end_time: %lld, interval_start_period_idx: %lld, cur_thread_end_period_idx: %lld\n", 
  //   cur_gate_idx, cur_thread_start_event_count, cur_thread_end_event_count, cur_thread_start_time, cur_thread_end_time, cur_thread_start_period_idx, cur_thread_end_period_idx);

  NEeventVal pin_start_event_idxes[MAX_N_PIN];
  NEeventVal pin_end_event_idxes[MAX_N_PIN];
  for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {
    pin_start_event_idxes[pin_idx] = getEventIdxByTime(
      events,
      sh_cur_gate.pin_waveform_starts[pin_idx], 
      sh_cur_gate.pin_waveform_ends[pin_idx],
      cur_thread_start_time
    );
    pin_end_event_idxes[pin_idx] = getEventIdxByTime(
      events,
      sh_cur_gate.pin_waveform_starts[pin_idx], 
      sh_cur_gate.pin_waveform_ends[pin_idx],
      cur_thread_end_time
    );
    // [pin_start_event_idxes, pin_end_event_idxes)
    // printf("cur_gate_idx: %lld, thread_idx_in_block: %lld, pin_idx: %hd, cur_gate->pin_waveform_starts[pin_idx]: %lld, cur_gate->pin_waveform_ends[pin_idx]: %lld pin_start_event_idxes[pin_idx]: %lld, pin_end_event_idxes[pin_idx]: %lld\n", 
    //   cur_gate_idx, thread_idx_in_block, pin_idx, cur_gate->pin_waveform_starts[pin_idx], cur_gate->pin_waveform_ends[pin_idx], pin_start_event_idxes[pin_idx], pin_end_event_idxes[pin_idx]);
  }
  // -----------------end of preparing------------------

  // ------------------------------loop initial------------------------------
  VcdEventVal prev_pin_states[MAX_N_PIN];
  NEeventVal pin_frontier_event_idxes[MAX_N_PIN];
  NPeriodVal pin_cur_period_idxes[MAX_N_PIN];
  NEeventInOnePeriodVal pin_n_event_in_cur_period[MAX_N_PIN];
  for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {
    if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR) {
      prev_pin_states[pin_idx] = getPinDefaultState(sh_cur_gate, pin_idx);
    } else {
      NEeventVal cur_pin_prev_event_idx = my_max(sh_cur_gate.pin_waveform_starts[pin_idx], pin_start_event_idxes[pin_idx] - 1);
      assert(cur_pin_prev_event_idx != INVALID_WAVEFORM_PTR);
      prev_pin_states[pin_idx] = events[cur_pin_prev_event_idx].val;
      pin_frontier_event_idxes[pin_idx] = cur_pin_prev_event_idx == sh_cur_gate.pin_waveform_starts[pin_idx] ? (sh_cur_gate.pin_waveform_starts[pin_idx] + 1) : pin_start_event_idxes[pin_idx];  // skip initial condition
    }

    pin_cur_period_idxes[pin_idx] = INVALID_N_PERIOD_VAL;
    pin_n_event_in_cur_period[pin_idx] = 0;
  }
  VcdEventTime MAX_TIME = max_time + 10;
  VcdEventTime prev_time = cur_thread_start_time;  // TODO check whether this is correct or not
  // VcdEventVal pin_states[MAX_N_PIN];
  // copyPinStates(pin_states, prev_pin_states, sh_cur_gate.n_pin);
  NPinVal triggered_pins[MAX_N_PIN];
  NPinVal n_triggered_pins = 0;
  ThreadPowerResult cur_thread_power_res;
  const PowerVal inv_max_time_vcd_scale = (PowerVal)1.0 / ((PowerVal)max_time * (PowerVal)vcd_time_scale);
  const PowerVal inv_clk_period = (PowerVal)1.0 / (PowerVal)clk_period;
  // ------------------------------end of loop initial------------------------------

  while (true) {
    // ------------------------------get current min time------------------------------
    VcdEventTime cur_time = MAX_TIME;
    n_triggered_pins = 0;
    bool has_output_toggle = false;

    for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {  // we iterate all pins here, since there might be both input and output pins in when condition
      if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR 
        || pin_frontier_event_idxes[pin_idx] >= pin_end_event_idxes[pin_idx]) {
        continue;
      }

      assert(pin_frontier_event_idxes[pin_idx] != INVALID_WAVEFORM_PTR);
      // cur_time = my_min(events[pin_frontier_event_idxes[pin_idx]].time, cur_time);
      const VcdEventTime t = events[pin_frontier_event_idxes[pin_idx]].time;
      if (t < cur_time) {
        cur_time = t;
        n_triggered_pins = 0;
        has_output_toggle = false;
        triggered_pins[n_triggered_pins++] = pin_idx;
      } else if (t == cur_time) {
        triggered_pins[n_triggered_pins++] = pin_idx;
      }
      if (t == cur_time && pin_idx >= cur_gate->n_input_pin
          && getNToggle(prev_pin_states[pin_idx], events[pin_frontier_event_idxes[pin_idx]].val) != 0.0f) {
        has_output_toggle = true;
      }
    }
    if (cur_time == MAX_TIME) {  // no unprocessed input events left 
      break;
    }
    for (NPinVal i = 1; i < n_triggered_pins; ++i) {
      const NPinVal triggered_pin = triggered_pins[i];
      const NEeventVal triggered_event_idx = pin_frontier_event_idxes[triggered_pin];
      NPinVal j = i;
      while (j > 0 && pin_frontier_event_idxes[triggered_pins[j - 1]] > triggered_event_idx) {
        triggered_pins[j] = triggered_pins[j - 1];
        --j;
      }
      triggered_pins[j] = triggered_pin;
    }
    // ------------------------------end of get current min time------------------------------

    // ------------------------------get new state and advance the frontier------------------------------
    // for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {
    //   if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR 
    //     || pin_frontier_event_idxes[pin_idx] >= pin_end_event_idxes[pin_idx]) {
    //     continue;
    //   }

    //   if (events[pin_frontier_event_idxes[pin_idx]].time == cur_time) {
    //     pin_states[pin_idx] = events[pin_frontier_event_idxes[pin_idx]].val;
    //     ++pin_frontier_event_idxes[pin_idx];
    //     triggered_pins[n_triggered_pins++] = pin_idx;
    //   }
    // }
    // for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; pin_idx++) {
    //   printf("%hd", prev_pin_states[pin_idx]);
    // }
    // printf("\n");
    // ------------------------------end of get new state and advance the frontier------------------------------

    // ------------------------------find leakage power value according to previous state------------------------------
    PowerVal leakage_val = findLeakageVal(cur_gate, sh_cur_gate, prev_pin_states);
    // ------------------------------end of find leakage power value according to previous state------------------------------

    // ------------------------------calculate leakage power------------------------------
    // cur_gate->per_tile_leakage_res[thread_idx_in_gate] += leakage_val * (cur_time - prev_time) / max_time;
    // atomicAdd(&cur_gate->per_tile_leakage_res[thread_idx_in_gate % power_res_length_for_each_gate], leakage_val * (cur_time - prev_time) / max_time);
    cur_thread_power_res.leakage_power += leakage_val * (cur_time - prev_time) / max_time;
    calculatePerCycleLeakagePower(cur_gate, leakage_val, prev_time, cur_time, vcd_time_scale, clk_period, per_cycle_leakage_powers);
    // ------------------------------end of calculate leakage power------------------------------

    // ------------------------------dynamic power------------------------------
    const auto cycle_idx = clkedWaveformIdx(cur_time, vcd_time_scale, clk_period);
    VcdEventVal prev_pin_states_before_cur_time[MAX_N_PIN];
    if (has_output_toggle) {
      copyPinStates(prev_pin_states_before_cur_time, prev_pin_states, sh_cur_gate.n_pin);
    }
    for (NPinVal i_triggered_pins = 0; i_triggered_pins < n_triggered_pins; ++i_triggered_pins) {
      const NPinVal pin_idx = triggered_pins[i_triggered_pins];
      const NEeventVal cur_event_idx = pin_frontier_event_idxes[pin_idx];  // current frontier event (time==cur_time)
      const VcdEventVal new_val = events[cur_event_idx].val;
      const VcdEventVal old_val = prev_pin_states[pin_idx];
      // Advance frontier now (does not affect prev_pin_states semantics)
      pin_frontier_event_idxes[pin_idx] = cur_event_idx + 1;
      const float n_toggle = getNToggle(old_val, new_val);
      if (n_toggle == 0.0f) {
        prev_pin_states[pin_idx] = new_val;
        continue;
      }

      NToggleVal glitch_scaling_ratio = getGlitchScalingRatioClockCycleBasedOnDeviceOnly(cur_gate, sh_cur_gate, events, pin_cur_period_idxes, pin_n_event_in_cur_period, cur_event_idx, pin_idx, clk_period, vcd_time_scale);

      EnergyVal cur_internal_energy = 0.0;
      if (pin_idx < cur_gate->n_input_pin) {  // for input pin
        cur_internal_energy = n_toggle * getInputPinInternalEnergyVal(cur_gate, sh_cur_gate, prev_pin_states, pin_idx, getRiseFallEdge(old_val, new_val));
        // printf("cur_time: %lld, pin_idx: %hd, rise_fall: %s, input internal cur_internal_energy: %e\n", cur_time, pin_idx, getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]) == 0 ? "rise" : "fall", cur_internal_energy);
      } else {  // for output pin
        cur_internal_energy = n_toggle * getOutputPinInternalEnergyVal(cur_gate, sh_cur_gate, events, cur_time, pin_idx, getRiseFallEdge(old_val, new_val), max_time, vcd_time_scale, clk_period, prev_pin_states_before_cur_time);
        // if (getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]) == RISE) {
        //   cur_internal_energy -= n_toggle * cur_gate->getSingleRiseTransitionEnergy(pin_idx) / 2;
        // } else {
        //   cur_internal_energy += n_toggle * cur_gate->getSingleRiseTransitionEnergy(pin_idx) / 2;
        // }
        // printf("cur_time: %lld, pin_idx: %hd, rise_fall: %s output internal cur_internal_energy: %e\n", cur_time, pin_idx, getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]) == 0 ? "rise" : "fall", cur_internal_energy);

        //-----switching-----
        NToggleVal n_cur_transition_rise = getNRise(old_val, new_val);
        if (n_cur_transition_rise > 0) {
          const EnergyVal cur_switching_energy = n_cur_transition_rise * sh_cur_gate.getSingleRiseTransitionEnergy(pin_idx);
          if (glitch_scaling_ratio > 0) {
            // cur_gate->per_tile_glitch_switching_res[thread_idx_in_gate] += (glitch_scaling_ratio * cur_switching_energy) / (max_time * vcd_time_scale);
            // atomicAdd(&cur_gate->per_tile_glitch_switching_res[thread_idx_in_gate % power_res_length_for_each_gate], (glitch_scaling_ratio * cur_switching_energy) / (max_time * vcd_time_scale));
            cur_thread_power_res.glitch_switching_power += (glitch_scaling_ratio * cur_switching_energy) * inv_max_time_vcd_scale;
            atomicAdd(&per_cycle_glitch_switching_powers[cycle_idx], glitch_scaling_ratio * cur_switching_energy * inv_clk_period);
          } else {
            // printf("cur_gate_idx: %lld, cur_time: %lld, n_cur_transition_rise: %f\n", cur_gate_idx, cur_time, n_cur_transition_rise);
            // cur_gate->per_tile_switching_res[thread_idx_in_gate] += cur_switching_energy / (max_time * vcd_time_scale);
            // atomicAdd(&cur_gate->per_tile_switching_res[thread_idx_in_gate % power_res_length_for_each_gate], cur_switching_energy / (max_time * vcd_time_scale));
            cur_thread_power_res.switching_power += cur_switching_energy * inv_max_time_vcd_scale;
            atomicAdd(&per_cycle_switching_powers[cycle_idx], cur_switching_energy * inv_clk_period);
          }
        }
        //-----end of switching-----
      }
      //-----internal-----
      // printf("gate_idx: %lld, pin_idx: %hd, prev_time: %lld, prev_val: %hd, cur_time: %lld, cur_val: %hd, n_toggle: %e, cur_internal_energy: %e, rise_fall: %s\n", 
      //   cur_gate_idx, pin_idx, prev_time, prev_pin_states[pin_idx], cur_time, pin_states[pin_idx],
      //   n_toggle, cur_internal_energy,
      //   getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]) == RISE ? "RISE" : "FALL"
      // );
      if (glitch_scaling_ratio > 0) {
        // cur_gate->per_tile_glitch_internal_res[thread_idx_in_gate] += glitch_scaling_ratio * cur_internal_energy / (max_time * vcd_time_scale);
        // atomicAdd(&cur_gate->per_tile_glitch_internal_res[thread_idx_in_gate % power_res_length_for_each_gate], glitch_scaling_ratio * cur_internal_energy / (max_time * vcd_time_scale));
        cur_thread_power_res.glitch_internal_power += glitch_scaling_ratio * cur_internal_energy * inv_max_time_vcd_scale;
        atomicAdd(&per_cycle_glitch_internal_powers[cycle_idx], glitch_scaling_ratio *  cur_internal_energy * inv_clk_period);
      } else {
        // cur_gate->per_tile_internal_res[thread_idx_in_gate] += cur_internal_energy / (max_time * vcd_time_scale);
        // atomicAdd(&cur_gate->per_tile_internal_res[thread_idx_in_gate % power_res_length_for_each_gate], cur_internal_energy / (max_time * vcd_time_scale));
        cur_thread_power_res.internal_power += cur_internal_energy * inv_max_time_vcd_scale;
        atomicAdd(&per_cycle_internal_powers[cycle_idx], cur_internal_energy * inv_clk_period);
      }
      //-----end of internal-----
      prev_pin_states[pin_idx] = new_val;
    }
    // ------------------------------end of dynamic power------------------------------

    // ------------------------------wrap up------------------------------
    prev_time = cur_time;
    // copyPinStates(prev_pin_states, pin_states, sh_cur_gate.n_pin);
    // n_triggered_pins = 0;
    // ------------------------------end of wrap up------------------------------
  }

  // -----------------warp up------------------
  PowerVal final_leakage_val = findLeakageVal(cur_gate, sh_cur_gate, prev_pin_states);
  // TODO check is cur_thread_end_time correct?
  VcdEventTime final_end_time = my_min(cur_thread_end_time, max_time);
  // cur_gate->per_tile_leakage_res[thread_idx_in_gate] += final_leakage_val * (final_end_time - prev_time) / max_time;
  // atomicAdd(&cur_gate->per_tile_leakage_res[thread_idx_in_gate % power_res_length_for_each_gate], final_leakage_val * (final_end_time - prev_time) / max_time);
  cur_thread_power_res.leakage_power += final_leakage_val * (final_end_time - prev_time) / max_time;
  calculatePerCycleLeakagePower(cur_gate, final_leakage_val, prev_time, final_end_time, vcd_time_scale, clk_period, per_cycle_leakage_powers);
  // -----------------end of wrap up------------------

  atomicAdd(&cur_gate->per_tile_leakage_res[threadIdx.x], cur_thread_power_res.leakage_power);
  atomicAdd(&cur_gate->per_tile_glitch_switching_res[threadIdx.x], cur_thread_power_res.glitch_switching_power);
  atomicAdd(&cur_gate->per_tile_switching_res[threadIdx.x], cur_thread_power_res.switching_power);
  atomicAdd(&cur_gate->per_tile_glitch_internal_res[threadIdx.x], cur_thread_power_res.glitch_internal_power);
  atomicAdd(&cur_gate->per_tile_internal_res[threadIdx.x], cur_thread_power_res.internal_power);

  (void)n_cycle_per_thread;
}

__global__ static void kernel2MergeAllPowerResult(
  NGateVal n_gate, Gate *gates, 
  PowerVal *gate_leakage_powers, PowerVal *gate_internal_powers, PowerVal *gate_glitch_internal_powers,
  PowerVal *gate_switching_powers, PowerVal *gate_glitch_switching_powers,
  int power_res_length_for_each_gate
) {
  const NGateVal cur_gate_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cur_gate_idx >= n_gate) {
    return;
  }

  const Gate* cur_gate = &gates[cur_gate_idx];
  // const NThreadVal n_cur_gate_thread = cur_gate->all_pins_end_thread - cur_gate->all_pins_start_thread;
  for (NThreadVal t_idx = 0; t_idx < power_res_length_for_each_gate; ++t_idx) {
    gate_leakage_powers[cur_gate_idx] += cur_gate->per_tile_leakage_res[t_idx];
    gate_internal_powers[cur_gate_idx] += cur_gate->per_tile_internal_res[t_idx];
    gate_glitch_internal_powers[cur_gate_idx] += cur_gate->per_tile_glitch_internal_res[t_idx];
    gate_switching_powers[cur_gate_idx] += cur_gate->per_tile_switching_res[t_idx];
    gate_glitch_switching_powers[cur_gate_idx] += cur_gate->per_tile_glitch_switching_res[t_idx];
  }
}


// -----------------------------------------------naive baseline with two separate kernels-----------------------------------------------
__device__ __forceinline__ NPeriodVal
eventPeriodIdx(const Event* events,
               NEeventVal event_idx,
               PeriodVal clk_period,
               EventTimeVal vcd_time_scale) {
  return clkedWaveformIdx(events[event_idx].time, vcd_time_scale, clk_period);
}

// 返回：在 [start, end) 中，第一个使 period >= target_period 的 event_idx
__device__ __forceinline__ NEeventVal
lowerBoundEventIdxByPeriod(const Event* events,
                           NEeventVal start,
                           NEeventVal end,
                           NPeriodVal target_period,
                           PeriodVal clk_period,
                           EventTimeVal vcd_time_scale) {
  NEeventVal left = start, right = end;
  while (left < right) {
    NEeventVal mid = left + (right - left) / 2;
    NPeriodVal mid_p = clkedWaveformIdx(events[mid].time, vcd_time_scale, clk_period);
    if (mid_p < target_period) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

__device__ __forceinline__ NToggleVal
getGlitchScalingRatioClockCycle_EventCentric(
  const Gate* cur_gate, const Gate& sh_cur_gate,
  const Event* events,
  NEeventVal cur_event_idx, NPinVal pin_idx,
  PeriodVal clk_period, EventTimeVal vcd_time_scale
) {
  if (cur_gate->pin_is_clocks[pin_idx]) {
    return INVALID_N_TOGGLE_VAL;
  }

  const NEeventVal pin_start = sh_cur_gate.pin_waveform_starts[pin_idx];
  const NEeventVal pin_end   = sh_cur_gate.pin_waveform_ends[pin_idx];
  if (pin_start == INVALID_WAVEFORM_PTR || pin_end == INVALID_WAVEFORM_PTR) {
    return INVALID_N_TOGGLE_VAL;
  }
  if (cur_event_idx < pin_start || cur_event_idx >= pin_end) {
    return INVALID_N_TOGGLE_VAL;
  }

  const NPeriodVal cur_period =
    clkedWaveformIdx(events[cur_event_idx].time, vcd_time_scale, clk_period);

  // 计算本周期在该 pin 上的 event 范围 [period_start_idx, period_end_idx)
  const NEeventVal period_start_idx =
    lowerBoundEventIdxByPeriod(events, pin_start, pin_end, cur_period, clk_period, vcd_time_scale);
  const NEeventVal period_end_idx =
    lowerBoundEventIdxByPeriod(events, pin_start, pin_end, cur_period + 1, clk_period, vcd_time_scale);

  const NEeventVal n_in_period = period_end_idx - period_start_idx;
  if (n_in_period <= 0) {
    return INVALID_N_TOGGLE_VAL;
  }

  const NEeventVal pos = cur_event_idx - period_start_idx;   // 0-based in this period
  if (pos < 0 || pos >= n_in_period) {
    return INVALID_N_TOGGLE_VAL;
  }

  const NEeventInOnePeriodVal count_so_far = (NEeventInOnePeriodVal)(pos + 1);

  VcdEventTime prev_glitch_pulse_width = INVALID_PULSE_WIDTH;
  if (pos >= 1) {
    // 上一个 event 必然同周期（因为都在 [period_start_idx, period_end_idx) 内）
    prev_glitch_pulse_width = events[cur_event_idx].time - events[cur_event_idx - 1].time;
  }

  VcdEventTime next_glitch_pulse_width = INVALID_PULSE_WIDTH;
  if (pos + 1 < n_in_period) {  // 有 next event（同周期）
    // 复刻你原来的条件：
    // if (cur+2 in same period || count_so_far is odd) => take next pulse width
    if ((pos + 2 < n_in_period) || (count_so_far % 2 == 1)) {
      next_glitch_pulse_width = events[cur_event_idx + 1].time - events[cur_event_idx].time;
    }
  } else if (count_so_far % 2 == 1) {
    // 是本周期最后一个 event 且 count_so_far 为奇数，复刻原逻辑：INVALID
    return INVALID_N_TOGGLE_VAL;
  }

  const VcdEventTime selected = my_max(prev_glitch_pulse_width, next_glitch_pulse_width);
  if (selected != INVALID_PULSE_WIDTH) {
    const SlewVal sum_slew = sh_cur_gate.pin_rise_slews[pin_idx] + sh_cur_gate.pin_fall_slews[pin_idx];
    return getTimeBasedGlitchScalingRatio(selected * vcd_time_scale, sum_slew);
  }
  return INVALID_N_TOGGLE_VAL;
}

// Helper: map a linear event index (flattened by pin order) to (pin_idx, global_event_idx).
__device__ __forceinline__ bool
mapFlattenedEventToPinAndEventIdx(const Gate* gate,
                                  NEeventVal linear_event_idx_in_gate,
                                  NPinVal* out_pin_idx,
                                  NEeventVal* out_event_idx) {
  NEeventVal base = 0;
  for (NPinVal pin_idx = 0; pin_idx < gate->n_pin; ++pin_idx) {
    const NEeventVal start = gate->pin_waveform_starts[pin_idx];
    const NEeventVal end = gate->pin_waveform_ends[pin_idx];
    if (start == INVALID_WAVEFORM_PTR || end == INVALID_WAVEFORM_PTR) {
      continue;
    }
    const NEeventVal len = end - start;
    if (linear_event_idx_in_gate < base + len) {
      *out_pin_idx = pin_idx;
      *out_event_idx = start + (linear_event_idx_in_gate - base);
      return true;
    }
    base += len;
  }
  return false;
}

__global__ static void kernel1_1NaiveDynamicPowerCalculationTimeRangePartitionedByCycleUnit(
    char cuda_thread_partition_basis, const int n_event_per_thread, const int n_cycle_per_thread, const int n_thread_per_block,
    const NPeriodVal interval_start_period_idx, const NPeriodVal interval_end_period_idx, const NEeventVal vcd_time_unit_per_cycle, const VcdEventTime max_time,
    PeriodVal clk_period, EventTimeVal vcd_time_scale,
    Gate* gates, const Event* events, const NGateVal* block_corr_gate_idxes,
    PowerVal* per_cycle_leakage_powers, PowerVal* per_cycle_internal_powers, PowerVal* per_cycle_glitch_internal_powers,
    PowerVal* per_cycle_switching_powers, PowerVal* per_cycle_glitch_switching_powers,
    int power_res_length_for_each_gate
  ) {
  const NThreadVal thread_idx_in_block = threadIdx.x;
  const NBlockVal idx_of_block = blockIdx.x;
  const NGateVal cur_gate_idx = block_corr_gate_idxes[idx_of_block];
  Gate* cur_gate = &gates[cur_gate_idx];
  const Gate& sh_cur_gate = gates[cur_gate_idx];

  assert(idx_of_block < cur_gate->getEndBlockIdx(cuda_thread_partition_basis));

  const NThreadVal thread_idx_in_gate = (idx_of_block - cur_gate->getStartBlockIdx(cuda_thread_partition_basis)) * n_thread_per_block + thread_idx_in_block;

  // We enforce: event partition + exactly 1 event per thread.
  if (cuda_thread_partition_basis != 'e' || n_event_per_thread != 1) {
    return;
  }

  // Bind this thread to exactly ONE concrete event (pin + event_idx).
  NPinVal pin_idx = -1;
  NEeventVal event_idx = -1;
  if (!mapFlattenedEventToPinAndEventIdx(cur_gate, thread_idx_in_gate, &pin_idx, &event_idx)) {
    return;  // oversubscribed thread or no valid waveform
  }

  // Skip "initial condition" event on this pin (same as your original "skip initial condition" behavior).
  if (event_idx == sh_cur_gate.pin_waveform_starts[pin_idx]) {
    return;
  }

  const VcdEventTime cur_time = events[event_idx].time;

  // Optional: filter by the requested interval.
  const VcdEventTime interval_start_time = (VcdEventTime)interval_start_period_idx * (VcdEventTime)vcd_time_unit_per_cycle;
  // +1 to be consistent with your old "cover the event occurs on the last time" trick.
  const VcdEventTime interval_end_time = (VcdEventTime)interval_end_period_idx * (VcdEventTime)vcd_time_unit_per_cycle + 1;

  if (cur_time < interval_start_time || cur_time >= interval_end_time) {
    return;
  }

  // Build the state seen by this event. Start from the state before cur_time,
  // then apply same-time events that the fused kernel processes first.
  VcdEventVal prev_pin_states_before_cur_time[MAX_N_PIN];
  VcdEventVal prev_pin_states[MAX_N_PIN];
  for (NPinVal p = 0; p < sh_cur_gate.n_pin; ++p) {
    if (sh_cur_gate.pin_waveform_starts[p] == INVALID_WAVEFORM_PTR ||
        sh_cur_gate.pin_waveform_ends[p] == INVALID_WAVEFORM_PTR) {
      prev_pin_states_before_cur_time[p] = getPinDefaultState(sh_cur_gate, p);
      prev_pin_states[p] = prev_pin_states_before_cur_time[p];
      continue;
    }

    const NEeventVal start = sh_cur_gate.pin_waveform_starts[p];
    const NEeventVal end = sh_cur_gate.pin_waveform_ends[p];

    const NEeventVal pos = getEventIdxByTime(events, start, end, cur_time, false);
    NEeventVal prev_idx = (pos > start) ? (pos - 1) : start;
    prev_pin_states_before_cur_time[p] = events[prev_idx].val;
    prev_pin_states[p] = prev_pin_states_before_cur_time[p];
    for (NEeventVal same_time_idx = pos;
         same_time_idx < end && events[same_time_idx].time == cur_time;
         ++same_time_idx) {
      if (same_time_idx >= event_idx) {
        break;
      }
      prev_pin_states[p] = events[same_time_idx].val;
    }
  }
  NToggleVal glitch_scaling_ratio = getGlitchScalingRatioClockCycle_EventCentric(cur_gate, sh_cur_gate,
                                               events, event_idx, pin_idx,
                                               clk_period, vcd_time_scale);

  // pin_states = state AFTER applying this ONE event (only this pin changes).
  VcdEventVal pin_states[MAX_N_PIN];
  copyPinStates(pin_states, prev_pin_states, sh_cur_gate.n_pin);
  pin_states[pin_idx] = events[event_idx].val;

  // If no toggle, nothing to do (keeps semantics consistent with your original n_toggle usage).
  const float n_toggle = getNToggle(prev_pin_states[pin_idx], pin_states[pin_idx]);
  if (n_toggle == 0.0f) {
    return;
  }

  ThreadPowerResult cur_thread_power_res;

  // ---------------- dynamic power for this ONE event ----------------
  EnergyVal cur_internal_energy = 0.0;

  if (pin_idx < cur_gate->n_input_pin) {
    cur_internal_energy =
      n_toggle * getInputPinInternalEnergyVal(cur_gate, sh_cur_gate, prev_pin_states, pin_idx,
                                             getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]));
  } else {
    cur_internal_energy =
      n_toggle * getOutputPinInternalEnergyVal(cur_gate, sh_cur_gate, events, cur_time, pin_idx,
                                              getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]),
                                              max_time, vcd_time_scale, clk_period, prev_pin_states_before_cur_time);

    // ----- switching (output pin only, as in your original code) -----
    NToggleVal n_cur_transition_rise = getNRise(prev_pin_states[pin_idx], pin_states[pin_idx]);
    if (n_cur_transition_rise > 0) {
      const EnergyVal cur_switching_energy =
        n_cur_transition_rise * sh_cur_gate.getSingleRiseTransitionEnergy(pin_idx);

      if (glitch_scaling_ratio > 0) {
        cur_thread_power_res.glitch_switching_power +=
          (glitch_scaling_ratio * cur_switching_energy) / (max_time * vcd_time_scale);

        atomicAdd(&per_cycle_glitch_switching_powers[clkedWaveformIdx(cur_time, vcd_time_scale, clk_period)],
                  glitch_scaling_ratio * cur_switching_energy / clk_period);
      } else {
        cur_thread_power_res.switching_power +=
          cur_switching_energy / (max_time * vcd_time_scale);

        atomicAdd(&per_cycle_switching_powers[clkedWaveformIdx(cur_time, vcd_time_scale, clk_period)],
                  cur_switching_energy / clk_period);
      }
    }
  }

  // ----- internal -----
  if (glitch_scaling_ratio > 0) {
    cur_thread_power_res.glitch_internal_power +=
      glitch_scaling_ratio * cur_internal_energy / (max_time * vcd_time_scale);

    atomicAdd(&per_cycle_glitch_internal_powers[clkedWaveformIdx(cur_time, vcd_time_scale, clk_period)],
              glitch_scaling_ratio * cur_internal_energy / clk_period);
  } else {
    cur_thread_power_res.internal_power +=
      cur_internal_energy / (max_time * vcd_time_scale);

    atomicAdd(&per_cycle_internal_powers[clkedWaveformIdx(cur_time, vcd_time_scale, clk_period)],
              cur_internal_energy / clk_period);
  }

  // Per-gate tile results (keep your original indexing scheme)
  atomicAdd(&cur_gate->per_tile_glitch_switching_res[threadIdx.x], cur_thread_power_res.glitch_switching_power);
  atomicAdd(&cur_gate->per_tile_switching_res[threadIdx.x], cur_thread_power_res.switching_power);
  atomicAdd(&cur_gate->per_tile_glitch_internal_res[threadIdx.x], cur_thread_power_res.glitch_internal_power);
  atomicAdd(&cur_gate->per_tile_internal_res[threadIdx.x], cur_thread_power_res.internal_power);

  // leakage-related arrays are intentionally untouched here
  (void)per_cycle_leakage_powers;
  (void)power_res_length_for_each_gate;
  (void)n_cycle_per_thread;
}

__global__ static void kernel1_2NaiveLeakagePowerCalculationTimeRangePartitionedByCycleUnit (
    char cuda_thread_partition_basis, const int n_event_per_thread, const int n_cycle_per_thread, const int n_thread_per_block, 
    const NPeriodVal interval_start_period_idx, const NPeriodVal interval_end_period_idx, const NEeventVal vcd_time_unit_per_cycle, const VcdEventTime max_time, 
    PeriodVal clk_period, EventTimeVal vcd_time_scale,
    Gate *gates, const Event *events, const NGateVal *block_corr_gate_idxes,
    // Return Values
    PowerVal *per_cycle_leakage_powers, PowerVal *per_cycle_internal_powers, PowerVal *per_cycle_glitch_internal_powers,
    PowerVal *per_cycle_switching_powers, PowerVal *per_cycle_glitch_switching_powers,
    int power_res_length_for_each_gate
  ) {
  // -----------------preparing------------------
  const NThreadVal thread_idx_in_block = threadIdx.x;
  const NBlockVal idx_of_block = blockIdx.x;
  const NGateVal cur_gate_idx = block_corr_gate_idxes[idx_of_block];
  const Gate* cur_gate = &gates[cur_gate_idx];
  const Gate& sh_cur_gate = gates[cur_gate_idx];

  assert(idx_of_block < cur_gate->getEndBlockIdx(cuda_thread_partition_basis));
  const NThreadVal thread_idx_in_gate = (idx_of_block - cur_gate->getStartBlockIdx(cuda_thread_partition_basis)) * n_thread_per_block + thread_idx_in_block;
  NPeriodVal cur_thread_start_period_idx = -1, cur_thread_end_period_idx = -1;
  if (cuda_thread_partition_basis == 'c') {
    cur_thread_start_period_idx = thread_idx_in_gate * sh_cur_gate.n_cycle_per_thread + interval_start_period_idx;
    cur_thread_end_period_idx = (thread_idx_in_gate + 1) * sh_cur_gate.n_cycle_per_thread + interval_start_period_idx;
  } else {
    assert(false);
    return;
  }

  const VcdEventTime cur_thread_start_time = cur_thread_start_period_idx * vcd_time_unit_per_cycle;
  const VcdEventTime cur_thread_end_time = cur_thread_end_period_idx * vcd_time_unit_per_cycle + (cur_thread_end_period_idx == interval_end_period_idx ? 1 : 0);  // add 1 to cover the event occurs on the last time
  // const NThreadVal cur_thread_accu_pin_count = cur_gate->accu_thread_pin_count + thread_idx_in_gate * cur_gate->n_pin;

  // [cur_thread_start_time, cur_thread_end_time)
  if (thread_idx_in_gate >= (cur_gate->getEndThreadIdx(cuda_thread_partition_basis) - cur_gate->getStartThreadIdx(cuda_thread_partition_basis)) || cur_thread_start_period_idx == cur_thread_end_period_idx) {
    return;
  }
  if (cur_thread_start_period_idx == cur_thread_end_period_idx) {
    printf("ERROR: cur_gate_idx: %lld thread_idx_in_gate: %lld, cur_thread_start_period_idx: %lld == cur_thread_end_period_idx: %lld\n", cur_gate_idx, thread_idx_in_gate, cur_thread_start_period_idx, cur_thread_end_period_idx);
    assert(false);
  }
  // printf("cur_gate_idx: %lld, cur_thread_start_event_count: %lld, cur_thread_end_event_count: %lld, cur_thread_start_time: %lld, cur_thread_end_time: %lld, interval_start_period_idx: %lld, cur_thread_end_period_idx: %lld\n", 
  //   cur_gate_idx, cur_thread_start_event_count, cur_thread_end_event_count, cur_thread_start_time, cur_thread_end_time, cur_thread_start_period_idx, cur_thread_end_period_idx);

  NEeventVal pin_start_event_idxes[MAX_N_PIN];
  NEeventVal pin_end_event_idxes[MAX_N_PIN];
  for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {
    pin_start_event_idxes[pin_idx] = getEventIdxByTime(
      events,
      sh_cur_gate.pin_waveform_starts[pin_idx], 
      sh_cur_gate.pin_waveform_ends[pin_idx],
      cur_thread_start_time
    );
    pin_end_event_idxes[pin_idx] = getEventIdxByTime(
      events,
      sh_cur_gate.pin_waveform_starts[pin_idx], 
      sh_cur_gate.pin_waveform_ends[pin_idx],
      cur_thread_end_time
    );
    // [pin_start_event_idxes, pin_end_event_idxes)
    // printf("cur_gate_idx: %lld, thread_idx_in_block: %lld, pin_idx: %hd, cur_gate->pin_waveform_starts[pin_idx]: %lld, cur_gate->pin_waveform_ends[pin_idx]: %lld pin_start_event_idxes[pin_idx]: %lld, pin_end_event_idxes[pin_idx]: %lld\n", 
    //   cur_gate_idx, thread_idx_in_block, pin_idx, cur_gate->pin_waveform_starts[pin_idx], cur_gate->pin_waveform_ends[pin_idx], pin_start_event_idxes[pin_idx], pin_end_event_idxes[pin_idx]);
  }
  // -----------------end of preparing------------------

  // ------------------------------loop initial------------------------------
  VcdEventVal prev_pin_states[MAX_N_PIN];
  NEeventVal pin_frontier_event_idxes[MAX_N_PIN];
  NPeriodVal pin_cur_period_idxes[MAX_N_PIN];
  NEeventInOnePeriodVal pin_n_event_in_cur_period[MAX_N_PIN];
  for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {
    if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR) {
      prev_pin_states[pin_idx] = getPinDefaultState(sh_cur_gate, pin_idx);
    } else {
      NEeventVal cur_pin_prev_event_idx = my_max(sh_cur_gate.pin_waveform_starts[pin_idx], pin_start_event_idxes[pin_idx] - 1);
      assert(cur_pin_prev_event_idx != INVALID_WAVEFORM_PTR);
      prev_pin_states[pin_idx] = events[cur_pin_prev_event_idx].val;
      pin_frontier_event_idxes[pin_idx] = cur_pin_prev_event_idx == sh_cur_gate.pin_waveform_starts[pin_idx] ? (sh_cur_gate.pin_waveform_starts[pin_idx] + 1) : pin_start_event_idxes[pin_idx];  // skip initial condition
    }

    pin_cur_period_idxes[pin_idx] = INVALID_N_PERIOD_VAL;
    pin_n_event_in_cur_period[pin_idx] = 0;
  }
  VcdEventTime MAX_TIME = max_time + 10;
  VcdEventTime prev_time = cur_thread_start_time;  // TODO check whether this is correct or not
  VcdEventVal pin_states[MAX_N_PIN];
  copyPinStates(pin_states, prev_pin_states, sh_cur_gate.n_pin);
  ThreadPowerResult cur_thread_power_res;
  // ------------------------------end of loop initial------------------------------

  while (true) {
    // ------------------------------get current min time------------------------------
    VcdEventTime cur_time = MAX_TIME;
    for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {  // we iterate all pins here, since there might be both input and output pins in when condition
      if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR 
        || pin_frontier_event_idxes[pin_idx] >= pin_end_event_idxes[pin_idx]) {
        continue;
      }

      assert(pin_frontier_event_idxes[pin_idx] != INVALID_WAVEFORM_PTR);
      cur_time = my_min(events[pin_frontier_event_idxes[pin_idx]].time, cur_time);
    }
    if (cur_time == MAX_TIME) {  // no unprocessed input events left 
      break;
    }
    // ------------------------------end of get current min time------------------------------

    // ------------------------------get new state and advance the frontier------------------------------
    for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; ++pin_idx) {
      if (sh_cur_gate.pin_waveform_starts[pin_idx] == INVALID_WAVEFORM_PTR || sh_cur_gate.pin_waveform_ends[pin_idx] == INVALID_WAVEFORM_PTR 
        || pin_frontier_event_idxes[pin_idx] >= pin_end_event_idxes[pin_idx]) {
        continue;
      }

      if (events[pin_frontier_event_idxes[pin_idx]].time == cur_time) {
        pin_states[pin_idx] = events[pin_frontier_event_idxes[pin_idx]].val;
        ++pin_frontier_event_idxes[pin_idx];
      }
    }
    // for (NPinVal pin_idx = 0; pin_idx < sh_cur_gate.n_pin; pin_idx++) {
    //   printf("%hd", prev_pin_states[pin_idx]);
    // }
    // printf("\n");
    // ------------------------------end of get new state and advance the frontier------------------------------

    // ------------------------------find leakage power value according to previous state------------------------------
    PowerVal leakage_val = findLeakageVal(cur_gate, sh_cur_gate, prev_pin_states);
    // ------------------------------end of find leakage power value according to previous state------------------------------

    // ------------------------------calculate leakage power------------------------------
    // cur_gate->per_tile_leakage_res[thread_idx_in_gate] += leakage_val * (cur_time - prev_time) / max_time;
    // atomicAdd(&cur_gate->per_tile_leakage_res[thread_idx_in_gate % power_res_length_for_each_gate], leakage_val * (cur_time - prev_time) / max_time);
    cur_thread_power_res.leakage_power += leakage_val * (cur_time - prev_time) / max_time;
    calculatePerCycleLeakagePower(cur_gate, leakage_val, prev_time, cur_time, vcd_time_scale, clk_period, per_cycle_leakage_powers);
    // ------------------------------end of calculate leakage power------------------------------

    // ------------------------------wrap up------------------------------
    prev_time = cur_time;
    copyPinStates(prev_pin_states, pin_states, sh_cur_gate.n_pin);
    // ------------------------------end of wrap up------------------------------
  }

  // -----------------warp up------------------
  PowerVal final_leakage_val = findLeakageVal(cur_gate, sh_cur_gate, prev_pin_states);
  // TODO check is cur_thread_end_time correct?
  VcdEventTime final_end_time = my_min(cur_thread_end_time, max_time);
  // cur_gate->per_tile_leakage_res[thread_idx_in_gate] += final_leakage_val * (final_end_time - prev_time) / max_time;
  // atomicAdd(&cur_gate->per_tile_leakage_res[thread_idx_in_gate % power_res_length_for_each_gate], final_leakage_val * (final_end_time - prev_time) / max_time);
  cur_thread_power_res.leakage_power += final_leakage_val * (final_end_time - prev_time) / max_time;
  calculatePerCycleLeakagePower(cur_gate, final_leakage_val, prev_time, final_end_time, vcd_time_scale, clk_period, per_cycle_leakage_powers);
  // -----------------end of wrap up------------------

  atomicAdd(&cur_gate->per_tile_leakage_res[threadIdx.x], cur_thread_power_res.leakage_power);

  (void)n_cycle_per_thread;
  (void)n_event_per_thread;
}
// -----------------------------------------------end of naive baseline with two separate kernels-----------------------------------------------

}  // end of namespace power

template <class Kernel>
void
CudaPower::logKernelResourceUsage(
  const char* kernel_name,
  Kernel* kernel,
  NBlockVal grid_block_count,
  int block_thread_count
) const
{
  cudaFuncAttributes attributes{};
  CHECK_CUDA_RUNTIME(cudaFuncGetAttributes(&attributes, kernel));

  int device_id = 0;
  CHECK_CUDA_RUNTIME(cudaGetDevice(&device_id));
  cudaDeviceProp device_prop{};
  CHECK_CUDA_RUNTIME(cudaGetDeviceProperties(&device_prop, device_id));

  int max_active_blocks_per_sm = 0;
  CHECK_CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks_per_sm,
    kernel,
    block_thread_count,
    0
  ));

  const size_t occupancy_resident_blocks = static_cast<size_t>(max_active_blocks_per_sm) * static_cast<size_t>(device_prop.multiProcessorCount);
  const size_t max_resident_blocks = std::min(static_cast<size_t>(grid_block_count), occupancy_resident_blocks);
  const size_t max_resident_threads = max_resident_blocks * static_cast<size_t>(block_thread_count);
  const size_t max_resident_local_memory_bytes = max_resident_threads * static_cast<size_t>(attributes.localSizeBytes);
  const double max_resident_local_memory_gb = max_resident_local_memory_bytes / (1024.0 * 1024.0 * 1024.0);
  const int warps_per_block = (block_thread_count + device_prop.warpSize - 1) / device_prop.warpSize;
  const int active_warps_per_sm = max_active_blocks_per_sm * warps_per_block;
  const int max_warps_per_sm = device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize;
  const double theoretical_occupancy = max_warps_per_sm > 0
                                        ? static_cast<double>(active_warps_per_sm) / max_warps_per_sm
                                        : 0.0;
  const size_t registers_per_block = static_cast<size_t>(attributes.numRegs) * static_cast<size_t>(block_thread_count);
  const size_t max_resident_registers_per_sm = registers_per_block * static_cast<size_t>(max_active_blocks_per_sm);

  LOG_INFO << kernel_name
           << " local memory max resident: " << max_resident_local_memory_gb << " GB, per thread: "
           << attributes.localSizeBytes << " bytes, max resident threads: " << max_resident_threads
           << ", max active blocks per SM: " << max_active_blocks_per_sm
           << ", theoretical occupancy: " << (theoretical_occupancy * 100.0) << "%"
           << ", active warps per SM: " << active_warps_per_sm << "/" << max_warps_per_sm
           << ", registers per thread: " << attributes.numRegs
           << ", registers per block: " << registers_per_block
           << ", max resident registers per SM: " << max_resident_registers_per_sm
           << ", registers per SM: " << device_prop.regsPerMultiprocessor
           << ", SM count: " << device_prop.multiProcessorCount
           << ", grid blocks: " << grid_block_count
           << ", block threads: " << block_thread_count;
}

void
CudaPower::reportPowerKernelResourceUsage(NBlockVal kernel1_all_grid_block_count) const
{
  utils::ScopedTimer timer_report_kernel_resource_usage("reportPowerKernelResourceUsage");
  const int block_thread_count = G_CONFIG.nums.n_thread_per_block_for_all_pins;

  LOG_BEGIN(INFO, "CUDA Power Kernel Resource Usage");
  if (G_CONFIG.flags.cuda_power_separate_kernels_for_dyn_and_leak) {
    logKernelResourceUsage(
      "kernel1_1",
      power::kernel1_1NaiveDynamicPowerCalculationTimeRangePartitionedByCycleUnit,
      n_block_for_event_partition_,
      block_thread_count
    );
    logKernelResourceUsage(
      "kernel1_2",
      power::kernel1_2NaiveLeakagePowerCalculationTimeRangePartitionedByCycleUnit,
      n_block_for_cycle_partition_,
      block_thread_count
    );
  } else {
    logKernelResourceUsage(
      "kernel1All",
      power::kernel1AllPowerCalculationTimeRangePartitionedByCycleUnit,
      kernel1_all_grid_block_count,
      block_thread_count
    );
  }
  LOG_END(INFO, "CUDA Power Kernel Resource Usage");
}

namespace {
const std::array<uint8_t, 256>&
vcdValueToEventValTable()
{
  static const std::array<uint8_t, 256> table = [] {
    std::array<uint8_t, 256> t{};
    t.fill(0xFF);
    t[static_cast<unsigned char>('0')] = 0;
    t[static_cast<unsigned char>('1')] = 1;
    t[static_cast<unsigned char>('X')] = 2;
    t[static_cast<unsigned char>('x')] = 2;
    t[static_cast<unsigned char>('Z')] = 3;
    t[static_cast<unsigned char>('z')] = 3;
    return t;
  }();
  return table;
}

}

CudaPower::CudaPower(StaState *sta) : 
  Power(sta),
  power_computation_kernel_total_time_(0),
  k11_kernel_total_time_(0),
  merge_result_kernel_total_time_(0),
  n_period_(0),
  max_event_time_(0),
  vcd_time_scale_(0.0),
  vcd_time_unit_per_cycle_(0),
  inst_to_pins_(),
  vcd_values_ptr_to_n_event_(),
  vcd_values_bit_pair_list_(),
  inst_to_res_(),
  //--------------------events and gates-----------------------
  events_(nullptr),
  d_events_(nullptr),
  h_multiple_output_gates_(),
  d_multiple_output_gates_(nullptr),
  //--------------------end of events and gates-----------------------
  //--------------------for leakage and internal-----------------------
  n_multiple_output_gate_(0),
  n_block_for_event_partition_(0),
  n_block_for_cycle_partition_(0),
  global_pin_waveform_starts_(nullptr),
  global_pin_waveform_ends_(nullptr),
  global_pin_default_states_(nullptr),
  global_pin_voltages_(nullptr),
  global_pin_rise_slews_(nullptr),
  global_pin_fall_slews_(nullptr),
  global_pin_load_capacitances_(nullptr),
  global_pin_is_clocks_(nullptr),
  global_cell_arc_delays_(nullptr),
  global_per_tile_leakage_res_(nullptr),
  global_per_tile_internal_res_(nullptr),
  global_per_tile_glitch_internal_res_(nullptr),
  global_per_tile_switching_res_(nullptr),
  global_per_tile_glitch_switching_res_(nullptr),
  var_val_ptr_to_value_bit_to_prev_right_bound_(),
  block_corr_gate_idxes_for_event_partition_(nullptr),
  block_corr_gate_idxes_for_cycle_partition_(nullptr),
  input_port_to_internal_luts_map_(),
  input_port_to_n_internal_LUTs_indexed_by_order_map_(),
  input_port_to_internal_luts_indexed_by_order_map_(),
  output_port_to_internal_luts_map_(),
  output_port_to_n_internal_LUTs_indexed_by_order_map_(),
  output_port_to_internal_luts_indexed_by_order_map_(),
  cell_to_leakage_power_data_(),
  cell_to_internal_power_data_(),
  gate_leakage_powers_(nullptr),
  gate_internal_powers_(nullptr),
  gate_glitch_internal_powers_(nullptr),
  per_cycle_leakage_powers_(nullptr),
  per_cycle_internal_powers_(nullptr),
  per_cycle_glitch_internal_powers_(nullptr),
  h_gate_leakage_powers_(nullptr),
  h_gate_internal_powers_(nullptr),
  h_gate_glitch_internal_powers_(nullptr),
  h_per_cycle_leakage_powers_(nullptr),
  h_per_cycle_internal_powers_(nullptr),
  h_per_cycle_glitch_internal_powers_(nullptr),
  //--------------------end of for leakage and internal-----------------------
  //--------------------for switching-----------------------
  gate_switching_powers_(nullptr),
  gate_glitch_switching_powers_(nullptr),
  per_cycle_switching_powers_(nullptr),
  per_cycle_glitch_switching_powers_(nullptr),
  h_gate_switching_powers_(nullptr),
  h_gate_glitch_switching_powers_(nullptr),
  h_per_cycle_switching_powers_(nullptr),
  h_per_cycle_glitch_switching_powers_(nullptr)
  //--------------------end of for switching-----------------------
{
  if (!G_CONFIG.flags.enable_time_based_analysis) {
    LOG_ERROR << "Only time based power analysis is supported in CUDA";
  }
}

// initialization and copy data to device side
void
CudaPower::power(const Corner *corner,
             // Return values.
             PowerResult &total,
             PowerResult &sequential,
             PowerResult &combinational,
             PowerResult &clock,
             PowerResult &macro,
             PowerResult &pad)
{
  LOG_INFO << "cudaPower";
  utils::cuda::setDevice(&G_CONFIG.nums.cuda_device_id);
  printSettings();
  // ---------------------------initialization------------------------------
  const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
  total.clear();
  sequential.clear();
  combinational.clear();
  clock.clear();
  macro.clear();
  pad.clear();
  ensureActivities();
  // ---------------------------end of initialization------------------------------

  utils::ScopedTimer cuda_time_based_power_analysis_timer("CUDA Time Based Power Analysis");
  TIMERSTART(CUDA_TIME_BASED_POWER_ANALYSIS);
  // ---------------------------CUDA initialization------------------------------
  CUDA_MEM_STATS.reset();
  const size_t event_bytes = G_CONFIG.nums.max_event_num * sizeof(power::Event);
  events_ = new power::Event[G_CONFIG.nums.max_event_num];
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_events_, event_bytes));
  CUDA_MEM_STATS.set(CudaMemCategory::events, event_bytes);
  setPeriodAndInitPerCyclePower(vcd_.timeMax(), vcd_.timeScale());
  // ---------------------------end of CUDA initialization------------------------------
  if (G_CONFIG.flags.enable_multiple_rounds_power_analysis) {
    std::vector<std::pair<VcdEventTime, VcdEventTime>> time_intervals;
    if (G_CONFIG.flags.time_slicing_by_activity_file_reading) {
      time_intervals = vcd_.timeIntervals();
      if (G_CONFIG.flags.partition_unit_is_cycle && G_CONFIG.strs.activity_file_format == "vcd") {
        for (auto& interval: time_intervals) {
          interval.first = ceil_div(interval.first, vcd_time_unit_per_cycle_) * vcd_time_unit_per_cycle_;
          interval.second = ceil_div(interval.second, vcd_time_unit_per_cycle_) * vcd_time_unit_per_cycle_;
          if (interval.first == interval.second) {
            LOG_ERROR << "interval.first " << interval.first << " equals to interval.second " << interval.second;
          }
        }
      }
    } else {
      if (G_CONFIG.flags.partition_unit_is_cycle) {
        if (G_CONFIG.flags.iterate_all_events_to_do_time_slicing) {
          getTimeIntervalsByCycleIterateAllEvents(time_intervals);
        } else {
          getTimeIntervalsByCycle(time_intervals);
        }
      } else {
        getTimeIntervals(time_intervals);
      }
    }
    for (size_t time_interval_idx = 0; time_interval_idx < time_intervals.size(); ++time_interval_idx) {
      const auto& interval = time_intervals.at(time_interval_idx);
      LOG_INFO << "interval " << time_interval_idx << ", start time: " << interval.first << " end time: " << interval.second;
      // if (getAccuEventCountOfAllGates(interval.second) - getAccuEventCountOfAllGates(interval.first) > G_CONFIG.nums.max_event_num) {
      //   LOG_ERROR << "GPU event allocation overflow! Accu event count of start time: " << getAccuEventCountOfAllGates(interval.first) 
      //     << " Accu event count of end time: " << getAccuEventCountOfAllGates(interval.second)
      //     << " G_CONFIG.nums.max_event_num: " << G_CONFIG.nums.max_event_num;
      // }

      // ------------------------------copy data to device side------------------------------
      if (time_interval_idx == 0) {
        initGateData(interval.first, interval.second, corner, dcalc_ap);
      } else {
        if (!G_CONFIG.flags.overlapping_get_gate_waveform_range) {
          std::vector<NEeventVal> h_global_pin_waveform_starts;
          std::vector<NEeventVal> h_global_pin_waveform_ends;
          getGateWaveformRangeSingleThread(interval.first, interval.second, h_global_pin_waveform_starts, h_global_pin_waveform_ends);
          // getGateWaveformRangeMultiThread(interval.first, interval.second, h_global_pin_waveform_starts, h_global_pin_waveform_ends);
          renewGateWaveform(h_global_pin_waveform_starts, h_global_pin_waveform_ends);
        }
      }
      scheduleKernel(interval.first, interval.second);
      copyWaveformToDeviceSide();
      copyGateDataToDeviceSide();
      CUDA_MEM_STATS.log("CUDA Power Device Memory Breakdown", "after CUDA data setup");
      // ------------------------------end of copy data to device side------------------------------
      runCudaPowerAnalysis(interval.first, interval.second, time_interval_idx < (time_intervals.size() - 1) ? &time_intervals.at(time_interval_idx + 1) : nullptr);
      mergeResult();
      recordResult();
    }
  } else {
    // ------------------------------copy data to device side------------------------------
    initGateData(0, max_event_time_ + 1, corner, dcalc_ap);
    scheduleKernel(0, vcd_.timeMax() + (G_CONFIG.flags.partition_unit_is_cycle ? vcd_time_unit_per_cycle_ : 1));
    copyWaveformToDeviceSide();
    copyGateDataToDeviceSide();
    CUDA_MEM_STATS.log("CUDA Power Device Memory Breakdown", "after CUDA data setup");
    // ------------------------------end of copy data to device side------------------------------
    runCudaPowerAnalysis(0, vcd_.timeMax() + (G_CONFIG.flags.partition_unit_is_cycle ? vcd_time_unit_per_cycle_ : 1));  // plus one here to make it as an open interval to keep same with multiple rounds
    mergeResult();
    recordResult();
  }

  // ---------------------------merge result------------------------------
  finalizeResult(total, sequential, combinational, clock, macro, pad);
  // ---------------------------end of merge result------------------------------

  TIMEREND(CUDA_TIME_BASED_POWER_ANALYSIS);
  DURATION_ms(CUDA_TIME_BASED_POWER_ANALYSIS);
  LOG_INFO << "power_computation_kernel_total_time_: " << power_computation_kernel_total_time_ << "ms";
  LOG_INFO << "k11_kernel_total_time_: " << k11_kernel_total_time_ << "ms";
  LOG_INFO << "merge_result_kernel_total_time_: " << merge_result_kernel_total_time_ << "ms";
}

void 
CudaPower::printSettings()
{
  LOG_BEGIN(INFO, "CUDA Power Settings");
  LOG_INFO << "current_device: " << G_CONFIG.nums.cuda_device_id;
  LOG_INFO << "enable_cuda_power_analysis: " << (G_CONFIG.flags.enable_cuda_power_analysis ? "true": "false");
  LOG_INFO << "enable_time_based_analysis: " << (G_CONFIG.flags.enable_time_based_analysis ? "true": "false");
  LOG_INFO << "enable_multiple_rounds_power_analysis: " << (G_CONFIG.flags.enable_multiple_rounds_power_analysis ? "true": "false");
  LOG_INFO << "overlapping_get_gate_waveform_range: " << (G_CONFIG.flags.overlapping_get_gate_waveform_range ? "true": "false");
  LOG_INFO << "cuda_power_separate_kernels_for_dyn_and_leak: " << (G_CONFIG.flags.cuda_power_separate_kernels_for_dyn_and_leak ? "true": "false");
  LOG_INFO << "enable_auto_select_n_cycle_per_thread_for_each_gate: " << (G_CONFIG.flags.enable_auto_select_n_cycle_per_thread_for_each_gate ? "true": "false");
  LOG_INFO << "the unit of workload partition is cycle: " << (G_CONFIG.flags.partition_unit_is_cycle ? "true": "false");

  LOG_INFO << "max_event_num: " << G_CONFIG.nums.max_event_num << " estimated mem usage for events is " << G_CONFIG.nums.max_event_num * (sizeof(VcdEventTime) + sizeof(VcdEventVal)) / (1024.0 * 1024 * 1024) << "GB";
  LOG_INFO << "cuda_thread_partition_basis: " << G_CONFIG.strs.cuda_thread_partition_basis;
  LOG_INFO << "n_cycle_per_thread: " << G_CONFIG.nums.n_cycle_per_thread;
  LOG_INFO << "n_event_per_thread_for_all_pins: " << G_CONFIG.nums.n_event_per_thread_for_all_pins;
  LOG_INFO << "n_thread_per_block_for_all_pins: " << G_CONFIG.nums.n_thread_per_block_for_all_pins;
  LOG_INFO << "n_cycle_auto_selection_parallelism_floor: " << G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor;
  LOG_INFO << "n_cycle_auto_selection_e_target: " << G_CONFIG.nums.n_cycle_auto_selection_e_target;
  LOG_INFO << "max_n_pin_for_leakage_power: " << G_CONFIG.nums.max_n_pin_for_leakage_power << " max_n_pin_for_internal_power: " << G_CONFIG.nums.max_n_pin_for_internal_power;

#ifdef DISABLE_BSIM
  LOG_INFO << "BSIM disabled";
#else
  LOG_INFO << "BSIM enabled";
#endif


  LOG_INFO << "result_dir:" << std::filesystem::current_path() / G_CONFIG.paths.result_dir;
  LOG_END(INFO, "CUDA Power Settings");
}

// ------------------------------time inteval scheduling------------------------------
void 
CudaPower::getTimeIntervals(
  // Return values.
  std::vector<std::pair<VcdEventTime, VcdEventTime>>& intervals
) const {
  utils::ScopedTimer timer_get_time_intervals("getTimeIntervals");
  size_t total_bus_width = utils::getTotalBusWidthOfAllVars(vcd_);
  if (G_CONFIG.nums.max_event_num < 2 * total_bus_width + 1) {
    LOG_ERROR << "G_CONFIG.nums.max_event_num " << G_CONFIG.nums.max_event_num << " is too small, which should be twice bigger than total_bus_width: " << total_bus_width;
  }
  NEeventVal n_event_per_interval = G_CONFIG.nums.max_event_num - total_bus_width - 1; // minus total_bus_width to reserve enuough space for time advancing
  LOG_INFO << "n_event_per_interval: " << n_event_per_interval;
  int interval_count = 0;
  intervals.clear();
  while (true) {
    // get the inetrval boundary of nth interval
    VcdEventTime left_boundary = 0, right_boundary = 0;
    getTimeIntervalBoudary(n_event_per_interval * interval_count, &left_boundary);
    getTimeIntervalBoudary(n_event_per_interval * (interval_count + 1), &right_boundary);
    intervals.emplace_back(left_boundary, right_boundary);
    if (right_boundary >= max_event_time_) {
      break;
    }
    ++interval_count;
  }
}

void 
CudaPower::getTimeIntervalBoudary(
  NEeventVal accu_event_count, 
  // Return values.
  VcdEventTime *boundary
) const {
  VcdEventTime left = 0, right = max_event_time_ + 1;
  while (left < right) {  // [left, right)
    VcdEventTime mid = left + (right - left) / 2;  // avoid overflow

    if (getAccuEventCountOfAllGates(mid) < accu_event_count) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  *boundary = right;
}

void
CudaPower::getTimeIntervalsByCycle(
  // Return values.
  std::vector<std::pair<VcdEventTime, VcdEventTime>>& intervals
) const {
  utils::ScopedTimer timer_get_time_intervals("getCycleIntervals");
  TIMERSTART(GET_CYCLE_INTERVALS);

  size_t total_bus_width = utils::getTotalBusWidthOfAllVars(vcd_);  // TODO this is the checking for vcd time unit, change it to cycle instead
  if (G_CONFIG.nums.max_event_num < 2 * total_bus_width + 1) {
    LOG_ERROR << "G_CONFIG.nums.max_event_num " << G_CONFIG.nums.max_event_num << " is too small, which should be twice bigger than total_bus_width: " << total_bus_width;
  }
  NEeventVal n_event_per_interval = G_CONFIG.nums.max_event_num - total_bus_width - 1; // minus total_bus_width to reserve enuough space for time advancing
  LOG_INFO << "n_event_per_interval: " << n_event_per_interval;

  std::vector<NPeriodVal> boundaries(1, 0);
  while (true) {
    NPeriodVal right_boundary = 0;
    getCycleIntervalBoudary(n_event_per_interval * boundaries.size(), boundaries.at(boundaries.size() - 1), &right_boundary);
    boundaries.push_back(right_boundary);
    if (right_boundary >= n_period_) {
      break;
    }
  }

  for (size_t bound_idx = 0; bound_idx < boundaries.size() - 1; ++bound_idx) {
    intervals.emplace_back(boundaries.at(bound_idx) * vcd_time_unit_per_cycle_, (boundaries.at(bound_idx + 1)) * vcd_time_unit_per_cycle_);
  }
  intervals.at(intervals.size() - 1).second += 1;

  TIMEREND(GET_CYCLE_INTERVALS);
  DURATION_ms(GET_CYCLE_INTERVALS);
}

void
CudaPower::getTimeIntervalsByCycleIterateAllEvents(
  // Return values.
  std::vector<std::pair<VcdEventTime, VcdEventTime>>& intervals
) {
  utils::ScopedTimer timer_get_time_intervals("getTimeIntervalsByCycleIterateAllEvents");
  TIMERSTART(GET_CYCLE_INTERVALS_ITERATE_ALL_EVENTS);

  size_t total_bus_width = utils::getTotalBusWidthOfAllVars(vcd_);  // TODO this is the checking for vcd time unit, change it to cycle instead
  if (G_CONFIG.nums.max_event_num < 2 * total_bus_width + 1) {
    LOG_ERROR << "G_CONFIG.nums.max_event_num " << G_CONFIG.nums.max_event_num << " is too small, which should be twice bigger than total_bus_width: " << total_bus_width;
  }
  NEeventVal n_event_per_interval = G_CONFIG.nums.max_event_num - total_bus_width - 1; // minus total_bus_width to reserve enuough space for time advancing
  LOG_INFO << "n_event_per_interval: " << n_event_per_interval;

  //-----------------preprocessing------------------
  LeafInstanceIterator *inst_iter = network_->leafInstanceIterator();
  std::unordered_map<NPeriodVal, NEeventVal> cycle_idx_to_n_event;
  std::unordered_map<const VcdValue*, NEeventVal> value_ptrs_to_n_event;
  while (inst_iter->hasNext()) {
    const Instance *inst = inst_iter->next();
    LibertyCell *cell = network_->libertyCell(inst);
    if (!cell) {
      continue;
    }
    std::vector<const Pin *> pins;
    NPinVal n_pin = 0, n_input_pin = 0, n_output_pin = 0;
    getPinInformation(inst, &pins, &n_pin, &n_input_pin, &n_output_pin);
    for (NPinVal pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
      const Pin *cur_pin = pins.at(pin_idx);
      PwrActivity activity = findActivity(cur_pin);
      if (value_ptrs_to_n_event.count(activity.vcdValues()) == 0) {
        value_ptrs_to_n_event[activity.vcdValues()] = activity.nEvent();
        for (NEeventVal i = 0; i < activity.nEvent(); ++i) {
          cycle_idx_to_n_event[clkedWaveformIdx(activity.vcdValues()[i].time(), vcd_.timeScale(), clk_period_)]++;
        }
      }
    }
  }
  delete inst_iter;

  std::unordered_map<NPeriodVal, NEeventVal> cycle_idx_to_accu_n_event;
  cycle_idx_to_accu_n_event[0] = 0;
  for (NPeriodVal cycle_idx = 1; cycle_idx <= n_period_; ++cycle_idx) {
    cycle_idx_to_accu_n_event[cycle_idx] = cycle_idx_to_accu_n_event.at(cycle_idx - 1) + cycle_idx_to_n_event.at(cycle_idx - 1);
  }
  //-----------------end of preprocessing------------------

  std::vector<NPeriodVal> boundaries(1, 0);
  while (true) {
    NPeriodVal right_boundary = 0;
    getCycleIntervalBoudaryByCycleIdxToAccuEvents(n_event_per_interval * boundaries.size(), cycle_idx_to_accu_n_event, &right_boundary);
    boundaries.push_back(right_boundary);
    if (right_boundary >= n_period_) {
      break;
    }
  }

  for (size_t bound_idx = 0; bound_idx < boundaries.size() - 1; ++bound_idx) {
    intervals.emplace_back(boundaries.at(bound_idx) * vcd_time_unit_per_cycle_, (boundaries.at(bound_idx + 1)) * vcd_time_unit_per_cycle_);
  }
  intervals.at(intervals.size() - 1).second += 1;

  TIMEREND(GET_CYCLE_INTERVALS_ITERATE_ALL_EVENTS);
  DURATION_ms(GET_CYCLE_INTERVALS_ITERATE_ALL_EVENTS);
}

void 
CudaPower::getCycleIntervalBoudary(
  NEeventVal accu_event_count, 
  // Return values.
  NPeriodVal *cycle_boundary
) const {
  NPeriodVal left = 0, right = n_period_;
  while (left < right) {  // [left, right)
    NPeriodVal mid = left + (right - left) / 2;  // avoid overflow

    if (getAccuEventCountOfAllGates(mid * vcd_time_unit_per_cycle_) < accu_event_count) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  *cycle_boundary = right;
}

void 
CudaPower::getCycleIntervalBoudary(
  NEeventVal accu_event_count, 
  NPeriodVal left,
  // Return values.
  NPeriodVal *cycle_boundary
) const {
  NPeriodVal right = n_period_;
  while (left < right) {  // [left, right)
    NPeriodVal mid = left + (right - left) / 2;  // avoid overflow

    if (getAccuEventCountOfAllGates(mid * vcd_time_unit_per_cycle_) < accu_event_count) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  *cycle_boundary = right;
}

void 
CudaPower::getCycleIntervalBoudaryByCycleIdxToAccuEvents(
  NEeventVal accu_event_count, 
  const std::unordered_map<NPeriodVal, NEeventVal>& cycle_idx_to_accu_n_event,
  // Return values.
  NPeriodVal *cycle_boundary
) const {
  NPeriodVal left = 0, right = n_period_;
  while (left < right) {  // [left, right)
    NPeriodVal mid = left + (right - left) / 2;  // avoid overflow

    if (cycle_idx_to_accu_n_event.at(mid) < accu_event_count) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  *cycle_boundary = right;
}

// get accu_event_count in range [0, time)
NEeventVal 
CudaPower::getAccuEventCountOfAllGates(VcdEventTime time) const
{
  NEeventVal accu_event_count = 0;
  for (const VcdVar* var: vcd_.vars()) {
    assert(vcd_.varIdValid(var->id()));
    const auto& values = vcd_.values(var);
    accu_event_count += getEventIdxByTime(values, time) * utils::getVarBusWidth(var, vcd_);  // we do not add one here, since the last one is out of range of time
  }

  return accu_event_count;
}

void 
CudaPower::initGateData(VcdEventTime start_time, VcdEventTime end_time, const Corner *corner, const DcalcAnalysisPt *dcalc_ap)
{
  utils::ScopedTimer timer_init_gate_data("initGateData");
  TIMERSTART(INIT_GATE_DATA);
  // ---------------------------copy data to device side------------------------------
  LOG_INFO << "Initing gate data";
  LeafInstanceIterator *inst_iter = network_->leafInstanceIterator();
  h_multiple_output_gates_.clear();
  inst_to_res_.clear();

  // -----auxiliary arrays-----
  std::vector<NEeventVal> global_pin_waveform_starts;
  std::vector<NEeventVal> global_pin_waveform_ends;
  std::vector<VcdEventVal> global_pin_default_states;
  std::vector<VoltageVal> global_pin_voltages;
  std::vector<SlewVal> global_pin_rise_slews;
  std::vector<SlewVal> global_pin_fall_slews;
  std::vector<CapacitanceVal> global_pin_load_capacitances;
  std::vector<short int> global_pin_is_clocks;
  std::vector<DelayVal> global_cell_arc_delays;
  // -----end of auxiliary arrays-----

  while (inst_iter->hasNext()) {
    const Instance *inst = inst_iter->next();
    if (h_multiple_output_gates_.size() % 1000 == 0) {
      LOG_INFO << "initGateData accu gate " << h_multiple_output_gates_.size() << " " << network_->pathName(inst);
    }
    LibertyCell *cell = network_->libertyCell(inst);
    if (!cell) {
      continue;
    }
    const LibertyCell *corner_cell = cell->cornerCell(dcalc_ap);
    inst_to_res_.emplace(inst, PowerResult{});

    // ---------------------------init pin information------------------------------
    std::vector<const Pin *> pins;
    NPinVal n_pin = 0, n_input_pin = 0, n_output_pin = 0;
    getPinInformation(inst, &pins, &n_pin, &n_input_pin, &n_output_pin);
    inst_to_pins_.emplace(inst, pins);
    if (n_pin > MAX_N_PIN) {
      LOG_ERROR << network_->pathName(inst) << " Too large n_pin: " << n_pin << " exceeds MAX_N_PIN: " << MAX_N_PIN;
    }
    // if (n_pin * 2 > G_CONFIG.nums.n_event_per_thread_for_all_pins) {
    //   LOG_ERROR << "Too small n_event_per_thread_for_all_pins: " << G_CONFIG.nums.n_event_per_thread_for_all_pins << " for n_pin: " << n_pin;
    // }
    if (LOG_DEBUG_FLAG) {
      LOG_DEBUG << "Gate " << h_multiple_output_gates_.size() << " " << network_->pathName(inst) << " n_pin: " << n_pin << " n_input_pin: " << n_input_pin << " n_output_pin: " << n_output_pin;
    }
    // ---------------------------end of init pin information------------------------------

    // ---------------------------basic information------------------------------
    // std::vector<VoltageVal> pin_voltages(n_pin);
    // std::vector<SlewVal> pin_rise_slews(n_pin);
    // std::vector<SlewVal> pin_fall_slews(n_pin);
    // std::vector<CapacitanceVal> pin_load_capacitances(n_pin);
    // std::shared_ptr<bool[]> pin_is_clocks(new bool[n_pin]);
    std::unordered_map<std::string, NPinVal> port_name_to_idx_map;  // input pin with small index, output pin wi big index
    utils::ScopedTimer timer_timing_preparation("Timing Preparation");
    for (NPinVal pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
      const Pin *cur_pin = pins.at(pin_idx);
      const LibertyPort *cur_port = network_->libertyPort(cur_pin);
      Vertex *cur_pin_vertex = graph_->pinLoadVertex(cur_pin);
      global_pin_default_states.push_back(::power::utils::getVertexDefaultState(cur_pin_vertex));
      if (cur_port) {
        global_pin_voltages.push_back(portVoltage(corner_cell, cur_port, dcalc_ap));
        const SlewVal rise_slew = getSlew(cur_pin_vertex, RiseFall::rise(), corner);
        const SlewVal fall_slew = getSlew(cur_pin_vertex, RiseFall::fall(), corner);
        if (delayInf(rise_slew) || delayInf(fall_slew)) {
          LOG_ERROR << "Invalid slew value of " << network_->pathName(cur_pin) 
            << " rise slew: " << rise_slew
            << " fall slew: " << fall_slew;
        }
        global_pin_rise_slews.push_back(rise_slew);
        global_pin_fall_slews.push_back(fall_slew);
        global_pin_load_capacitances.push_back(cur_port->direction()->isAnyOutput()
          ? graph_delay_calc_->loadCap(cur_pin, dcalc_ap)
          : 0.0);
        global_pin_is_clocks.push_back(cur_port->isClock() ? 1 : 0);

        port_name_to_idx_map[cur_port->name()] = pin_idx;
      } else {
        LOG_ERROR << "Port not found for " << network_->pathName(cur_pin);
      }
    }
    timer_timing_preparation.EndTiming();
    if (port_name_to_idx_map.size() != n_pin) {
      LOG_ERROR << "port_name_to_idx_map.size() != n_pin " << " port_name_to_idx_map.size(): " << port_name_to_idx_map.size() << " n_pin: " << n_pin;
    }
    // ---------------------------end of basic information------------------------------

    // ---------------------------leakage power------------------------------
    const LeakagePowerData* leakage_power_data = nullptr;
    utils::ScopedTimer timer_leakage_preparation("Leakage Preparation");
    leakage_power_data = &getLeakagePowerData(cell, corner_cell, n_pin, port_name_to_idx_map);
    timer_leakage_preparation.EndTiming();
    // ---------------------------end of leakage power------------------------------

    // ---------------------------internal power------------------------------
    const InternalPowerData* internal_power_data = nullptr;
    utils::ScopedTimer timer_lut_preparation("LUT Preparation");
    internal_power_data = &getInternalPower(inst, corner_cell, dcalc_ap, n_pin, pins, port_name_to_idx_map);
    timer_lut_preparation.EndTiming();
    // ---------------------------end of internal power------------------------------

    // ---------------------------delay------------------------------
    utils::ScopedTimer timer_delay_preparation("Delay Preparation");
    getDelay(pins, n_pin, n_input_pin, dcalc_ap, global_cell_arc_delays);
    timer_delay_preparation.EndTiming();
    // ---------------------------end of delay------------------------------

    h_multiple_output_gates_.emplace_back(new power::Gate(
      inst, corner_cell, !(cell->isMacro() || cell->isMemory() || cell->interfaceTiming() || cell->isPad() || inClockNetwork(inst) || cell->hasSequentials()),
      leakage_power_data->leakage_power_values, leakage_power_data->default_leakage_exists, leakage_power_data->default_leakage_exists ? leakage_power_data->default_leakage_power_val : INVALID_LEAKAGE_POWER_VAL,
      internal_power_data->input_ports_internal_power_LUTs, internal_power_data->input_ports_n_internal_power_LUTs_index_by_order, internal_power_data->input_ports_internal_power_LUTs_index_by_order,
      internal_power_data->output_ports_internal_power_LUTs, internal_power_data->output_ports_n_internal_power_LUTs_index_by_order, internal_power_data->output_ports_internal_power_LUTs_index_by_order,
      n_pin, n_input_pin, n_output_pin, INVALID_PIN_IDX
    ));
  }
  delete inst_iter;
  {
    utils::ScopedTimer timer_initial_wave_preparation("Initial Waveform Preparation");
    for (const power::Gate* cur_gate : h_multiple_output_gates_) {
      const std::vector<const Pin *>& pins = inst_to_pins_.at(cur_gate->inst);
      for (const Pin* cur_pin : pins) {
        PwrActivity activity = findActivity(cur_pin);
        const VcdValue* cur_pin_vcd_values = activity.vcdValues();
        if (cur_pin_vcd_values == nullptr) {
          continue;
        }

        auto emplace_result = vcd_values_ptr_to_n_event_.try_emplace(cur_pin_vcd_values, activity.nEvent());
        if (emplace_result.second) {
          int bus_width = cur_pin_vcd_values[0].busWidth();
          for (int bit_idx = 0; bit_idx < bus_width; ++bit_idx) {
            vcd_values_bit_pair_list_.emplace_back(cur_pin_vcd_values, bit_idx);
          }
        }
        var_val_ptr_to_value_bit_to_prev_right_bound_[cur_pin_vcd_values].try_emplace(activity.valueBit(), 0);
      }
    }
    getGateWaveformRangeMultiThread(start_time, end_time, global_pin_waveform_starts, global_pin_waveform_ends);
    timer_initial_wave_preparation.EndTiming();
  }
  n_multiple_output_gate_ = h_multiple_output_gates_.size();
  initPerGatePower();
  LOG_INFO << "Init gate data complete";

  // ----------copy auxiliary arrays to device----------
  utils::ScopedTimer timer_h2d_transfer("H2D Transfer");
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_waveform_starts_, sizeof(NEeventVal) * global_pin_waveform_starts.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_waveform_starts_, global_pin_waveform_starts.data(), sizeof(NEeventVal) * global_pin_waveform_starts.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_waveform_ends_, sizeof(NEeventVal) * global_pin_waveform_ends.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_waveform_ends_, global_pin_waveform_ends.data(), sizeof(NEeventVal) * global_pin_waveform_ends.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_default_states_, sizeof(VcdEventVal) * global_pin_default_states.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_default_states_, global_pin_default_states.data(), sizeof(VcdEventVal) * global_pin_default_states.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_voltages_, sizeof(VoltageVal) * global_pin_voltages.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_voltages_, global_pin_voltages.data(), sizeof(VoltageVal) * global_pin_voltages.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_rise_slews_, sizeof(SlewVal) * global_pin_rise_slews.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_rise_slews_, global_pin_rise_slews.data(), sizeof(SlewVal) * global_pin_rise_slews.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_fall_slews_, sizeof(SlewVal) * global_pin_fall_slews.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_fall_slews_, global_pin_fall_slews.data(), sizeof(SlewVal) * global_pin_fall_slews.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_load_capacitances_, sizeof(CapacitanceVal) * global_pin_load_capacitances.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_load_capacitances_, global_pin_load_capacitances.data(), sizeof(CapacitanceVal) * global_pin_load_capacitances.size(), cudaMemcpyHostToDevice));
  bool* tmp_is_clock = new bool[global_pin_is_clocks.size()];
  for (size_t global_pin_idx = 0; global_pin_idx < global_pin_is_clocks.size(); ++global_pin_idx) {
    tmp_is_clock[global_pin_idx] = global_pin_is_clocks.at(global_pin_idx) == 0 ? false : true;
  }
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_is_clocks_, sizeof(bool) * global_pin_is_clocks.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_is_clocks_, tmp_is_clock, sizeof(bool) * global_pin_is_clocks.size(), cudaMemcpyHostToDevice));
  if (!global_cell_arc_delays.empty()) {
    CHECK_CUDA_RUNTIME(cudaMalloc(&global_cell_arc_delays_, sizeof(DelayVal) * global_cell_arc_delays.size()));
    CHECK_CUDA_RUNTIME(cudaMemcpy(global_cell_arc_delays_, global_cell_arc_delays.data(), sizeof(DelayVal) * global_cell_arc_delays.size(), cudaMemcpyHostToDevice));
  } else {
    global_cell_arc_delays_ = nullptr;
  }
  timer_h2d_transfer.EndTiming();
  size_t accu_pin_count = 0;
  size_t accu_cell_arc_delay_count = 0;
  for (power::Gate* cur_gate: h_multiple_output_gates_) {
    cur_gate->setGateInfoByGlobalPointer(
      global_pin_waveform_starts_, global_pin_waveform_ends_, global_pin_waveform_starts, global_pin_waveform_ends,
      global_pin_default_states_,
      global_pin_voltages_,
      global_pin_rise_slews_, global_pin_fall_slews_,
      global_pin_load_capacitances_, 
      global_pin_is_clocks_,
      global_cell_arc_delays_,
      accu_pin_count,
      accu_cell_arc_delay_count
    );
    accu_pin_count += cur_gate->n_pin;
    accu_cell_arc_delay_count += static_cast<size_t>(cur_gate->n_output_pin) * cur_gate->n_input_pin * 4;
  }
  delete[] tmp_is_clock;
  const size_t total_pin_count = global_pin_waveform_starts.size();
  CUDA_MEM_STATS.set(CudaMemCategory::gate_aux, total_pin_count * (
    sizeof(NEeventVal) * 2
    + sizeof(VcdEventVal)
    + sizeof(VoltageVal)
    + sizeof(SlewVal) * 2
    + sizeof(CapacitanceVal)
    + sizeof(bool)
  ));
  CUDA_MEM_STATS.set(CudaMemCategory::delay, sizeof(DelayVal) * global_cell_arc_delays.size());
  // ----------end of copy auxiliary arrays to device----------
  // ---------------------------end of copy data to device side------------------------------

  // std::random_device rd;
  // std::mt19937 random_engine(rd());
  std::mt19937 random_engine(123456u);  // fixed random seed to keep deterministic
  std::shuffle(vcd_values_bit_pair_list_.begin(), vcd_values_bit_pair_list_.end(), random_engine);

  TIMEREND(INIT_GATE_DATA);
  DURATION_ms(INIT_GATE_DATA);
}

void 
CudaPower::copyGateDataToDeviceSide() 
{
  utils::ScopedTimer timer_copy_gate_data_to_device_side("copyGateDataToDeviceSide");
  TIMERSTART(COPY_GATE_DATA_TO_DEVICE);

  power::Gate* tmp_h_gates = nullptr;
  CHECK_CUDA_RUNTIME(cudaMallocHost(&tmp_h_gates, n_multiple_output_gate_ * sizeof(power::Gate)));
  NGateVal gate_count = 0;
  for (const power::Gate* cur_gate: h_multiple_output_gates_) {
    std::memcpy(tmp_h_gates + gate_count, cur_gate, sizeof(power::Gate));
    ++gate_count;
  }
  CHECK_CUDA_RUNTIME(cudaFree(d_multiple_output_gates_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_multiple_output_gates_, n_multiple_output_gate_ * sizeof(power::Gate)));
  utils::ScopedTimer timer_h2d_transfer("H2D Transfer");
  CHECK_CUDA_RUNTIME(cudaMemcpy(d_multiple_output_gates_, tmp_h_gates, n_multiple_output_gate_ * sizeof(power::Gate), cudaMemcpyHostToDevice));
  timer_h2d_transfer.EndTiming();
  CHECK_CUDA_RUNTIME(cudaFreeHost(tmp_h_gates));
  CUDA_MEM_STATS.set(CudaMemCategory::gate, n_multiple_output_gate_ * sizeof(power::Gate));

  TIMEREND(COPY_GATE_DATA_TO_DEVICE);
  DURATION_ms(COPY_GATE_DATA_TO_DEVICE);
  LOG_INFO << "Transferred gate data size: " << sizeof(power::Gate) * n_multiple_output_gate_ / (1024.0 * 1024 * 1024) << "GB";
}

void 
CudaPower::getGateWaveformRangeSingleThread(VcdEventTime start_time, VcdEventTime end_time, std::vector<NEeventVal>& h_global_pin_waveform_starts, std::vector<NEeventVal>& h_global_pin_waveform_ends)
{
  LOG_INFO << "Getting gate waveform range [" << start_time << ", " << end_time << ")...";
  TIMERSTART(GET_GATE_WAVEFORM_RANGE_SINGLE_THREAD);

  NEeventVal accu_event_count = 0;
  std::unordered_map<const VcdValue*, std::unordered_map<int, std::pair<NEeventVal, NEeventVal>>> pin_vcd_values_ptr_to_value_bit_to_range_map;  // avoid copy redundant waveforms
  for (const power::Gate* cur_gate: h_multiple_output_gates_) {
    // ---------------------------init pin information------------------------------
    std::vector<const Pin *> pins;
    NPinVal n_pin = 0, n_input_pin = 0, n_output_pin = 0;
    getPinInformation(cur_gate->inst, &pins, &n_pin, &n_input_pin, &n_output_pin);
    // ---------------------------end of init pin information------------------------------

    // ---------------------------waveform------------------------------
    // std::vector<NEeventVal> pin_waveform_starts(n_pin);
    // std::vector<NEeventVal> pin_waveform_ends(n_pin);
    getWaveform(cur_gate->inst, pins, start_time, end_time, accu_event_count, pin_vcd_values_ptr_to_value_bit_to_range_map, h_global_pin_waveform_starts, h_global_pin_waveform_ends);
    // ---------------------------end of waveform------------------------------

    // cur_gate->renewWaveformPointers(pin_waveform_starts.data(), pin_waveform_ends.data());
  }
  LOG_INFO << "Get gate waveform range complete";

  TIMEREND(GET_GATE_WAVEFORM_RANGE_SINGLE_THREAD);
  DURATION_ms(GET_GATE_WAVEFORM_RANGE_SINGLE_THREAD);
}

// accelerated version provided by ChatGPT
void
CudaPower::getGateWaveformRangeMultiThread(
  VcdEventTime start_time,
  VcdEventTime end_time,
  std::vector<NEeventVal>& h_global_pin_waveform_starts,
  std::vector<NEeventVal>& h_global_pin_waveform_ends
) {
  TIMERSTART(GET_GATE_WAVEFORM_RANGE_MULTI_THREAD);

  const bool is_the_first_time_interval = (start_time == 0);
  const int n_threads = G_CONFIG.nums.multi_thread_number;

  struct VcdBitKey {
    const VcdValue* ptr;
    int bit;
  };
  struct VcdBitKeyHash {
    size_t operator()(const VcdBitKey& k) const noexcept {
      size_t h1 = std::hash<const void*>{}(static_cast<const void*>(k.ptr));
      size_t h2 = std::hash<int>{}(k.bit);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };
  struct VcdBitKeyEq {
    bool operator()(const VcdBitKey& a, const VcdBitKey& b) const noexcept {
      return a.ptr == b.ptr && a.bit == b.bit;
    }
  };

  const size_t n_pairs = vcd_values_bit_pair_list_.size();

  // -------------------------
  // Phase 0: 预取 prev_right_bound（串行，避免并发读写 unordered_map）
  // -------------------------
  std::vector<uint8_t> used(n_pairs, 0);
  std::vector<NEeventVal> prev_right_bound(n_pairs, 0);
  std::vector<NEeventVal> total_event_in_vcd(n_pairs, 0);

  for (size_t i = 0; i < n_pairs; ++i) {
    const auto& [vcd_values, bit_idx] = vcd_values_bit_pair_list_[i];

    auto it_outer = var_val_ptr_to_value_bit_to_prev_right_bound_.find(vcd_values);
    if (it_outer == var_val_ptr_to_value_bit_to_prev_right_bound_.end()) {
      continue;
    }
    auto& inner = it_outer->second;
    auto it_inner = inner.find(bit_idx);
    if (it_inner == inner.end()) {
      continue;
    }

    used[i] = 1;
    prev_right_bound[i] = it_inner->second;
    total_event_in_vcd[i] = vcd_values_ptr_to_n_event_.at(vcd_values);
  }

  // -------------------------
  // Phase 1: 计算每个 pair 的 vcd index 区间以及事件数（并行）
  // -------------------------
  std::vector<NEeventVal> start_idx_in_vcd(n_pairs, 0);
  std::vector<NEeventVal> end_idx_in_vcd(n_pairs, 0);
  std::vector<NEeventVal> n_event_in_range(n_pairs, 0);

  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for (size_t i = 0; i < n_pairs; ++i) {
    if (!used[i]) {
      continue;
    }

    const auto& [vcd_values, bit_idx] = vcd_values_bit_pair_list_[i];

    const NEeventVal prb = prev_right_bound[i];
    const NEeventVal start_idx = is_the_first_time_interval ? 0 : (prb > 0 ? prb - 1 : 0);

    const NEeventVal n_total = total_event_in_vcd[i];
    const NEeventVal end_idx =
      (end_time >= max_event_time_)
        ? n_total
        : getEventIdxByTime(
            vcd_values,
            n_total,
            end_time,
            is_the_first_time_interval ? 0 : prb
          );

    start_idx_in_vcd[i] = start_idx;
    end_idx_in_vcd[i] = end_idx;
    n_event_in_range[i] = (end_idx >= start_idx) ? (end_idx - start_idx) : 0;
  }

  // -------------------------
  // Phase 2: 前缀和分配全局 event buffer（串行）
  // -------------------------
  std::vector<NEeventVal> event_range_start(n_pairs, 0);
  std::vector<NEeventVal> event_range_end(n_pairs, 0);

  NEeventVal accu_event_count = 0;
  for (size_t i = 0; i < n_pairs; ++i) {
    if (!used[i]) {
      continue;
    }
    const NEeventVal off = accu_event_count;
    const NEeventVal cnt = n_event_in_range[i];
    event_range_start[i] = off;
    event_range_end[i] = off + cnt;
    accu_event_count += cnt;
  }

  if (accu_event_count > G_CONFIG.nums.max_event_num) {
    LOG_ERROR << "accu_event_count: " << accu_event_count
              << " exceeds G_CONFIG.nums.max_event_num: " << G_CONFIG.nums.max_event_num;
  }

  // -------------------------
  // Phase 3: 更新 prev_right_bound（串行）
  // -------------------------
  for (size_t i = 0; i < n_pairs; ++i) {
    if (!used[i]) {
      continue;
    }
    const auto& [vcd_values, bit_idx] = vcd_values_bit_pair_list_[i];
    var_val_ptr_to_value_bit_to_prev_right_bound_[vcd_values][bit_idx] = end_idx_in_vcd[i];
  }

  // -------------------------
  // Phase 4: (ptr,bit) -> pair index 映射（只存 index）
  // -------------------------
  std::unordered_map<VcdBitKey, size_t, VcdBitKeyHash, VcdBitKeyEq> pair_index;
  pair_index.reserve(n_pairs);

  for (size_t i = 0; i < n_pairs; ++i) {
    if (!used[i]) {
      continue;
    }
    const auto& [vcd_values, bit_idx] = vcd_values_bit_pair_list_[i];
    pair_index.emplace(VcdBitKey{vcd_values, bit_idx}, i);
  }

  // -------------------------
  // Phase 5: 写 events_（并行，无锁）
  //   关键优化：不调用 setEvent；不查 vcd_value_to_int_map_；直接写裸指针 events_
  // -------------------------
  sta::power::Event* __restrict events = events_;
  const auto& v2i = vcdValueToEventValTable();

  #pragma omp parallel for num_threads(n_threads) schedule(guided, 1)
  for (size_t i = 0; i < n_pairs; ++i) {
    if (!used[i]) {
      continue;
    }

    const auto& [vcd_values, bit_idx] = vcd_values_bit_pair_list_[i];
    const NEeventVal start_idx = start_idx_in_vcd[i];
    const NEeventVal end_idx = end_idx_in_vcd[i];
    const NEeventVal base = event_range_start[i];

    for (NEeventVal pos = start_idx; pos < end_idx; ++pos) {
      const NEeventVal out_idx = base + (pos - start_idx);
      assert(out_idx < G_CONFIG.nums.max_event_num);

      events[out_idx].time = vcd_values[pos].time();
      events[out_idx].val = v2i[static_cast<unsigned char>(vcd_values[pos].value(bit_idx))];
    }
  }

  // -------------------------
  // Phase 6: gate->pin waveform range 收集（预先算偏移 + 并行写最终数组）
  // -------------------------
  const size_t n_gates = h_multiple_output_gates_.size();
  std::vector<size_t> gate_pin_offset(n_gates + 1, 0);

  for (size_t gate_idx = 0; gate_idx < n_gates; ++gate_idx) {
    const power::Gate* cur_gate = h_multiple_output_gates_[gate_idx];
    const auto& pins = inst_to_pins_.at(cur_gate->inst);
    gate_pin_offset[gate_idx + 1] = gate_pin_offset[gate_idx] + pins.size();
  }

  const size_t total_pin_count = gate_pin_offset.back();
  h_global_pin_waveform_starts.assign(total_pin_count, INVALID_WAVEFORM_PTR);
  h_global_pin_waveform_ends.assign(total_pin_count, INVALID_WAVEFORM_PTR);

  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for (size_t gate_idx = 0; gate_idx < n_gates; ++gate_idx) {
    const power::Gate* cur_gate = h_multiple_output_gates_[gate_idx];
    const auto& pins = inst_to_pins_.at(cur_gate->inst);

    const size_t base_pin = gate_pin_offset[gate_idx];
    for (NPinVal pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
      const Pin* cur_pin = pins[pin_idx];
      const PwrActivity& activity = findActivity(cur_pin);
      const VcdValue* cur_pin_vcd_values = activity.vcdValues();
      const int cur_pin_value_bit = activity.valueBit();

      NEeventVal start_ptr = INVALID_WAVEFORM_PTR;
      NEeventVal end_ptr = INVALID_WAVEFORM_PTR;

      if (cur_pin_vcd_values != nullptr) {
        auto it = pair_index.find(VcdBitKey{cur_pin_vcd_values, cur_pin_value_bit});
        if (it != pair_index.end()) {
          const size_t pi = it->second;
          start_ptr = event_range_start[pi];
          end_ptr = event_range_end[pi];
        }
      }

      h_global_pin_waveform_starts[base_pin + pin_idx] = start_ptr;
      h_global_pin_waveform_ends[base_pin + pin_idx] = end_ptr;
    }
  }

  TIMEREND(GET_GATE_WAVEFORM_RANGE_MULTI_THREAD);
  DURATION_ms(GET_GATE_WAVEFORM_RANGE_MULTI_THREAD);
}

void 
CudaPower::renewGateWaveform(const std::vector<NEeventVal>& h_global_pin_waveform_starts, const std::vector<NEeventVal>& h_global_pin_waveform_ends)
{
  LOG_INFO << "Renewing gate waveform... ";
  utils::ScopedTimer renew_timer("renewGateWaveform");
  TIMERSTART(RENEW_WAVEFORM);
  
  utils::ScopedTimer timer_h2d_transfer("H2D Transfer");
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_waveform_starts_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_waveform_starts_, sizeof(NEeventVal) * h_global_pin_waveform_starts.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_waveform_starts_, h_global_pin_waveform_starts.data(), sizeof(NEeventVal) * h_global_pin_waveform_starts.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_waveform_ends_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_pin_waveform_ends_, sizeof(NEeventVal) * h_global_pin_waveform_ends.size()));
  CHECK_CUDA_RUNTIME(cudaMemcpy(global_pin_waveform_ends_, h_global_pin_waveform_ends.data(), sizeof(NEeventVal) * h_global_pin_waveform_ends.size(), cudaMemcpyHostToDevice));
  timer_h2d_transfer.EndTiming();
  CUDA_MEM_STATS.set(
    CudaMemCategory::gate_aux,
    (h_global_pin_waveform_starts.size() * (sizeof(VcdEventVal) + sizeof(VoltageVal) + sizeof(SlewVal) * 2 + sizeof(CapacitanceVal) + sizeof(bool)))
    + sizeof(NEeventVal) * (h_global_pin_waveform_starts.size() + h_global_pin_waveform_ends.size())
  );
  size_t accu_pin_count = 0;
  for (power::Gate* cur_gate: h_multiple_output_gates_) {
    cur_gate->renewWaveformPointers(global_pin_waveform_starts_, global_pin_waveform_ends_, h_global_pin_waveform_starts, h_global_pin_waveform_ends, accu_pin_count);
    accu_pin_count += cur_gate->n_pin;
  }
  LOG_INFO << "Renew gate waveform complete.";

  TIMEREND(RENEW_WAVEFORM);
  DURATION_ms(RENEW_WAVEFORM);
}

void
CudaPower::copyWaveformToDeviceSide() {
  utils::ScopedTimer timer_copy_waveform_to_device_side("copyWaveformToDeviceSide");
  TIMERSTART(COPY_WAVEFORM_TO_DEVICE);

  utils::ScopedTimer timer_h2d_transfer("H2D Transfer");
  CHECK_CUDA_RUNTIME(cudaMemcpy(d_events_, events_, G_CONFIG.nums.max_event_num * sizeof(power::Event), cudaMemcpyHostToDevice));
  timer_h2d_transfer.EndTiming();

  TIMEREND(COPY_WAVEFORM_TO_DEVICE);
  DURATION_ms(COPY_WAVEFORM_TO_DEVICE);
}
// ------------------------------end of time inteval scheduling------------------------------

void
CudaPower::getWaveform(
  const Instance* inst,
  const std::vector<const Pin *>& pins,
  VcdEventTime start_time, 
  VcdEventTime end_time,
  // Return values.
  NEeventVal& accu_event_count,
  std::unordered_map<const VcdValue*, std::unordered_map<int, std::pair<NEeventVal, NEeventVal>>>& pin_vcd_values_ptr_to_value_bit_to_range_map,
  std::vector<NEeventVal>& global_pin_waveform_starts, 
  std::vector<NEeventVal>& global_pin_waveform_ends
) {
  bool is_the_first_time_interval = (start_time == 0);

  for (NPinVal pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
    const Pin *cur_pin = pins.at(pin_idx);
    const LibertyPort *cur_port = network_->libertyPort(cur_pin);
    PwrActivity activity = findActivity(cur_pin);
    const VcdValue *cur_pin_vcd_values = activity.vcdValues();
    const int cur_pin_value_bit = activity.valueBit();
    NEeventVal cur_pin_waveform_start = INVALID_WAVEFORM_PTR, cur_pin_waveform_end = INVALID_WAVEFORM_PTR;
    if (cur_pin_vcd_values != nullptr) {
      if (vcd_values_ptr_to_n_event_.count(cur_pin_vcd_values) == 0) {
        vcd_values_ptr_to_n_event_.emplace(cur_pin_vcd_values, activity.nEvent());
        int bus_width = cur_pin_vcd_values[0].busWidth();
        for (int bit_idx = 0; bit_idx < bus_width; ++bit_idx) {
          vcd_values_bit_pair_list_.emplace_back(cur_pin_vcd_values, bit_idx);
        }
      }

      if (pin_vcd_values_ptr_to_value_bit_to_range_map.count(cur_pin_vcd_values) && pin_vcd_values_ptr_to_value_bit_to_range_map.at(cur_pin_vcd_values).count(cur_pin_value_bit)) {
        cur_pin_waveform_start = pin_vcd_values_ptr_to_value_bit_to_range_map.at(cur_pin_vcd_values).at(cur_pin_value_bit).first;
        cur_pin_waveform_end = pin_vcd_values_ptr_to_value_bit_to_range_map.at(cur_pin_vcd_values).at(cur_pin_value_bit).second;
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << network_->pathName(cur_pin) << " " << activity.originName() << " " << cur_pin_vcd_values << " value_bit: " << cur_pin_value_bit << " existed " << cur_pin_waveform_start << " " << cur_pin_waveform_end;
        }
      } else {
        // get events in range [start_time, end_time)
        NEeventVal start_idx_in_vcd_values = is_the_first_time_interval ? 0 : var_val_ptr_to_value_bit_to_prev_right_bound_.at(cur_pin_vcd_values).at(cur_pin_value_bit) - 1;  // minus one for getting the last state of the previous time interval as previous state
        NEeventVal end_idx_in_vcd_values = end_time >= max_event_time_ ? activity.nEvent() : getEventIdxByTime(cur_pin_vcd_values, activity.nEvent(), end_time, is_the_first_time_interval ? 0 : var_val_ptr_to_value_bit_to_prev_right_bound_.at(cur_pin_vcd_values).at(cur_pin_value_bit));
        var_val_ptr_to_value_bit_to_prev_right_bound_[cur_pin_vcd_values][cur_pin_value_bit] = end_idx_in_vcd_values;
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << network_->pathName(cur_pin) << " getWaveform: start_time: " << start_time << " end_time: " << end_time << " activity.nEvent(): " << activity.nEvent() << " start_idx_in_vcd_values: " << start_idx_in_vcd_values << " end_idx_in_vcd_values: " << end_idx_in_vcd_values;
        }
        NEeventVal n_event_in_range = (end_idx_in_vcd_values - start_idx_in_vcd_values);  // we do not add one here, since the last one is out of range of time
        cur_pin_waveform_start = accu_event_count;
        cur_pin_waveform_end = accu_event_count + n_event_in_range;
        if (cur_pin_waveform_end > G_CONFIG.nums.max_event_num) {
          LOG_ERROR << "GPU event allocation overflow!";
        }

        pin_vcd_values_ptr_to_value_bit_to_range_map[cur_pin_vcd_values].emplace(cur_pin_value_bit, std::pair{cur_pin_waveform_start, cur_pin_waveform_end});
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << network_->pathName(cur_pin) << " " << activity.originName() << " " << cur_pin_vcd_values << " value_bit: " << cur_pin_value_bit << " new " << cur_pin_waveform_start << " " << cur_pin_waveform_end;
        }
        //----------get cycle accurate glitch information----------
        // std::vector<bool> event_glitch_flags(end_idx_in_vcd_values - start_idx_in_vcd_values, false);
        // if (!cur_port->isClock()) {
        //   getEventClockCycleBasedGlitchFlags(cur_pin_vcd_values, start_idx_in_vcd_values, end_idx_in_vcd_values, event_glitch_flags);
        // }
        //----------end of get cycle accurate glitch information----------
        for (NEeventVal pos = start_idx_in_vcd_values; pos < end_idx_in_vcd_values; ++pos) {
          setEvent(cur_pin_vcd_values[pos].time(), vcd_value_to_int_map_.at(cur_pin_vcd_values[pos].value(cur_pin_value_bit)), false, cur_pin_waveform_start + (pos - start_idx_in_vcd_values));
        }

        accu_event_count += n_event_in_range;
      }
    } else {
      if (LOG_DEBUG_FLAG) {
        LOG_DEBUG << network_->pathName(cur_pin) << " " << activity.originName() << " " << cur_pin_vcd_values << " nullptr " << cur_pin_waveform_start << " " << cur_pin_waveform_end;
      }
    }

    global_pin_waveform_starts.push_back(cur_pin_waveform_start);
    global_pin_waveform_ends.push_back(cur_pin_waveform_end);
  }
}

void 
CudaPower::getDelay(
  const std::vector<const Pin *>& pins, 
  NPinVal n_pin, 
  NPinVal n_input_pin, 
  const DcalcAnalysisPt *dcalc_ap,
  // Return values.
  std::vector<DelayVal>& global_cell_arc_delays
) const
{
  assert(pins.size() == n_pin);
  const NPinVal n_output_pin = n_pin - n_input_pin;
  const size_t delay_base = global_cell_arc_delays.size();
  global_cell_arc_delays.resize(delay_base + static_cast<size_t>(n_output_pin) * n_input_pin * 4, INVALID_DELAY_VAL);
  for (NPinVal out_pin_idx = n_input_pin; out_pin_idx < n_pin; ++out_pin_idx) {
    const Pin* out_pin = pins.at(out_pin_idx);
    const NPinVal output_pin_local_idx = out_pin_idx - n_input_pin;
    for (NPinVal in_pin_idx = 0; in_pin_idx < n_input_pin; ++in_pin_idx) {
      const Pin* in_pin = pins.at(in_pin_idx);
      for (RiseFall *to_rf : RiseFall::range()) {
        for (RiseFall *from_rf : RiseFall::range()) {
          const size_t delay_idx = delay_base 
            + (static_cast<size_t>(output_pin_local_idx) * n_input_pin + in_pin_idx) * 4
            + riseFallEdgeIndex(from_rf, to_rf);
          Edge *edge;
          const TimingArc *arc;
          graph_->gateEdgeArc(in_pin, from_rf, out_pin, to_rf, edge, arc);
          if (edge && arc) {
            ArcDelay cell_arc_delay = graph_->arcDelay(edge, arc, dcalc_ap->index());
            global_cell_arc_delays.at(delay_idx) = cell_arc_delay;
            if (LOG_DEBUG_FLAG) {
              LOG_DEBUG << network_->pathName(in_pin) << " -> " << network_->pathName(out_pin) << " from_rf: " << from_rf->asString() << " to_rf: " << to_rf->asString() << " cell_arc_delay: " << cell_arc_delay;
            }
          }
        }
      }
    }
  }
}

const CudaPower::LeakagePowerData&
CudaPower::getLeakagePowerData(
  const LibertyCell *cell,
  const LibertyCell *corner_cell,
  NPinVal n_pin,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
)
{
  auto cell_leakage_iter = cell_to_leakage_power_data_.find(corner_cell);
  if (cell_leakage_iter != cell_to_leakage_power_data_.end()) {
    return cell_leakage_iter->second;
  }

  LeakagePowerData leakage_power_data;
  getLeakagePower(
    cell,
    corner_cell,
    n_pin,
    port_name_to_idx_map,
    leakage_power_data.leakage_power_values,
    leakage_power_data.default_leakage_power_val,
    leakage_power_data.default_leakage_exists
  );

  return cell_to_leakage_power_data_.emplace(corner_cell, std::move(leakage_power_data)).first->second;
}

const CudaPower::InternalPowerData&
CudaPower::getInternalPower(
  const Instance *inst, 
  const LibertyCell *corner_cell, 
  const DcalcAnalysisPt *dcalc_ap, 
  const NPinVal n_pin,
  const std::vector<const Pin *>& pins,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
)  
{
  auto cell_lut_iter = cell_to_internal_power_data_.find(corner_cell);
  if (cell_lut_iter != cell_to_internal_power_data_.end()) {
    return cell_lut_iter->second;
  }

  InternalPowerData internal_power_data;
  internal_power_data.input_ports_internal_power_LUTs.reserve(pins.size());
  internal_power_data.input_ports_n_internal_power_LUTs_index_by_order.reserve(pins.size());
  internal_power_data.input_ports_internal_power_LUTs_index_by_order.reserve(pins.size());
  internal_power_data.output_ports_internal_power_LUTs.reserve(pins.size());
  internal_power_data.output_ports_n_internal_power_LUTs_index_by_order.reserve(pins.size());
  internal_power_data.output_ports_internal_power_LUTs_index_by_order.reserve(pins.size());

  for (const Pin* cur_pin: pins) {
    const LibertyPort *cur_port = network_->libertyPort(cur_pin);
    if (cur_port) {
      if (cur_port->direction()->isAnyInput()) {
        OneDimensionalLUTPair** d_cur_port_internal_power_LUTs = nullptr;
        NStateVal cur_port_n_internal_power_LUTs_indexed_by_order = 0;
        OneDimensionalLUTPair** d_cur_port_internal_power_LUTs_indexed_by_order = nullptr;
        if (input_port_to_internal_luts_map_.find(cur_port) != input_port_to_internal_luts_map_.end()) {
          d_cur_port_internal_power_LUTs = input_port_to_internal_luts_map_.at(cur_port);
          cur_port_n_internal_power_LUTs_indexed_by_order = input_port_to_n_internal_LUTs_indexed_by_order_map_.at(cur_port);
          d_cur_port_internal_power_LUTs_indexed_by_order = input_port_to_internal_luts_indexed_by_order_map_.at(cur_port);
        } else {
          getInputInternalPower(corner_cell, cur_port, dcalc_ap, n_pin, port_name_to_idx_map, d_cur_port_internal_power_LUTs, cur_port_n_internal_power_LUTs_indexed_by_order, d_cur_port_internal_power_LUTs_indexed_by_order);
          input_port_to_internal_luts_map_[cur_port] = d_cur_port_internal_power_LUTs;
          input_port_to_n_internal_LUTs_indexed_by_order_map_[cur_port] = cur_port_n_internal_power_LUTs_indexed_by_order;
          input_port_to_internal_luts_indexed_by_order_map_[cur_port] = d_cur_port_internal_power_LUTs_indexed_by_order;
        }

        internal_power_data.input_ports_internal_power_LUTs.push_back(d_cur_port_internal_power_LUTs);
        internal_power_data.input_ports_n_internal_power_LUTs_index_by_order.push_back(cur_port_n_internal_power_LUTs_indexed_by_order);
        internal_power_data.input_ports_internal_power_LUTs_index_by_order.push_back(d_cur_port_internal_power_LUTs_indexed_by_order);
      } else if (cur_port->direction()->isAnyOutput()) {
        TwoDimensionalLUTPair*** d_cur_port_internal_power_LUTs = nullptr;
        NStateVal* d_cur_port_n_internal_power_LUTs_indexed_by_order = nullptr;
        TwoDimensionalLUTPair*** d_cur_port_internal_power_LUTs_indexed_by_order = nullptr;
        if (output_port_to_internal_luts_map_.find(cur_port) != output_port_to_internal_luts_map_.end()) {
          d_cur_port_internal_power_LUTs = output_port_to_internal_luts_map_.at(cur_port);
          d_cur_port_n_internal_power_LUTs_indexed_by_order = output_port_to_n_internal_LUTs_indexed_by_order_map_.at(cur_port);
          d_cur_port_internal_power_LUTs_indexed_by_order = output_port_to_internal_luts_indexed_by_order_map_.at(cur_port);
        } else {
          getOutputInternalPower(inst, corner_cell, cur_port, dcalc_ap, n_pin, port_name_to_idx_map, d_cur_port_internal_power_LUTs, d_cur_port_n_internal_power_LUTs_indexed_by_order, d_cur_port_internal_power_LUTs_indexed_by_order);
          output_port_to_internal_luts_map_[cur_port] = d_cur_port_internal_power_LUTs;
          output_port_to_n_internal_LUTs_indexed_by_order_map_[cur_port] = d_cur_port_n_internal_power_LUTs_indexed_by_order;
          output_port_to_internal_luts_indexed_by_order_map_[cur_port] = d_cur_port_internal_power_LUTs_indexed_by_order;
        }

        internal_power_data.output_ports_internal_power_LUTs.push_back(d_cur_port_internal_power_LUTs);
        internal_power_data.output_ports_n_internal_power_LUTs_index_by_order.push_back(d_cur_port_n_internal_power_LUTs_indexed_by_order);
        internal_power_data.output_ports_internal_power_LUTs_index_by_order.push_back(d_cur_port_internal_power_LUTs_indexed_by_order);
      }
    }
  }

  return cell_to_internal_power_data_.emplace(corner_cell, std::move(internal_power_data)).first->second;
}

void
CudaPower::getInputInternalPower(
  const LibertyCell *corner_cell, 
  const LibertyPort *port, 
  const DcalcAnalysisPt *dcalc_ap, 
  const NPinVal n_pin,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  // Return values.
  OneDimensionalLUTPair**& d_cur_port_internal_power_LUTs,
  NStateVal& n_cur_port_internal_power_LUTs_indexed_by_order,
  OneDimensionalLUTPair**& d_cur_port_internal_power_LUTs_indexed_by_order
) const
{
  const LibertyPort *corner_port = port->cornerPort(dcalc_ap);
  if (!corner_cell || !corner_port) {
    LOG_WARN << "No valid corner cell and port for " << corner_cell->name() << "/" << port->name();
    return;
  }

  InternalPowerSeq internal_pwrs;
  corner_cell->internalPowers(corner_port, internal_pwrs);
  if (internal_pwrs.empty()) {
    LOG_WARN << "No input internal power table for input pin " << corner_cell->name() << "/" << port->name();
    return;
  }

  NStateVal n_state = INVALID_N_STATE;
  if (n_pin > G_CONFIG.nums.max_n_pin_for_internal_power) {  // too large
    if (internal_pwrs.size() != 0 && internal_pwrs.size() != 1) {
      LOG_WARN << corner_cell->name() << "/" << port->name() << " has too large n_pin " << n_pin << " with internal_pwrs.size() " << internal_pwrs.size();
    }
  } else {
    n_state = 1 << n_pin;
  }

  std::vector<OneDimensionalLUTPair*> h_cur_port_internal_power_LUTs(n_state == INVALID_N_STATE ? 0 : n_state, nullptr);
  std::vector<OneDimensionalLUTPair*> h_cur_port_internal_power_LUTs_indexed_by_order;
  for (const InternalPower *pwr: internal_pwrs) {  // TODO sort internal power to make unconditioned one ahead
    std::vector<VcdEventVal> when_state_vec;
    const bool when_satisfiable = ::power::utils::getWhenPinStates(pwr->when(), port_name_to_idx_map, when_state_vec);
    OneDimensionalLUTPair* d_1d_lut_pair_ptr = new OneDimensionalLUTPair;
    CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(OneDimensionalLUTPair));
    for (RiseFall *rf : RiseFall::range()) {
      OneDimensionalLUT* d_1d_lut_ptr = nullptr;
      if (const std::shared_ptr<const Table1> lookup_table = std::dynamic_pointer_cast<const Table1>(pwr->lookupTable(rf)); lookup_table) {
        d_1d_lut_ptr = new OneDimensionalLUT(lookup_table->axis1()->size(), lookup_table->axis1()->values()->data(), lookup_table->values()->data());
        CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(OneDimensionalLUT));
      } else if (const std::shared_ptr<const Table0> scalar_table = std::dynamic_pointer_cast<const Table0>(pwr->lookupTable(rf)); scalar_table) {
        d_1d_lut_ptr = new OneDimensionalLUT(0, nullptr, nullptr, true, scalar_table->value(0,0,0));
        CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(OneDimensionalLUT));
      } else {
        LOG_ERROR << "CudaPower::getInputInternalPower convert table pointer to table1 and table0 pointer failed for " << corner_cell->name() << "/" << port->name();
      }

      if (strcmp(rf->name(), "rise") == 0) {
        d_1d_lut_pair_ptr->setTable(d_1d_lut_ptr, RISE);
      } else if (strcmp(rf->name(), "fall") == 0) {
        d_1d_lut_pair_ptr->setTable(d_1d_lut_ptr, FALL);
      } else {
        LOG_ERROR << "CudaPower::getInputInternalPower unexpected rf name: " << rf->name();
      }
    }

    d_1d_lut_pair_ptr->setWhenState(when_state_vec, when_satisfiable);

    if (n_state != INVALID_N_STATE) {
      utils::ScopedTimer timer_bsim_construction("BSIM/state-index mapping construction");
      assert(h_cur_port_internal_power_LUTs.size() != 0);
      std::vector<NStateVal> matched_state_idxs;
      getStateIdx(port_name_to_idx_map, pwr->when(), matched_state_idxs);
      for (const NStateVal matched_state_idx: matched_state_idxs) {
        h_cur_port_internal_power_LUTs[matched_state_idx] = d_1d_lut_pair_ptr;
      }
    }
    h_cur_port_internal_power_LUTs_indexed_by_order.push_back(d_1d_lut_pair_ptr);
  }

  // generate results
  d_cur_port_internal_power_LUTs = nullptr;
  if (!h_cur_port_internal_power_LUTs.empty()) {
    CHECK_CUDA_RUNTIME(cudaMalloc(&d_cur_port_internal_power_LUTs, sizeof(OneDimensionalLUTPair*) * h_cur_port_internal_power_LUTs.size()));
    CHECK_CUDA_RUNTIME(cudaMemcpy(d_cur_port_internal_power_LUTs, h_cur_port_internal_power_LUTs.data(), sizeof(OneDimensionalLUTPair*) * h_cur_port_internal_power_LUTs.size(), cudaMemcpyHostToDevice));
    CUDA_MEM_STATS.add(CudaMemCategory::bsim_lut_index, sizeof(OneDimensionalLUTPair*) * h_cur_port_internal_power_LUTs.size());
  }
  n_cur_port_internal_power_LUTs_indexed_by_order = 0;
  d_cur_port_internal_power_LUTs_indexed_by_order = nullptr;
  if (!h_cur_port_internal_power_LUTs_indexed_by_order.empty()) {
    n_cur_port_internal_power_LUTs_indexed_by_order = h_cur_port_internal_power_LUTs_indexed_by_order.size();
    CHECK_CUDA_RUNTIME(cudaMalloc(&d_cur_port_internal_power_LUTs_indexed_by_order, sizeof(OneDimensionalLUTPair*) * h_cur_port_internal_power_LUTs_indexed_by_order.size()));
    CHECK_CUDA_RUNTIME(cudaMemcpy(d_cur_port_internal_power_LUTs_indexed_by_order, h_cur_port_internal_power_LUTs_indexed_by_order.data(), sizeof(OneDimensionalLUTPair*) * h_cur_port_internal_power_LUTs_indexed_by_order.size(), cudaMemcpyHostToDevice));
    CUDA_MEM_STATS.add(CudaMemCategory::fallback_lut_index, sizeof(OneDimensionalLUTPair*) * h_cur_port_internal_power_LUTs_indexed_by_order.size());
  }
}

void
CudaPower::getOutputInternalPower(
  const Instance *inst,
  const LibertyCell *corner_cell, 
  const LibertyPort *port, 
  const DcalcAnalysisPt *dcalc_ap,
  const NPinVal n_pin,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  // Return values.
  TwoDimensionalLUTPair***& d_cur_port_internal_power_LUTs,
  NStateVal*& d_cur_port_n_internal_power_LUTs_indexed_by_order,
  TwoDimensionalLUTPair***& d_cur_port_internal_power_LUTs_indexed_by_order
) const
{
  // ----------------init----------------
  assert(n_pin == port_name_to_idx_map.size());

  const LibertyPort *corner_port = port->cornerPort(dcalc_ap);
  if (!corner_cell || !corner_port) {
    LOG_WARN << "No valid corner cell and port for " << corner_cell->name() << "/" << port->name();
    return;
  }

  InternalPowerSeq internal_pwrs;
  corner_cell->internalPowers(corner_port, internal_pwrs);
  if (internal_pwrs.empty()) {
    LOG_WARN << "No input internal power table for output pin " << corner_cell->name() << "/" << port->name();
    return;
  }

  NStateVal n_state = INVALID_N_STATE;
  if (n_pin > G_CONFIG.nums.max_n_pin_for_internal_power) {  // too large
    if (internal_pwrs.size() != 0 && internal_pwrs.size() != 1) {
      LOG_WARN << corner_cell->name() << "/" << port->name() << " has too large n_pin " << n_pin << " with internal_pwrs.size() " << internal_pwrs.size();
    }
  } else {
    n_state = 1 << n_pin;
  }
  // ----------------end of init----------------

  // ----------------aggregate InternalPower* by related_pin first----------------
  std::unordered_map<std::string, std::vector<const InternalPower*>> port_to_internal_pwrs_map;
  // std::vector<const LibertyPort*> sorted_related_ports(n_pin);
  bool no_related_pin_pwr_exists = false;  // if no_related_pin_pwr_exists, that means it can be used as a default internel power lut
  for (const InternalPower *pwr: internal_pwrs) { 
    const LibertyPort *from_corner_port = pwr->relatedPort();
    if (from_corner_port) {
      port_to_internal_pwrs_map[from_corner_port->name()].push_back(pwr);
      // sorted_related_ports[port_name_to_idx_map.at(from_corner_port->name())] = from_corner_port; 
    } else {
      no_related_pin_pwr_exists = true;
      port_to_internal_pwrs_map[INVALID_PORT_NAME].push_back(pwr);
    }
  }
  // assert(!no_related_pin_pwr_exists || (no_related_pin_pwr_exists && port_to_internal_pwrs_map.size() == 1));
  // if (no_related_pin_pwr_exists && port_to_internal_pwrs_map.size() != 1) { // make sure that exists and only exists no related pin internal powers
  //   LOG_ERROR << "CudaPower::getOutputInternalPower exists related pin and no related pin internal powers at the same time.";
  // }
  // ----------------end of aggregate InternalPower* by related_pin first----------------

  // ----------------generate LUTs for each related pin----------------
  std::vector<std::vector<TwoDimensionalLUTPair*>> h_cur_port_internal_power_LUTs(n_pin);
  std::vector<std::vector<TwoDimensionalLUTPair*>> h_cur_port_internal_power_LUTs_indexed_by_order(n_pin);
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {
    const Pin *cur_related_pin = pin_iter->next();
    LibertyPort *cur_related_port = network_->libertyPort(cur_related_pin);
    if (!cur_related_port) {
      LOG_ERROR << "CudaPower::getOutputInternalPower get libertyPort error for " << network_->pathName(cur_related_pin);
    }
    if (!cur_related_port->direction()->isAnyInput()) {  // only of input port
      continue;
    }
    const NPinVal cur_port_idx = port_name_to_idx_map.at(cur_related_port->name());
    auto& h_cur_related_port_internal_power_LUTs = h_cur_port_internal_power_LUTs[cur_port_idx];
    auto& h_cur_related_port_internal_power_LUTs_indexed_by_order = h_cur_port_internal_power_LUTs_indexed_by_order[cur_port_idx];
    if (port_to_internal_pwrs_map.find(cur_related_port->name()) == port_to_internal_pwrs_map.end() && !no_related_pin_pwr_exists) {
      LOG_WARN << "CudaPower::getOutputInternalPower no internal power found for related_pin " << network_->pathName(cur_related_pin);
      h_cur_related_port_internal_power_LUTs.resize(0);  // TODO decide this logic, whether make it zero or the average of all LUTs
      continue;
    } else {
      h_cur_related_port_internal_power_LUTs.resize(n_state == INVALID_N_STATE ? 0 : n_state, nullptr);
    }

    const auto& cur_related_port_internal_pwrs = port_to_internal_pwrs_map.find(cur_related_port->name()) == port_to_internal_pwrs_map.end() ? 
      port_to_internal_pwrs_map.at(INVALID_PORT_NAME) : port_to_internal_pwrs_map.at(cur_related_port->name());
    for (const InternalPower *pwr: cur_related_port_internal_pwrs) {  // TODO sort internal power to make unconditioned one ahead
      std::vector<VcdEventVal> when_state_vec;
      const bool when_satisfiable = ::power::utils::getWhenPinStates(pwr->when(), port_name_to_idx_map, when_state_vec);
      TwoDimensionalLUTPair* d_2d_lut_pair_ptr = new TwoDimensionalLUTPair;
      CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(TwoDimensionalLUTPair));
      for (RiseFall *rf : RiseFall::range()) {
        TwoDimensionalLUT* d_2d_lut_ptr = nullptr;
        if (const std::shared_ptr<const Table2> lookup_table = std::dynamic_pointer_cast<const Table2>(pwr->lookupTable(rf)); lookup_table) {
          d_2d_lut_ptr = new TwoDimensionalLUT(
            lookup_table->axis1()->size(), lookup_table->axis1()->values()->data(), 
            lookup_table->axis2()->size(), lookup_table->axis2()->values()->data(), 
            lookup_table->values()
          );
          CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(TwoDimensionalLUT));
        } else if (const std::shared_ptr<const Table0> scalar_table = std::dynamic_pointer_cast<const Table0>(pwr->lookupTable(rf)); scalar_table) {
          d_2d_lut_ptr = new TwoDimensionalLUT(
            0, nullptr, 
            0, nullptr, 
            static_cast<sta::FloatTable*>(nullptr),
            true, scalar_table->value(0,0,0)
          );
          CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(TwoDimensionalLUT));
        } else if (const std::shared_ptr<const Table1> lookup_table = std::dynamic_pointer_cast<const Table1>(pwr->lookupTable(rf)); lookup_table) {
          if (lookup_table->axis1()->variable() != TableAxisVariable::total_output_net_capacitance) {
            LOG_ERROR << "Error type of lookup_table->axis1(): " << static_cast<typename std::underlying_type<TableAxisVariable>::type>(lookup_table->axis1()->variable()) << " when converting 1d lut to 2d lut.";
          }
          d_2d_lut_ptr = new TwoDimensionalLUT(
            lookup_table->axis1()->size(), lookup_table->axis1()->values()->data(), 
            lookup_table->values()->data()
          );
          CUDA_MEM_STATS.add(CudaMemCategory::lut_managed, sizeof(TwoDimensionalLUT));
        } else {
          LOG_ERROR << "CudaPower::getOutputInternalPower convert table pointer to table2 and table0 pointer failed.";
        }

        if (strcmp(rf->name(), "rise") == 0) {
          d_2d_lut_pair_ptr->setTable(d_2d_lut_ptr, RISE);
        } else if (strcmp(rf->name(), "fall") == 0) {
          d_2d_lut_pair_ptr->setTable(d_2d_lut_ptr, FALL);
        } else {
          LOG_ERROR << "CudaPower::getInputInternalPower unexpected rf name: " << rf->name();
        }
      }
      d_2d_lut_pair_ptr->setWhenState(when_state_vec, when_satisfiable);

      if (n_state != INVALID_N_STATE) {
        utils::ScopedTimer timer_bsim_construction("BSIM/state-index mapping construction");
        assert(h_cur_related_port_internal_power_LUTs.size() != 0);
        std::vector<NStateVal> matched_state_idxs;
        getStateIdx(port_name_to_idx_map, pwr->when(), matched_state_idxs);
        for (const NStateVal matched_state_idx: matched_state_idxs) {
          h_cur_related_port_internal_power_LUTs[matched_state_idx] = d_2d_lut_pair_ptr;
        }
      }
      h_cur_related_port_internal_power_LUTs_indexed_by_order.push_back(d_2d_lut_pair_ptr);
    }
  }
  delete pin_iter;
  // ----------------end of generate LUTs for each related pin----------------

  // ----------------copy to device side----------------
  std::vector<TwoDimensionalLUTPair**> d_luts_of_related_ports(n_pin, nullptr);
  for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    const auto& h_cur_related_port_internal_power_LUTs = h_cur_port_internal_power_LUTs.at(pin_idx);
    if (!h_cur_related_port_internal_power_LUTs.empty()) {
      CHECK_CUDA_RUNTIME(cudaMalloc(&d_luts_of_related_ports[pin_idx], sizeof(TwoDimensionalLUTPair*) * h_cur_related_port_internal_power_LUTs.size()));
      CHECK_CUDA_RUNTIME(cudaMemcpy(d_luts_of_related_ports[pin_idx], h_cur_related_port_internal_power_LUTs.data(), sizeof(TwoDimensionalLUTPair*) * h_cur_related_port_internal_power_LUTs.size(), cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(CudaMemCategory::bsim_lut_index, sizeof(TwoDimensionalLUTPair*) * h_cur_related_port_internal_power_LUTs.size());
    } else {  // no internal power for the current related port
      d_luts_of_related_ports[pin_idx] = nullptr;
    }
  }
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_cur_port_internal_power_LUTs, sizeof(TwoDimensionalLUTPair**) * n_pin));
  CHECK_CUDA_RUNTIME(cudaMemcpy(d_cur_port_internal_power_LUTs, d_luts_of_related_ports.data(), sizeof(TwoDimensionalLUTPair**) * n_pin, cudaMemcpyHostToDevice));
  CUDA_MEM_STATS.add(CudaMemCategory::bsim_lut_index, sizeof(TwoDimensionalLUTPair**) * n_pin);

  std::vector<NStateVal> n_luts_of_related_ports_indexed_by_order(n_pin, 0);
  std::vector<TwoDimensionalLUTPair**> d_luts_of_related_ports_indexed_by_order(n_pin, nullptr);
  for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    const auto& h_cur_related_port_internal_power_LUTs_indexed_by_order = h_cur_port_internal_power_LUTs_indexed_by_order.at(pin_idx);
    if (!h_cur_related_port_internal_power_LUTs_indexed_by_order.empty()) {
      n_luts_of_related_ports_indexed_by_order[pin_idx] = h_cur_related_port_internal_power_LUTs_indexed_by_order.size();
      CHECK_CUDA_RUNTIME(cudaMalloc(&d_luts_of_related_ports_indexed_by_order[pin_idx], sizeof(TwoDimensionalLUTPair*) * h_cur_related_port_internal_power_LUTs_indexed_by_order.size()));
      CHECK_CUDA_RUNTIME(cudaMemcpy(d_luts_of_related_ports_indexed_by_order[pin_idx], h_cur_related_port_internal_power_LUTs_indexed_by_order.data(), sizeof(TwoDimensionalLUTPair*) * h_cur_related_port_internal_power_LUTs_indexed_by_order.size(), cudaMemcpyHostToDevice));
      CUDA_MEM_STATS.add(CudaMemCategory::fallback_lut_index, sizeof(TwoDimensionalLUTPair*) * h_cur_related_port_internal_power_LUTs_indexed_by_order.size());
    } else {
      n_luts_of_related_ports_indexed_by_order[pin_idx] = 0;
      d_luts_of_related_ports_indexed_by_order[pin_idx] = nullptr;
    }
  }
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_cur_port_n_internal_power_LUTs_indexed_by_order, sizeof(NStateVal) * n_pin));
  CHECK_CUDA_RUNTIME(cudaMemcpy(d_cur_port_n_internal_power_LUTs_indexed_by_order, n_luts_of_related_ports_indexed_by_order.data(), sizeof(NStateVal) * n_pin, cudaMemcpyHostToDevice));
  CUDA_MEM_STATS.add(CudaMemCategory::fallback_lut_index, sizeof(NStateVal) * n_pin);
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_cur_port_internal_power_LUTs_indexed_by_order, sizeof(TwoDimensionalLUTPair**) * n_pin));
  CHECK_CUDA_RUNTIME(cudaMemcpy(d_cur_port_internal_power_LUTs_indexed_by_order, d_luts_of_related_ports_indexed_by_order.data(), sizeof(TwoDimensionalLUTPair**) * n_pin, cudaMemcpyHostToDevice));
  CUDA_MEM_STATS.add(CudaMemCategory::fallback_lut_index, sizeof(TwoDimensionalLUTPair**) * n_pin);
  // ----------------end of copy to device side----------------
}

void
CudaPower::setEvent(VcdEventTime time, VcdEventVal val, bool is_glitch, NEeventVal pos) 
{
  assert(0 <= pos && pos < G_CONFIG.nums.max_event_num);
  events_[pos].time = time;
  events_[pos].val = val;
  // events_[pos].is_glitch = is_glitch;
}

void 
CudaPower::initPerGatePower()
{
  utils::ScopedTimer timer_init_per_gate_power("initPerGatePower");
  LOG_INFO << "Initing per gate power...";
  // -------------------------------for multiple output gates------------------------------
  CHECK_CUDA_RUNTIME(cudaMalloc(&gate_leakage_powers_, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&gate_internal_powers_, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&gate_glitch_internal_powers_, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&gate_switching_powers_, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&gate_glitch_switching_powers_, n_multiple_output_gate_ * sizeof(PowerVal)));
  h_gate_leakage_powers_ = new PowerVal[n_multiple_output_gate_];
  h_gate_internal_powers_ = new PowerVal[n_multiple_output_gate_];
  h_gate_glitch_internal_powers_ = new PowerVal[n_multiple_output_gate_];
  h_gate_switching_powers_ = new PowerVal[n_multiple_output_gate_];
  h_gate_glitch_switching_powers_ = new PowerVal[n_multiple_output_gate_];
  CUDA_MEM_STATS.add(CudaMemCategory::power_result, n_multiple_output_gate_ * sizeof(PowerVal) * 5);
  // -------------------------------end of for multiple output gates------------------------------
  LOG_INFO << "Initing per gate power";
}

void
CudaPower::setPeriodAndInitPerCyclePower(VcdEventTime _max_event_time, EventTimeVal _vcd_time_scale) 
{
  max_event_time_ = _max_event_time;
  vcd_time_scale_ = _vcd_time_scale;
  vcd_time_unit_per_cycle_ = static_cast<VcdEventTime>(ceil(clk_period_ / vcd_time_scale_));
  n_period_ = ceil_div(max_event_time_, vcd_time_unit_per_cycle_);
  LOG_INFO << "max_event_time_: " << max_event_time_ << " n_period_: " << n_period_ << " clk_period_: " << clk_period_ << " vcd_time_scale_: " << vcd_time_scale_ << " vcd_time_unit_per_cycle_: " << vcd_time_unit_per_cycle_;

  //--------------------for leakage and internal-----------------------
  CHECK_CUDA_RUNTIME(cudaMalloc(&per_cycle_leakage_powers_, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&per_cycle_internal_powers_, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&per_cycle_glitch_internal_powers_, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(per_cycle_leakage_powers_, 0, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(per_cycle_internal_powers_, 0, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(per_cycle_glitch_internal_powers_, 0, n_period_ * sizeof(PowerVal)));
  h_per_cycle_leakage_powers_ = new PowerVal[n_period_];
  h_per_cycle_internal_powers_ = new PowerVal[n_period_];
  h_per_cycle_glitch_internal_powers_ = new PowerVal[n_period_];
  // CHECK_CUDA_RUNTIME(cudaMemAdvise(per_cycle_leakage_powers_, n_period_ * sizeof(PowerVal), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
  // CHECK_CUDA_RUNTIME(cudaMemAdvise(per_cycle_internal_powers_, n_period_ * sizeof(PowerVal), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
  // CHECK_CUDA_RUNTIME(cudaMemAdvise(per_cycle_glitch_internal_powers_, n_period_ * sizeof(PowerVal), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
  //--------------------end of for leakage and internal-----------------------

  //--------------------for switching-----------------------
  CHECK_CUDA_RUNTIME(cudaMalloc(&per_cycle_switching_powers_, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMalloc(&per_cycle_glitch_switching_powers_, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(per_cycle_switching_powers_, 0, n_period_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(per_cycle_glitch_switching_powers_, 0, n_period_ * sizeof(PowerVal)));
  h_per_cycle_switching_powers_ = new PowerVal[n_period_];
  h_per_cycle_glitch_switching_powers_ = new PowerVal[n_period_];
  CUDA_MEM_STATS.add(CudaMemCategory::power_result, n_period_ * sizeof(PowerVal) * 5);
  // CHECK_CUDA_RUNTIME(cudaMemAdvise(per_cycle_switching_powers_, n_period_ * sizeof(PowerVal), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
  // CHECK_CUDA_RUNTIME(cudaMemAdvise(per_cycle_glitch_switching_powers_, n_period_ * sizeof(PowerVal), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
  //--------------------end of for switching-----------------------

  // int check_device_concurrent_attr_res = -1;
  // cudaDeviceGetAttribute(&check_device_concurrent_attr_res, cudaDevAttrConcurrentManagedAccess, G_CONFIG.nums.cuda_device_id);  // only works for pascal
  // if (check_device_concurrent_attr_res) {
  //   CHECK_CUDA_RUNTIME(cudaMemPrefetchAsync(per_cycle_leakage_powers_, n_period_ * sizeof(PowerVal), G_CONFIG.nums.cuda_device_id, NULL));
  //   CHECK_CUDA_RUNTIME(cudaMemPrefetchAsync(per_cycle_internal_powers_, n_period_ * sizeof(PowerVal), G_CONFIG.nums.cuda_device_id, NULL));
  //   CHECK_CUDA_RUNTIME(cudaMemPrefetchAsync(per_cycle_glitch_internal_powers_, n_period_ * sizeof(PowerVal), G_CONFIG.nums.cuda_device_id, NULL));
  //   CHECK_CUDA_RUNTIME(cudaMemPrefetchAsync(per_cycle_switching_powers_, n_period_ * sizeof(PowerVal), G_CONFIG.nums.cuda_device_id, NULL));
  //   CHECK_CUDA_RUNTIME(cudaMemPrefetchAsync(per_cycle_glitch_switching_powers_, n_period_ * sizeof(PowerVal), G_CONFIG.nums.cuda_device_id, NULL));
  // }
}

void 
CudaPower::scheduleKernel(VcdEventTime interval_start_time, VcdEventTime interval_end_time) 
{
  LOG_INFO << "Scheduling kernel... ";
  utils::ScopedTimer timer_schedule_kernel("scheduleKernel");
  scheduleKernelForEventAndCycleBasedPartition(interval_start_time, interval_end_time);
  LOG_INFO << "Schedule kernel complete.";

  if (G_CONFIG.flags.report_cuda_power_thread_alloc_stat) {
    getThreadAllocationStats(interval_start_time, interval_end_time);
  }
}

// generate tiles with start time and end time
// the number of events in range [start time, end time] roughly equals to n_event_per_thread_for_all_pins
void 
CudaPower::scheduleKernelForEventAndCycleBasedPartition(VcdEventTime interval_start_time, VcdEventTime interval_end_time)
{
  if (G_CONFIG.nums.power_res_length_for_each_gate != G_CONFIG.nums.n_thread_per_block_for_all_pins) {
    LOG_ERROR << "power_res_length_for_each_gate " << G_CONFIG.nums.power_res_length_for_each_gate << " != n_thread_per_block_for_all_pins " <<  G_CONFIG.nums.n_thread_per_block_for_all_pins;
  }

  NPeriodVal n_cycle_in_current_interval = (interval_end_time / vcd_time_unit_per_cycle_) - (interval_start_time / vcd_time_unit_per_cycle_);
  // ---------------------------set n_cycle_per_thread for each gate---------------------------
  utils::ScopedTimer timer_select_n_cycle_per_thread("Select n_cycle_per_thread");
  #pragma omp parallel for num_threads(G_CONFIG.nums.multi_thread_number) schedule(static)
  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
    if (G_CONFIG.flags.enable_auto_select_n_cycle_per_thread_for_each_gate) {
      cur_gate->setNCyclePerThreadByEventCount(
        n_cycle_in_current_interval, 
        G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor, 
        G_CONFIG.nums.n_cycle_auto_selection_e_target
      );
    } else {
      cur_gate->setNCyclePerThread(G_CONFIG.nums.n_cycle_per_thread);
    }
  }
  timer_select_n_cycle_per_thread.EndTiming();
  // ---------------------------end of set n_cycle_per_thread for each gate---------------------------

  // ---------------------------get stat information---------------------------
  // const double n_event_per_block = G_CONFIG.nums.n_event_per_thread_for_all_pins * G_CONFIG.nums.n_thread_per_block_for_all_pins;  // double type to make sure ceil makes sense
  utils::ScopedTimer timer_compute_thread_block_ranges("Compute Thread/Block Ranges");
  const int n_event_per_thread = G_CONFIG.flags.cuda_power_separate_kernels_for_dyn_and_leak ? 1 : G_CONFIG.nums.n_event_per_thread_for_all_pins;
  std::vector<NThreadVal> event_thread_counts(n_multiple_output_gate_);
  std::vector<NThreadVal> cycle_thread_counts(n_multiple_output_gate_);
  std::vector<NBlockVal> event_block_counts(n_multiple_output_gate_);
  std::vector<NBlockVal> cycle_block_counts(n_multiple_output_gate_);
  #pragma omp parallel for num_threads(G_CONFIG.nums.multi_thread_number) schedule(static)
  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
    event_thread_counts[gate_idx] = ceil_div(std::accumulate(cur_gate->pin_waveform_sizes, cur_gate->pin_waveform_sizes + cur_gate->n_pin, static_cast<size_t>(0)), n_event_per_thread);
    cycle_thread_counts[gate_idx] = ceil_div(n_cycle_in_current_interval, cur_gate->n_cycle_per_thread);  // n_cycle_per_thread is already set for each gate
    event_block_counts[gate_idx] = ceil_div(event_thread_counts[gate_idx], G_CONFIG.nums.n_thread_per_block_for_all_pins);
    cycle_block_counts[gate_idx] = ceil_div(cycle_thread_counts[gate_idx], G_CONFIG.nums.n_thread_per_block_for_all_pins);
  }

  std::vector<NThreadVal> event_thread_offsets(n_multiple_output_gate_ + 1, 0);
  std::vector<NThreadVal> cycle_thread_offsets(n_multiple_output_gate_ + 1, 0);
  std::vector<NBlockVal> event_block_offsets(n_multiple_output_gate_ + 1, 0);
  std::vector<NBlockVal> cycle_block_offsets(n_multiple_output_gate_ + 1, 0);
  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    event_thread_offsets[gate_idx + 1] = event_thread_offsets[gate_idx] + event_thread_counts[gate_idx];
    cycle_thread_offsets[gate_idx + 1] = cycle_thread_offsets[gate_idx] + cycle_thread_counts[gate_idx];
    event_block_offsets[gate_idx + 1] = event_block_offsets[gate_idx] + event_block_counts[gate_idx];
    cycle_block_offsets[gate_idx + 1] = cycle_block_offsets[gate_idx] + cycle_block_counts[gate_idx];
  }

  NThreadVal accu_thread_count_for_event_partition = event_thread_offsets.back();
  NBlockVal accu_block_count_for_event_partition = event_block_offsets.back();
  NThreadVal accu_thread_count_for_cycle_partition = cycle_thread_offsets.back();
  NBlockVal accu_block_count_for_cycle_partition = cycle_block_offsets.back();
  std::unique_ptr<NGateVal[]> block_corresonpding_gate_idxes_for_event_partition(new NGateVal[accu_block_count_for_event_partition]);
  std::unique_ptr<NGateVal[]> block_corresonpding_gate_idxes_for_cycle_partition(new NGateVal[accu_block_count_for_cycle_partition]);

  #pragma omp parallel for num_threads(G_CONFIG.nums.multi_thread_number) schedule(guided, 64)
  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
    cur_gate->setAllPinsBlockThreadRange(
      event_thread_offsets[gate_idx], event_thread_offsets[gate_idx + 1],
      event_block_offsets[gate_idx], event_block_offsets[gate_idx + 1],
      cycle_thread_offsets[gate_idx], cycle_thread_offsets[gate_idx + 1],
      cycle_block_offsets[gate_idx], cycle_block_offsets[gate_idx + 1]
    );
    std::fill(
      block_corresonpding_gate_idxes_for_event_partition.get() + event_block_offsets[gate_idx],
      block_corresonpding_gate_idxes_for_event_partition.get() + event_block_offsets[gate_idx + 1],
      gate_idx
    );
    std::fill(
      block_corresonpding_gate_idxes_for_cycle_partition.get() + cycle_block_offsets[gate_idx],
      block_corresonpding_gate_idxes_for_cycle_partition.get() + cycle_block_offsets[gate_idx + 1],
      gate_idx
    );
  }
  const size_t power_res_length_for_all_gates = n_multiple_output_gate_ * G_CONFIG.nums.power_res_length_for_each_gate;
  timer_compute_thread_block_ranges.EndTiming();

  LOG_INFO << "accu_thread_count_for_event_partition: " << accu_thread_count_for_event_partition << " accu_thread_count_for_cycle_partition: " << accu_thread_count_for_cycle_partition
    << " power_res_length_for_all_gates: " << power_res_length_for_all_gates << " estimated global auxiliary memory usage: " 
    << sizeof(PowerVal) * power_res_length_for_all_gates * 5 / (1024.0 * 1024 * 1024) << "GB";
  n_block_for_event_partition_ = accu_block_count_for_event_partition;
  utils::ScopedTimer timer_h2d_transfer("H2D Transfer");
  CHECK_CUDA_RUNTIME(cudaFree(block_corr_gate_idxes_for_event_partition_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&block_corr_gate_idxes_for_event_partition_, sizeof(NGateVal) * n_block_for_event_partition_));
  CHECK_CUDA_RUNTIME(cudaMemcpy(block_corr_gate_idxes_for_event_partition_, block_corresonpding_gate_idxes_for_event_partition.get(), sizeof(NGateVal) * n_block_for_event_partition_, cudaMemcpyHostToDevice));
  n_block_for_cycle_partition_ = accu_block_count_for_cycle_partition;
  CHECK_CUDA_RUNTIME(cudaFree(block_corr_gate_idxes_for_cycle_partition_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&block_corr_gate_idxes_for_cycle_partition_, sizeof(NGateVal) * n_block_for_cycle_partition_));
  CHECK_CUDA_RUNTIME(cudaMemcpy(block_corr_gate_idxes_for_cycle_partition_, block_corresonpding_gate_idxes_for_cycle_partition.get(), sizeof(NGateVal) * n_block_for_cycle_partition_, cudaMemcpyHostToDevice));
  timer_h2d_transfer.EndTiming();
  CUDA_MEM_STATS.set(CudaMemCategory::schedule, sizeof(NGateVal) * (n_block_for_event_partition_ + n_block_for_cycle_partition_));

  utils::ScopedTimer timer_allocate_reset_tile_results("Allocate/Reset Tile Results");
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_leakage_res_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_per_tile_leakage_res_, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaMemset(global_per_tile_leakage_res_, 0, sizeof(PowerVal) *  power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_internal_res_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_per_tile_internal_res_, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaMemset(global_per_tile_internal_res_, 0, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_glitch_internal_res_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_per_tile_glitch_internal_res_, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaMemset(global_per_tile_glitch_internal_res_, 0, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_switching_res_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_per_tile_switching_res_, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaMemset(global_per_tile_switching_res_, 0, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_glitch_switching_res_));
  CHECK_CUDA_RUNTIME(cudaMalloc(&global_per_tile_glitch_switching_res_, sizeof(PowerVal) * power_res_length_for_all_gates));
  CHECK_CUDA_RUNTIME(cudaMemset(global_per_tile_glitch_switching_res_, 0, sizeof(PowerVal) * power_res_length_for_all_gates));
  timer_allocate_reset_tile_results.EndTiming();
  CUDA_MEM_STATS.set(CudaMemCategory::tile_result, sizeof(PowerVal) * power_res_length_for_all_gates * 5);

  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
    const size_t cur_gate_power_res_offset = gate_idx * G_CONFIG.nums.power_res_length_for_each_gate;
    cur_gate->setPerTileResultPointer(
      global_per_tile_leakage_res_,
      global_per_tile_internal_res_,
      global_per_tile_glitch_internal_res_,
      global_per_tile_switching_res_,
      global_per_tile_glitch_switching_res_,
      cur_gate_power_res_offset
    );
  }
  // ---------------------------end of get stat information---------------------------
}

void
CudaPower::runCudaPowerAnalysis(VcdEventTime interval_start_time, VcdEventTime interval_end_time, const std::pair<VcdEventTime, VcdEventTime>* next_interval)
{
  utils::ScopedTimer timer_run_cuda_power_analysis("runCudaPowerAnalysis");
  char cuda_thread_partition_basis;
  NBlockVal n_block;
  NGateVal* block_corr_gate_idxes;
  if (G_CONFIG.strs.cuda_thread_partition_basis == "event") {
    cuda_thread_partition_basis = 'e';
    n_block = n_block_for_event_partition_;
    block_corr_gate_idxes = block_corr_gate_idxes_for_event_partition_;
  } else if (G_CONFIG.strs.cuda_thread_partition_basis == "cycle") {
    cuda_thread_partition_basis = 'c';
    n_block = n_block_for_cycle_partition_;
    block_corr_gate_idxes = block_corr_gate_idxes_for_cycle_partition_;
  } else {
    LOG_ERROR << "Unknown cuda_thread_partition_basis: " << G_CONFIG.strs.cuda_thread_partition_basis;
  }
  reportPowerKernelResourceUsage(n_block);

  cudaStream_t stream = 0;
  cudaEvent_t all_start_cu_event, all_stop_cu_event, k11_stop_cu_event;
  CHECK_CUDA_RUNTIME(cudaEventCreate(&all_start_cu_event));
  CHECK_CUDA_RUNTIME(cudaEventCreate(&all_stop_cu_event));
  CHECK_CUDA_RUNTIME(cudaEventCreate(&k11_stop_cu_event));
  // --------------------------run power computation kernel--------------------------
  LOG_INFO << "Launching power calculation kernel...";
  utils::ScopedTimer timer_power_analysis_kernel("Power Analysis Kernel");
  if (G_CONFIG.flags.partition_unit_is_cycle) {
    LOG_INFO << "Runnig power calculation, the time range is partitioned by cycle unit, n_period_: " << n_period_
      << " start cycle: " << (interval_start_time / vcd_time_unit_per_cycle_) << " end cycle: "  <<  (interval_end_time / vcd_time_unit_per_cycle_);
    CHECK_CUDA_RUNTIME(cudaEventRecord(all_start_cu_event, stream));
    if (G_CONFIG.flags.cuda_power_separate_kernels_for_dyn_and_leak) {
      LOG_INFO << "Running separate power computation kernels (1T1E dynamic + 1T1C leakage)";
      kernel1_1NaiveDynamicPowerCalculationTimeRangePartitionedByCycleUnit<<<n_block_for_event_partition_, G_CONFIG.nums.n_thread_per_block_for_all_pins, 0, stream>>> (
        'e', 1, 0, G_CONFIG.nums.n_thread_per_block_for_all_pins,
        (interval_start_time / vcd_time_unit_per_cycle_), (interval_end_time / vcd_time_unit_per_cycle_), vcd_time_unit_per_cycle_, max_event_time_, 
        clk_period_, vcd_time_scale_,
        d_multiple_output_gates_, d_events_, block_corr_gate_idxes_for_event_partition_,
        // Return Values
        per_cycle_leakage_powers_, per_cycle_internal_powers_, per_cycle_glitch_internal_powers_,
        per_cycle_switching_powers_, per_cycle_glitch_switching_powers_,
        G_CONFIG.nums.power_res_length_for_each_gate
      );
      CHECK_CUDA_RUNTIME(cudaEventRecord(k11_stop_cu_event, stream));
      kernel1_2NaiveLeakagePowerCalculationTimeRangePartitionedByCycleUnit<<<n_block_for_cycle_partition_, G_CONFIG.nums.n_thread_per_block_for_all_pins, 0, stream>>>(
        'c', 0, 0, G_CONFIG.nums.n_thread_per_block_for_all_pins,
        (interval_start_time / vcd_time_unit_per_cycle_), (interval_end_time / vcd_time_unit_per_cycle_), vcd_time_unit_per_cycle_, max_event_time_, 
        clk_period_, vcd_time_scale_,
        d_multiple_output_gates_, d_events_, block_corr_gate_idxes_for_cycle_partition_,
        // Return Values
        per_cycle_leakage_powers_, per_cycle_internal_powers_, per_cycle_glitch_internal_powers_,
        per_cycle_switching_powers_, per_cycle_glitch_switching_powers_,
        G_CONFIG.nums.power_res_length_for_each_gate
      );
    } else {
      LOG_INFO << "Running fused power computation kernel";
      kernel1AllPowerCalculationTimeRangePartitionedByCycleUnit<<<n_block, G_CONFIG.nums.n_thread_per_block_for_all_pins, 0, stream>>> (
        cuda_thread_partition_basis, G_CONFIG.nums.n_event_per_thread_for_all_pins, 0, G_CONFIG.nums.n_thread_per_block_for_all_pins,
        (interval_start_time / vcd_time_unit_per_cycle_), (interval_end_time / vcd_time_unit_per_cycle_), vcd_time_unit_per_cycle_, max_event_time_, 
        clk_period_, vcd_time_scale_,
        d_multiple_output_gates_, d_events_, block_corr_gate_idxes,
        per_cycle_leakage_powers_, per_cycle_internal_powers_, per_cycle_glitch_internal_powers_,
        per_cycle_switching_powers_, per_cycle_glitch_switching_powers_,
        G_CONFIG.nums.power_res_length_for_each_gate
      );
    }
    CHECK_CUDA_RUNTIME(cudaEventRecord(all_stop_cu_event, stream));
  } else {
    LOG_ERROR << "Runnig power calculation partitioned by time";
  }
  CHECK_CUDA_RUNTIME(cudaGetLastError());

  if (G_CONFIG.flags.overlapping_get_gate_waveform_range && next_interval) {
    std::vector<NEeventVal> h_global_pin_waveform_starts;
    std::vector<NEeventVal> h_global_pin_waveform_ends;
    std::thread cpu_thread([&next_interval, &h_global_pin_waveform_starts, &h_global_pin_waveform_ends, this](){
      // getGateWaveformRangeSingleThread(next_interval->first, next_interval->second, h_global_pin_waveform_starts, h_global_pin_waveform_ends);
      getGateWaveformRangeMultiThread(next_interval->first, next_interval->second, h_global_pin_waveform_starts, h_global_pin_waveform_ends);
    });
    // CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());
    CHECK_CUDA_RUNTIME(cudaEventSynchronize(all_stop_cu_event));
    timer_power_analysis_kernel.EndTiming();
    cpu_thread.join();
    renewGateWaveform(h_global_pin_waveform_starts, h_global_pin_waveform_ends);
  } else {
    // CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());
    CHECK_CUDA_RUNTIME(cudaEventSynchronize(all_stop_cu_event));
    timer_power_analysis_kernel.EndTiming();
  }
  LOG_INFO << "Power computation kernel complete";
  // --------------------------end of run power computation kernel--------------------------

  float all_elapsed_time;
  CHECK_CUDA_RUNTIME(cudaEventElapsedTime(&all_elapsed_time, all_start_cu_event, all_stop_cu_event));
  LOG_INFO << "runCudaPowerAnalysis kernel runtime: " << all_elapsed_time << "ms";
  power_computation_kernel_total_time_ += all_elapsed_time;
  if (G_CONFIG.flags.cuda_power_separate_kernels_for_dyn_and_leak) {
    float k11_elapsed_time;
    CHECK_CUDA_RUNTIME(cudaEventElapsedTime(&k11_elapsed_time, all_start_cu_event, k11_stop_cu_event));
    LOG_INFO << "Kernel1_1 (dynamic power) runtime: " << k11_elapsed_time << "ms";
    k11_kernel_total_time_ += k11_elapsed_time;
  }
  CHECK_CUDA_RUNTIME(cudaEventDestroy(all_start_cu_event));
  CHECK_CUDA_RUNTIME(cudaEventDestroy(all_stop_cu_event));
  CHECK_CUDA_RUNTIME(cudaEventDestroy(k11_stop_cu_event));
}

void
CudaPower::mergeResult()
{
  utils::ScopedTimer timer_merge_result("mergeResult");

  // --------------------------leakage and internal--------------------------
  LOG_INFO << "Launching merge leakage and internal tile result kernel...";
  CHECK_CUDA_RUNTIME(cudaMemset(gate_leakage_powers_, 0, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(gate_internal_powers_, 0, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(gate_glitch_internal_powers_, 0, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(gate_switching_powers_, 0, n_multiple_output_gate_ * sizeof(PowerVal)));
  CHECK_CUDA_RUNTIME(cudaMemset(gate_glitch_switching_powers_, 0, n_multiple_output_gate_ * sizeof(PowerVal)));
  cudaEvent_t start, stop;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  kernel2MergeAllPowerResult<<<(n_multiple_output_gate_ - 1) / G_CONFIG.nums.n_thread_per_block_for_all_pins + 1, G_CONFIG.nums.n_thread_per_block_for_all_pins>>>(
    n_multiple_output_gate_, d_multiple_output_gates_, 
    gate_leakage_powers_, gate_internal_powers_, gate_glitch_internal_powers_, 
    gate_switching_powers_, gate_glitch_switching_powers_,
    G_CONFIG.nums.power_res_length_for_each_gate
  );
  cudaEventRecord(stop, 0);
  CHECK_CUDA_RUNTIME(cudaGetLastError());
  // CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());
  CHECK_CUDA_RUNTIME(cudaEventSynchronize(stop));

  cudaEventElapsedTime(&elapsed_time, start, stop);
  LOG_INFO << "mergeResult kernel runtime: " << elapsed_time << "ms";
  merge_result_kernel_total_time_ += elapsed_time;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  LOG_INFO << "Merge leakage and internal tile result kernel complete";
  // --------------------------end of leakage and internal--------------------------
}

void
CudaPower::recordResult()
{
  utils::ScopedTimer timer_record_result("recordResult");
  LOG_INFO << "Recording result...";
  // --------------------------leakage and internal--------------------------
  utils::ScopedTimer timer_d2h_transfer("D2H Transfer");
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_gate_leakage_powers_, gate_leakage_powers_, n_multiple_output_gate_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_gate_internal_powers_, gate_internal_powers_, n_multiple_output_gate_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_gate_glitch_internal_powers_, gate_glitch_internal_powers_, n_multiple_output_gate_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_gate_switching_powers_, gate_switching_powers_, n_multiple_output_gate_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_gate_glitch_switching_powers_, gate_glitch_switching_powers_, n_multiple_output_gate_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  timer_d2h_transfer.EndTiming();
  utils::ScopedTimer timer_aggregation("Reduction/Aggregation");
  #pragma omp parallel for num_threads(G_CONFIG.nums.multi_thread_number) schedule(static)
  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    const sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
    PowerResult& inst_power_res = inst_to_res_.at(cur_gate->inst);
    inst_power_res.leakage() += h_gate_leakage_powers_[gate_idx];
    inst_power_res.internal() += h_gate_internal_powers_[gate_idx] + h_gate_glitch_internal_powers_[gate_idx];
    inst_power_res.glitchInternal() += h_gate_glitch_internal_powers_[gate_idx];
    inst_power_res.switching() += h_gate_switching_powers_[gate_idx] + h_gate_glitch_switching_powers_[gate_idx];
    inst_power_res.glitchSwitching() += h_gate_glitch_switching_powers_[gate_idx];
  }
  if (LOG_DEBUG_FLAG) {
    for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
      const sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
      LOG_DEBUG << "In recordResult, " 
        << gate_idx << " " << network_->pathName(cur_gate->inst) 
        << " leakage power: " << h_gate_leakage_powers_[gate_idx]
        << " internal power: " << h_gate_internal_powers_[gate_idx]
        << " glitch internal power: " << h_gate_glitch_internal_powers_[gate_idx]
        << " switching power: " << h_gate_switching_powers_[gate_idx]
        << " glitch switching power: " << h_gate_glitch_switching_powers_[gate_idx];
    }
  }
  timer_aggregation.EndTiming();
  // --------------------------end of leakage and internal--------------------------
  LOG_INFO << "Record result complete";
}

void
CudaPower::finalizeResult(
  PowerResult &total,
  PowerResult &sequential,
  PowerResult &combinational,
  PowerResult &clock,
  PowerResult &macro,
  PowerResult &pad
)
{
  utils::ScopedTimer finalize_result_timer("finalizeResult");
  LOG_INFO << "Finalizing result...";
  utils::ScopedTimer timer_aggregation("Reduction/Aggregation");
  for (auto iter = inst_to_res_.begin(); iter != inst_to_res_.end(); ++iter){
    LibertyCell *cell = network_->libertyCell(iter->first);
    if (cell) {
      const PowerResult& inst_power_res = iter->second;
      if (cell->isMacro() || cell->isMemory() || cell->interfaceTiming()) {
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << "Gate " << network_->pathName(iter->first) 
            << " is macro, internal power: " << inst_power_res.internal() << " glitch internal power: " << inst_power_res.glitchInternal()
            << " switching power: " << inst_power_res.switching() << " glitch switching power: " << inst_power_res.glitchSwitching()
            << " total glitch power: " << inst_power_res.glitchInternal() + inst_power_res.glitchSwitching()
            << " leakage power: " << inst_power_res.leakage();
        }
        macro.incr(inst_power_res);
      }
      else if (cell->isPad()) {
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << "Gate " << network_->pathName(iter->first) 
            << " is pad, internal power: " << inst_power_res.internal() << " glitch internal power: " << inst_power_res.glitchInternal()
            << " switching power: " << inst_power_res.switching() << " glitch switching power: " << inst_power_res.glitchSwitching()
            << " total glitch power: " << inst_power_res.glitchInternal() + inst_power_res.glitchSwitching()
            << " leakage power: " << inst_power_res.leakage();
        }
        pad.incr(inst_power_res);
      }
      else if (inClockNetwork(iter->first)) {
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << "Gate " << network_->pathName(iter->first) 
            << " is clock, internal power: " << inst_power_res.internal() << " glitch internal power: " << inst_power_res.glitchInternal()
            << " switching power: " << inst_power_res.switching() << " glitch switching power: " << inst_power_res.glitchSwitching()
            << " total glitch power: " << inst_power_res.glitchInternal() + inst_power_res.glitchSwitching()
            << " leakage power: " << inst_power_res.leakage();
        }
        clock.incr(inst_power_res);
      }
      else if (cell->hasSequentials()) {
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << "Gate " << network_->pathName(iter->first) 
            << " is sequential, internal power: " << inst_power_res.internal() << " glitch internal power: " << inst_power_res.glitchInternal()
            << " switching power: " << inst_power_res.switching() << " glitch switching power: " << inst_power_res.glitchSwitching()
            << " total glitch power: " << inst_power_res.glitchInternal() + inst_power_res.glitchSwitching()
            << " leakage power: " << inst_power_res.leakage();
        }
        sequential.incr(inst_power_res);
      }
      else {
        if (LOG_DEBUG_FLAG) {
          LOG_DEBUG << "Gate " << network_->pathName(iter->first) 
            << " is combinational, internal power: " << inst_power_res.internal() << " glitch internal power: " << inst_power_res.glitchInternal()
            << " switching power: " << inst_power_res.switching() << " glitch switching power: " << inst_power_res.glitchSwitching()
            << " total glitch power: " << inst_power_res.glitchInternal() + inst_power_res.glitchSwitching()
            << " leakage power: " << inst_power_res.leakage();
        }
        combinational.incr(inst_power_res);
      }
      total.incr(inst_power_res);
    }
  }
  timer_aggregation.EndTiming();

  const Instance* top_instance = network_->topInstance();
  total.initLeakagePowerClkedWaveform(top_instance, clk_period_, n_period_);
  total.initInternalPowerClkedWaveform(top_instance, clk_period_, n_period_);
  total.initGlitchInternalPowerClkedWaveform(top_instance, clk_period_, n_period_);
  total.initSwitchingPowerClkedWaveform(top_instance, clk_period_, n_period_);
  total.initGlitchSwitchingPowerClkedWaveform(top_instance, clk_period_, n_period_);
  utils::ScopedTimer timer_d2h_transfer("D2H Transfer");
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_per_cycle_leakage_powers_, per_cycle_leakage_powers_, n_period_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_per_cycle_internal_powers_, per_cycle_internal_powers_, n_period_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_per_cycle_glitch_internal_powers_, per_cycle_glitch_internal_powers_, n_period_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_per_cycle_switching_powers_, per_cycle_switching_powers_, n_period_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_per_cycle_glitch_switching_powers_, per_cycle_glitch_switching_powers_, n_period_ * sizeof(PowerVal), cudaMemcpyDeviceToHost));
  timer_d2h_transfer.EndTiming();
  utils::ScopedTimer timer_waveform_aggregation("Reduction/Aggregation");
  for (NPeriodVal cycle_idx = 0; cycle_idx < n_period_; ++cycle_idx) {
    total.findLeakagePowerClkedWaveform(top_instance).waveform()[cycle_idx] += h_per_cycle_leakage_powers_[cycle_idx];
    total.findInternalPowerClkedWaveform(top_instance).waveform()[cycle_idx] += h_per_cycle_internal_powers_[cycle_idx] + h_per_cycle_glitch_internal_powers_[cycle_idx];
    total.findGlitchInternalPowerClkedWaveform(top_instance).waveform()[cycle_idx] += h_per_cycle_glitch_internal_powers_[cycle_idx];
    total.findSwitchingPowerClkedWaveform(top_instance).waveform()[cycle_idx] += h_per_cycle_switching_powers_[cycle_idx] + h_per_cycle_glitch_switching_powers_[cycle_idx];
    total.findGlitchSwitchingPowerClkedWaveform(top_instance).waveform()[cycle_idx] += h_per_cycle_glitch_switching_powers_[cycle_idx];
  }
  timer_waveform_aggregation.EndTiming();
  LOG_INFO << "Finalize result complete!";
}

void
CudaPower::printCellRes() const
{
  namespace fs = std::filesystem;
  std::ofstream out_file;
  out_file.open(fs::path(utils::get_power_analysis_cell_res_path()), std::ios::out);
  out_file << "cell" << G_CONFIG.strs.power_analysis_res_file_separator 
    << "Internal_Power" << G_CONFIG.strs.power_analysis_res_file_separator << "Glitch_Internal_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Switching_Power" << G_CONFIG.strs.power_analysis_res_file_separator << "Glitch_Switching_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Leakage_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Total_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Standard_Cell"
    << "\n";

  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    const sta::power::Gate* cur_gate = h_multiple_output_gates_.at(gate_idx);
    LibertyCell* cell = network_->libertyCell(cur_gate->inst);
    PowerVal cur_gate_regular_internal = inst_to_res_.at(cur_gate->inst).internal() - inst_to_res_.at(cur_gate->inst).glitchInternal();
    PowerVal cur_gate_glitch_internal = inst_to_res_.at(cur_gate->inst).glitchInternal();
    PowerVal cur_gate_regular_switching = inst_to_res_.at(cur_gate->inst).switching() - inst_to_res_.at(cur_gate->inst).glitchSwitching();
    PowerVal cur_gate_glitch_switching = inst_to_res_.at(cur_gate->inst).glitchSwitching();
    PowerVal cur_gate_leakage = inst_to_res_.at(cur_gate->inst).leakage();

    out_file << network_->pathName(cur_gate->inst) << G_CONFIG.strs.power_analysis_res_file_separator 
      << cur_gate_regular_internal << G_CONFIG.strs.power_analysis_res_file_separator << cur_gate_glitch_internal << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_switching << G_CONFIG.strs.power_analysis_res_file_separator << cur_gate_glitch_switching << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_leakage << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_internal + cur_gate_glitch_internal + cur_gate_regular_switching + cur_gate_glitch_switching + cur_gate_leakage << G_CONFIG.strs.power_analysis_res_file_separator
      << (cell ? cell->name() : "")
      << "\n";
  }
}

void 
CudaPower::getThreadAllocationStats(const VcdEventTime interval_start_time, const VcdEventTime interval_end_time) const
{
  // NOTE!!! 
  // The configs are hard coded here
  // Currently only compare global n_cycle_per_thread=1, global n_cycle_per_thread=8, and n_cycle auto selection (E_target=G_CONFIG.nums.n_cycle_auto_selection_e_targe)
  // If you want see other configs' stats, please change the variables below.
  const NEeventVal E_target = G_CONFIG.nums.n_cycle_auto_selection_e_target;
  const NPeriodVal global_n_cycle_baseline_1 = 1;
  const NPeriodVal global_n_cycle_baseline_2 = 8;
  const NEeventVal sparse_thr = E_target;  // sparse gates are defined by n_event <= sparse_thr (we set sparse_thr=E_target for fairness across configs)

  LOG_INFO
    << "ThreadAllocRoundConfig"
    << " interval_start_time=" << interval_start_time
    << " interval_end_time=" << interval_end_time
    << " n_cycle_interval=" << (interval_end_time / vcd_time_unit_per_cycle_) - (interval_start_time / vcd_time_unit_per_cycle_)
    << " E_target=" << E_target
    << " sparse_thr=" << sparse_thr
    << " threads_target_sparse=" << G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor
    << " n_thread_per_block=" << G_CONFIG.nums.n_thread_per_block_for_all_pins
    << " baselines=" << global_n_cycle_baseline_1 << "," << global_n_cycle_baseline_2;

  auto s_cpt1 = sta::power::collect_cycle_partition_thread_alloc_stats(
    h_multiple_output_gates_,
    interval_start_time, interval_end_time,
    vcd_time_unit_per_cycle_,
    G_CONFIG.nums.n_thread_per_block_for_all_pins,
    /*enable_auto*/ false,
    /*global_n_cycle_per_thread*/ global_n_cycle_baseline_1,
    /*threads_target_sparse*/ G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor,
    /*E_target*/ E_target,
    /*sparse_override*/ sparse_thr
  );

  auto s_cpt2 = sta::power::collect_cycle_partition_thread_alloc_stats(
    h_multiple_output_gates_,
    interval_start_time, interval_end_time,
    vcd_time_unit_per_cycle_,
    G_CONFIG.nums.n_thread_per_block_for_all_pins,
    /*enable_auto*/ false,
    /*global_n_cycle_per_thread*/ global_n_cycle_baseline_2,
    /*threads_target_sparse*/ G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor,
    /*E_target*/ E_target,
    /*sparse_override*/ sparse_thr
  );

  auto s_auto = sta::power::collect_cycle_partition_thread_alloc_stats(
    h_multiple_output_gates_,
    interval_start_time, interval_end_time,
    vcd_time_unit_per_cycle_,
    G_CONFIG.nums.n_thread_per_block_for_all_pins,
    /*enable_auto*/ true,
    /*global_n_cycle_per_thread*/ global_n_cycle_baseline_1,  // unused when enable_auto=true
    /*threads_target_sparse*/ G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor,
    /*E_target*/ E_target,
    /*sparse_override*/ sparse_thr
  );

  sta::power::logThreadAllocSummaryRound(s_cpt1, "global_cpt" + std::to_string(global_n_cycle_baseline_1), interval_start_time, interval_end_time);
  sta::power::logThreadAllocSummaryRound(s_cpt2, "global_cpt" + std::to_string(global_n_cycle_baseline_2), interval_start_time, interval_end_time);
  sta::power::logThreadAllocSummaryRound(s_auto, "auto_E" + std::to_string(E_target), interval_start_time, interval_end_time);
}

void 
CudaPower::releaseMemory() 
{  // TODO add release LUTs
  LOG_INFO << "Releasing memory...";
  //--------------------events and gates-----------------------
  for (NGateVal gate_idx = 0; gate_idx < n_multiple_output_gate_; ++gate_idx) {
    sta::power::Gate*& cur_gate = h_multiple_output_gates_.at(gate_idx);
    delete cur_gate;
    cur_gate = nullptr;
  }
  CHECK_CUDA_RUNTIME(cudaFree(d_multiple_output_gates_));
  d_multiple_output_gates_ = nullptr;

  delete[] events_;
  events_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(d_events_));
  d_events_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(block_corr_gate_idxes_for_event_partition_));
  block_corr_gate_idxes_for_event_partition_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(block_corr_gate_idxes_for_cycle_partition_));
  block_corr_gate_idxes_for_cycle_partition_ = nullptr;
  //--------------------end of events and gates-----------------------

  //--------------------power analysis result-----------------------
  CHECK_CUDA_RUNTIME(cudaFree(gate_leakage_powers_));
  gate_leakage_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(gate_internal_powers_));
  gate_internal_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(gate_glitch_internal_powers_));
  gate_glitch_internal_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(per_cycle_leakage_powers_));
  per_cycle_leakage_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(per_cycle_internal_powers_));
  per_cycle_internal_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(per_cycle_glitch_internal_powers_));
  per_cycle_glitch_internal_powers_ = nullptr;

  CHECK_CUDA_RUNTIME(cudaFree(gate_switching_powers_));
  gate_switching_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(gate_glitch_switching_powers_));
  gate_glitch_switching_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(per_cycle_switching_powers_));
  per_cycle_switching_powers_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(per_cycle_glitch_switching_powers_));
  per_cycle_glitch_switching_powers_ = nullptr;

  delete h_gate_leakage_powers_;
  h_gate_leakage_powers_ = nullptr;
  delete h_gate_internal_powers_;
  h_gate_internal_powers_ = nullptr;
  delete h_gate_glitch_internal_powers_;
  h_gate_glitch_internal_powers_ = nullptr;
  delete h_gate_switching_powers_;
  h_gate_switching_powers_ = nullptr;
  delete h_gate_glitch_switching_powers_;
  h_gate_glitch_switching_powers_ = nullptr;

  delete h_per_cycle_leakage_powers_;
  h_per_cycle_leakage_powers_ = nullptr;
  delete h_per_cycle_internal_powers_;
  h_per_cycle_internal_powers_ = nullptr; 
  delete h_per_cycle_glitch_internal_powers_;
  h_per_cycle_glitch_internal_powers_ = nullptr;
  delete h_per_cycle_switching_powers_;
  h_per_cycle_switching_powers_ = nullptr;
  delete h_per_cycle_glitch_switching_powers_;
  h_per_cycle_glitch_switching_powers_ = nullptr;
  //--------------------end of power analysis result-----------------------

  // -----auxiliary arrays-----
  // -----auxiliary arrays for gates-----
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_waveform_starts_));
  global_pin_waveform_starts_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_waveform_ends_));
  global_pin_waveform_ends_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_default_states_));
  global_pin_default_states_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_voltages_));
  global_pin_voltages_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_rise_slews_));
  global_pin_rise_slews_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_fall_slews_));
  global_pin_fall_slews_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_load_capacitances_));
  global_pin_load_capacitances_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_pin_is_clocks_));
  global_pin_is_clocks_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_cell_arc_delays_));
  global_cell_arc_delays_ = nullptr;
  // -----end of auxiliary arrays for gates-----
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_leakage_res_));
  global_per_tile_leakage_res_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_internal_res_));
  global_per_tile_internal_res_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_glitch_internal_res_));
  global_per_tile_glitch_internal_res_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_switching_res_));
  global_per_tile_switching_res_ = nullptr;
  CHECK_CUDA_RUNTIME(cudaFree(global_per_tile_glitch_switching_res_));
  global_per_tile_glitch_switching_res_ = nullptr;
  // -----auxiliary arrays to record res-----
  // -----end of auxiliary arrays-----

  LOG_INFO << "Release memory complete!";
}

CudaPower::~CudaPower()
{
  releaseMemory();
}
} // end of namespace sta
