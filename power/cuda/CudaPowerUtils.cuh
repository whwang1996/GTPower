#pragma once
#include "Types.hh"
#include "Event.cuh"
#include "Gate.cuh"

namespace utils::cuda::power {
__device__ __host__ NPeriodVal
clkedWaveformIdx(VcdEventTime time, EventTimeVal vcd_time_scale, PeriodVal period)
{
  VcdEventTime time_unit_per_cycle = static_cast<VcdEventTime>(ceil(period / vcd_time_scale));
  return time / time_unit_per_cycle;
}
template<typename T>
__device__ __host__ NPeriodVal
clkedWaveformIdx(T, T, PeriodVal period) = delete;

// get the minimum event idx of current pin that the time of it is larger than or equals to time
// that is similiar to Leetcode problem 34
__device__ NEeventVal 
getEventIdxByTime(const sta::power::Event *events, NEeventVal start_pos, NEeventVal end_pos, VcdEventTime time, bool contains_equal=false) {
  NEeventVal left = start_pos, right = end_pos;
  while (left < right) { // [left, right)
    NEeventVal mid = left + (right - left) / 2;  // avoid overflow
    if (events[mid].time < time || (contains_equal && events[mid].time <= time)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  return right;
}

__device__ NEeventVal 
getAccuEventCountOfGateByPeriodIdx(const sta::power::Gate* gate, const sta::power::Event *events, NPeriodVal period_idx, const NEeventVal vcd_time_unit_per_cycle)
{
  NEeventVal accu_event_count = 0;
  for (NPinVal pin_idx = 0; pin_idx < gate->n_pin; ++pin_idx) {
    NEeventVal cur_pin_event_idx = getEventIdxByTime(
      events,
      gate->pin_waveform_starts[pin_idx], 
      gate->pin_waveform_ends[pin_idx],
      period_idx * vcd_time_unit_per_cycle
    );
    // the cur_pin_event_idx -1 is the last event index of which the time is less than time variable
    accu_event_count += cur_pin_event_idx - gate->pin_waveform_starts[pin_idx];  // we do not add one here, since the last one is out of range of time
  }

  return accu_event_count;
}

// get the minimum time that the accumulated event count to it is larger than or equals to accu_event_count
__device__ NPeriodVal 
getCycleIndexByAccuEventCount(const sta::power::Gate* gate, const sta::power::Event *events, NEeventVal accu_event_count, NPeriodVal min_period_idx, NPeriodVal max_period_idx, const NEeventVal vcd_time_unit_per_cycle)
{
  NPeriodVal left = min_period_idx, right = max_period_idx;  // max_time is already an open interval here, just use it
  while (left < right) {  // [left, right)
    NPeriodVal mid = left + (right - left) / 2;  // avoid overflow

    if (getAccuEventCountOfGateByPeriodIdx(gate, events, mid, vcd_time_unit_per_cycle) < accu_event_count) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  return right;
}

__device__ NStateVal 
getPinStatesIndex(
  const VcdEventVal* pin_states,
  NPinVal n_pin
) {
  NStateVal res = 0;
  for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    VcdEventVal cur_pin_state = pin_states[pin_idx] == 1 ? 1 : 0;  // 0 or 'X' are both 0;
    res += cur_pin_state * (1 << pin_idx);
  }

  return res;
}

__device__ void 
copyPinStates(
  VcdEventVal* dst_pin_states, 
  const VcdEventVal* ori_pin_states, 
  NPinVal n_pin
) {
  for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    dst_pin_states[pin_idx] = ori_pin_states[pin_idx];
  }
}

__device__ float
getNRise(
  VcdEventVal prev_val,
  VcdEventVal cur_val
) {
  static const float nRiseTable[3][3] = {
    // 0      1      2
    {0.0,    1.0,    0.5},  // prev_val = 0
    {0.0,    0.0,    0.0},  // prev_val = 1
    {0.0,    0.5,    0.0}   // prev_val = 2
  };

  return nRiseTable[prev_val][cur_val];
}

__device__ RISEFALL
getRiseFallEdge(
  VcdEventVal prev_val,
  VcdEventVal cur_val
) {
  return getNRise(prev_val, cur_val) > 0 ? RISE : FALL;
}

__device__ float
getNToggle(
  VcdEventVal prev_val,
  VcdEventVal cur_val
) {
  static const float nToggleTable[3][3] = {
    // 0      1      2
    {0.0,    1.0,    0.5},  // prev_val = 0
    {1.0,    0.0,    0.5},  // prev_val = 1
    {0.5,    0.5,    0.0}   // prev_val = 2
  };

  return nToggleTable[prev_val][cur_val];
}

__device__ int
getRiseFallArcIndex(RISEFALL from_rf, RISEFALL to_rf)
{
  return (2 * (1 - from_rf) + (1 - to_rf));
}

__device__ bool
isValidDelay(DelayVal delay)
{
  return delay >= 0;
}

__device__ bool
isValidSlew(SlewVal slew)
{
  return slew >= 0;
}

__device__ NToggleVal
getTimeBasedGlitchScalingRatio(EventTimeVal pulse_width, SlewVal sum_slew)
{
  if (sum_slew == 0) {
    return 1.0;
  }

  NToggleVal ratio = (pulse_width * 2.0) / sum_slew;
  return my_min(1.0, ratio * ratio);  // clip to 1
}
}
