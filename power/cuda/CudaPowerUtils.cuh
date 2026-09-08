#pragma once
#include "Types.hh"
#include "Event.cuh"
#include "Gate.cuh"

namespace utils::cuda::power {
// __device__ struct PathDependentPinDiff {
//   NPinVal pin_idx;
//   VcdEventTime toggle_time_diff;
//   NEeventVal toggle_event_idx;
// };

// __device__ void swapPathDependentPinDiff(PathDependentPinDiff& p1, PathDependentPinDiff& p2) {
//   NPinVal tmp_idx = p1.pin_idx;
//   p1.pin_idx = p2.pin_idx;
//   p2.pin_idx = tmp_idx;
//   VcdEventTime tmp_diff_time = p1.toggle_time_diff;
//   p1.toggle_time_diff = p2.toggle_time_diff;
//   p2.toggle_time_diff = tmp_diff_time;
//   NEeventVal tmp_event_idx = p1.toggle_event_idx;
//   p1.toggle_event_idx = p2.toggle_event_idx;
//   p2.toggle_event_idx = tmp_event_idx;
// }

// __device__ void sortPathDependentPinDiffs(PathDependentPinDiff* path_dependent_pin_diffs, NPinVal n_pin) {
//   for (int i = 0; i < n_pin; ++i) {
//     for (int j = 0; j < n_pin - 1 - i; ++j) {
//       if (path_dependent_pin_diffs[j].toggle_time_diff < path_dependent_pin_diffs[j + 1].toggle_time_diff) {
//         swapPathDependentPinDiff(path_dependent_pin_diffs[j], path_dependent_pin_diffs[j + 1]);
//       }
//     }
//   }
// }

// __device__ __host__ NPeriodVal
// clkedWaveformIdx(EventTimeVal time, PeriodVal period)
// {
//   return floor(time / period);
// }
// template<typename T>
// __device__ __host__ NPeriodVal
// clkedWaveformIdx(T, PeriodVal period) = delete;  // https://zhuanlan.zhihu.com/p/687000894

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

  // return events[left].time >= time ? left : right;
  assert(left == right);
  return right;
}

__device__ NEeventVal 
getAccuEventCountOfGateByTime(const sta::power::Gate* gate, const sta::power::Event *events, VcdEventTime time)
{
  NEeventVal accu_event_count = 0;
  for (NPinVal pin_idx = 0; pin_idx < gate->n_pin; ++pin_idx) {
    NEeventVal cur_pin_event_idx = getEventIdxByTime(
      events,
      gate->pin_waveform_starts[pin_idx], 
      gate->pin_waveform_ends[pin_idx],
      time
    );
    // the cur_pin_event_idx -1 is the last event index of which the time is less than time variable
    accu_event_count += cur_pin_event_idx - gate->pin_waveform_starts[pin_idx];  // we do not add one here, since the last one is out of range of time
  }

  return accu_event_count;
}

// get the minimum time that the accumulated event count to it is larger than or equals to accu_event_count
__device__ VcdEventTime 
getTimeByAccuEventCount(const sta::power::Gate* gate, const sta::power::Event *events, NEeventVal accu_event_count, VcdEventTime min_time, VcdEventTime max_time)
{
  VcdEventTime left = min_time, right = max_time;  // max_time is already an open interval here, just use it
  while (left < right) {  // [left, right)
    VcdEventTime mid = left + (right - left) / 2;  // avoid overflow

    if (getAccuEventCountOfGateByTime(gate, events, mid) < accu_event_count) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  // return getAccuEventCountOfGateByTime(gate, events, left) >= accu_event_count ? left : right;
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

  // return getAccuEventCountOfGateByTime(gate, events, left) >= accu_event_count ? left : right;
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
  // if (prev_val == cur_val) {
  //   return 0.0;
  // }

  // float n_rise = 0.0;
  // if (cur_val == 2) {  // 'X'
  //   if (prev_val == 0) {
  //     n_rise = 0.5;
  //   }
  // } else if (cur_val == 1) {
  //   if (prev_val == 2) {
  //     n_rise = 0.5;
  //   } else if (prev_val == 0) {
  //     n_rise = 1.0;
  //   }
  // }

  // return n_rise;

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
  // if (prev_val == cur_val) {
  //   return 0.0;
  // }

  // return (prev_val == 2 || cur_val == 2) ? 0.5 : 1;  // TODO use truth table to decide whether this is correct?

  // float n_toggle = 0.0;
  // if (cur_val == 2) {  // 'X'
  //   n_toggle = 0.5;
  // } else if (cur_val == 1) {
  //   if (prev_val == 2) {
  //     n_toggle = 0.5;
  //   } else if (prev_val == 0) {
  //     n_toggle = 1.0;
  //   }
  // } else if (cur_val == 0) {
  //   if (prev_val == 2) {
  //     n_toggle = 0.5;
  //   } else if (prev_val == 1) {
  //     n_toggle = 1.0;
  //   }
  // }

  // return n_toggle;

  static const float nToggleTable[3][3] = {
    // 0      1      2
    {0.0,    1.0,    0.5},  // prev_val = 0
    {1.0,    0.0,    0.5},  // prev_val = 1
    {0.5,    0.5,    0.0}   // prev_val = 2
  };

  return nToggleTable[prev_val][cur_val];
}

__device__ void
resetBoolArray(bool* arr, size_t len)
{
  for (size_t i = 0; i < len; ++i) {
    arr[i] = false;
  }
}

__device__ int
getRiseFallArcIndex(RISEFALL from_rf, RISEFALL to_rf)
{
  // if (from_rf == FALL && to_rf == FALL) {
  //   return 0;
  // } else if (from_rf == FALL && to_rf == RISE) {
  //   return 1;
  // } else if (from_rf == RISE && to_rf == FALL) {
  //   return 2;
  // } else if (from_rf == RISE && to_rf == RISE) {
  //   return 3;
  // }
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

__device__ NToggleVal
getTimeBasedNGlitch(VcdEventTime prev_time, VcdEventVal prev_val, VcdEventTime cur_time, VcdEventVal cur_val, EventTimeVal vcd_time_scale, SlewVal sum_slew)
{
  const VcdEventTime pulse_width = cur_time - prev_time;
  assert(pulse_width > 0);
  if (pulse_width * vcd_time_scale <= (sum_slew / 2.0)) {
    NToggleVal scaling_ratio = getTimeBasedGlitchScalingRatio(pulse_width * vcd_time_scale, sum_slew);
    NToggleVal n_cur_glitch = ((prev_val == 2 || cur_val == 2) ? 0.5 : 1) * scaling_ratio * 0.5;  // TODO that 0.5 is wrong
    return n_cur_glitch;
  } else {
    return 0.0;
  }
}
}