#include <cstddef>
#include <cassert>
#include <cmath>  // floor

#include "PowerUtils.hh"
#include "StringUtil.hh"
#include "GlobalConfig.hh"
#include "Graph.hh"
#include "LeakagePower.hh"
#include "InternalPower.hh"

using namespace sta;

namespace power::utils {
// NPeriodVal
// clkedWaveformIdx(EventTimeVal time, PeriodVal period)
// {
//   return floor(time / period);
// }
// template<typename T>
// NPeriodVal 
// clkedWaveformIdx(T, PeriodVal period) = delete;  // https://zhuanlan.zhihu.com/p/687000894

NPeriodVal
clkedWaveformIdx(VcdEventTime time, EventTimeVal vcd_time_scale, PeriodVal period)
{
  VcdEventTime time_unit_per_cycle = static_cast<VcdEventTime>(ceil(period / vcd_time_scale));
  return time / time_unit_per_cycle;
}

std::string
pinStateAnnotation(char val, std::string pin_name) {
  return val == '1' ? pin_name : "!" + pin_name;
}

NStateVal
getPinStatesIndex(const std::vector<VcdEventVal>& pin_states)
{
  NStateVal res = 0;
  for (size_t pin_idx = 0; pin_idx < pin_states.size(); ++pin_idx) {
    VcdEventVal cur_pin_state = pin_states[pin_idx] == 1 ? 1 : 0;  // 0 or 'X' are both 0;
    res += cur_pin_state * (1 << pin_idx);
  }
  return res;
}

VcdEventVal
getVertexDefaultState(const Vertex* vertex)
{
  if (vertex == nullptr || !vertex->isConstant()) {
    return 2;
  }

  const LogicValue value = vertex->simValue();
  if (value == LogicValue::zero) {
    return 0;
  }
  if (value == LogicValue::one) {
    return 1;
  }
  return 2;
}

void
findLeakageVal(
  const std::vector<VcdEventVal>& pin_states,
  const std::vector<PowerVal>& leakage_power_values,
  PowerVal default_leakage_power_val,
  bool default_leakage_exists,
  // Return values
  PowerVal* leakage_val
)
{
  if (leakage_power_values.size() != 0) {  // has leakage power table, n_pin <= G_CONFIG.nums.max_n_pin_for_leakage_power
    // *leakage_val = leakage_power_values.at(getPinStatesIndex(pin_states));
    for (size_t i_leak = 0; i_leak < leakage_power_values.size(); ++i_leak) { // do not use BSIM here
      if (i_leak == getPinStatesIndex(pin_states)) {
        *leakage_val = leakage_power_values.at(i_leak);
        break;
      }
    }
  } else if (default_leakage_exists) {
    *leakage_val = default_leakage_power_val;
  } else {
    *leakage_val = 0.0;
  }
}

// find leakage power value according to previous state
void
findLeakageVal(
  const std::vector<std::string>& input_pin_states, 
  const std::unordered_map<std::string, const sta::LeakagePower*>& when_str_to_leakage_power,
  const bool default_leakage_exists,
  const PowerVal default_leakage_power_val,
  // Return values
  PowerVal* leakage_val,
  std::string* input_state_annotation
  ) 
{
  auto tmp_states = input_pin_states;
  std::sort(tmp_states.begin(), tmp_states.end());
  *input_state_annotation = sta::strJoin(tmp_states, G_CONFIG.strs.leakage_power_separator);

  if (when_str_to_leakage_power.find(*input_state_annotation) != when_str_to_leakage_power.end()) {
    *leakage_val = when_str_to_leakage_power.at(*input_state_annotation)->power();
  } else {
    bool substr_condition_found = false;
    for (auto it = when_str_to_leakage_power.begin(); it != when_str_to_leakage_power.end(); ++it) {
      if (input_state_annotation->find(it->first) != std::string::npos) {
        substr_condition_found = true;
        *leakage_val = it->second->power();
        break;
      }
    }

    if (!substr_condition_found) {  // default condition
      if (!default_leakage_exists) {
        // LOG_ERROR << "Unconditioned leakage power has no default setting in lib file";
        *leakage_val = 0.0;
      }
      *leakage_val = default_leakage_power_val;
    }
  }
}

bool
isInternalPowerMatchPinStates(
  const std::vector<VcdEventVal>& pin_states,
  const InternalPower *internal_pwr,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map 
)
{
  const auto& when_states = internal_pwr->whenStates();
  const auto& port_names = internal_pwr->portNames();
  bool matched = true;
  assert(when_states.size() == port_names.size());
  for (size_t port_idx = 0; port_idx < when_states.size(); ++port_idx) {
    const auto& when_state = when_states.at(port_idx);
    NPinVal corr_pin_idx = port_name_to_idx_map.at(port_names.at(port_idx));
    VcdEventVal cur_port_state = 1;
    if (when_state.at(0) == '!') {
      cur_port_state = 0;
    }
    if (cur_port_state != ((pin_states.at(corr_pin_idx) == 0 || pin_states.at(corr_pin_idx) == 2) ? 0 : 1)) {
      matched = false;
    }
  }

  return matched;
}

float
getNRise(
  VcdEventVal prev_val,
  VcdEventVal cur_val
) {
  if (prev_val == cur_val) {
    return 0.0;
  }

  float n_rise = 0.0;
  if (cur_val == 2) {  // 'X'
    if (prev_val == 0) {
      n_rise = 0.5;
    }
  } else if (cur_val == 1) {
    if (prev_val == 2) {
      n_rise = 0.5;
    } else if (prev_val == 0) {
      n_rise = 1.0;
    }
  }

  return n_rise;
}

RISEFALL
getRiseFallEdge(
  VcdEventVal prev_val,
  VcdEventVal cur_val
) {
  return getNRise(prev_val, cur_val) > 0 ? RISE : FALL;
}

NToggleVal
getNToggle(
  VcdEventVal prev_val,
  VcdEventVal cur_val
) {
  if (prev_val == cur_val) {
    return 0.0;
  }

  return (prev_val == 2 || cur_val == 2) ? 0.5 : 1;  // TODO use truth table to decide whether this is correct?

  // NToggleVal n_toggle = 0.0;
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
}

void
resetBoolVector(std::vector<bool>& arr)
{
  for (size_t i = 0; i < arr.size(); ++i) {
    arr[i] = false;
  }
}

NToggleVal
getTimeBasedGlitchScalingRatioH(EventTimeVal pulse_width, SlewVal sum_slew)
{
  if (sum_slew == 0) {
    return 1.0;
  }

  NToggleVal ratio = (pulse_width * 2.0) / sum_slew;
  return std::min(static_cast<NToggleVal>(1.0), ratio * ratio);  // clip to 1
}

} // end of namespace power::utils
