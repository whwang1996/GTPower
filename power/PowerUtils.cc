#include <cstddef>
#include <cassert>
#include <cmath>  // floor

#include "PowerUtils.hh"
#include "StringUtil.hh"
#include "GlobalConfig.hh"
#include "Graph.hh"
#include "InternalPower.hh"
#include "FuncExpr.hh"
#include "Liberty.hh"
#include "Defines.hh"
#include "Log.hh"

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

namespace {

NPinVal
whenPinIndex(
  const FuncExpr* expr,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
)
{
  const LibertyPort* port = expr->port();
  const auto pin = port_name_to_idx_map.find(port->name());
  if (pin == port_name_to_idx_map.end()) {
    LOG_ERROR << "Power when expression references an unavailable pin: "
      << port->libertyCell()->name() << "/" << port->name();
    return INVALID_PIN_IDX;
  }
  return pin->second;
}

bool
collectWhenPinStates(
  const FuncExpr* expr,
  bool negated,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  std::vector<VcdEventVal>& when_pin_states
)
{
  switch (expr->op()) {
  case FuncExpr::op_port: {
    const NPinVal pin_idx = whenPinIndex(expr, port_name_to_idx_map);
    const VcdEventVal required_state = negated ? 0 : 1;
    VcdEventVal& previous_state = when_pin_states.at(pin_idx);
    if (previous_state != INVALID_VCD_EVENT_VAL && previous_state != required_state) {
      return false;
    }
    previous_state = required_state;
    return true;
  }
  case FuncExpr::op_not:
    return collectWhenPinStates(expr->left(), !negated, port_name_to_idx_map, when_pin_states);
  case FuncExpr::op_one:
    return !negated;
  case FuncExpr::op_zero:
    return negated;
  case FuncExpr::op_and:
  case FuncExpr::op_or:
    // De Morgan's law also lets !(A | B) use the same pin-state representation.
    if ((expr->op() == FuncExpr::op_and && !negated)
        || (expr->op() == FuncExpr::op_or && negated)) {
      const bool left = collectWhenPinStates(expr->left(), negated, port_name_to_idx_map, when_pin_states);
      const bool right = collectWhenPinStates(expr->right(), negated, port_name_to_idx_map, when_pin_states);
      return left && right;
    }
    break;
  case FuncExpr::op_xor:
    break;
  }

  LOG_ERROR << "Unsupported power when expression: " << (negated ? "!(" : "(")
    << expr->asString() << "). State-based power lookup requires a conjunction of pin states; "
    << "general OR/XOR conditions are not supported.";
  return false;
}

bool
evalWhen(
  const FuncExpr* expr,
  const std::vector<VcdEventVal>& pin_states,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
)
{
  if (expr == nullptr) {
    return true;
  }
  switch (expr->op()) {
  case FuncExpr::op_port:
    // Keep the existing convention that X is treated as zero for power lookup.
    return pin_states.at(whenPinIndex(expr, port_name_to_idx_map)) == 1;
  case FuncExpr::op_not:
    return !evalWhen(expr->left(), pin_states, port_name_to_idx_map);
  case FuncExpr::op_and:
    return evalWhen(expr->left(), pin_states, port_name_to_idx_map)
      && evalWhen(expr->right(), pin_states, port_name_to_idx_map);
  case FuncExpr::op_or:
    return evalWhen(expr->left(), pin_states, port_name_to_idx_map)
      || evalWhen(expr->right(), pin_states, port_name_to_idx_map);
  case FuncExpr::op_xor:
    return evalWhen(expr->left(), pin_states, port_name_to_idx_map)
      != evalWhen(expr->right(), pin_states, port_name_to_idx_map);
  case FuncExpr::op_one:
    return true;
  case FuncExpr::op_zero:
    return false;
  }
  LOG_ERROR << "Unknown operator in power when expression.";
  return false;
}

} // namespace

bool
getWhenPinStates(
  const FuncExpr* when,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  std::vector<VcdEventVal>& when_pin_states
)
{
  when_pin_states.assign(port_name_to_idx_map.size(), INVALID_VCD_EVENT_VAL);
  return when == nullptr
    || collectWhenPinStates(when, false, port_name_to_idx_map, when_pin_states);
}

bool
isInternalPowerMatchPinStates(
  const std::vector<VcdEventVal>& pin_states,
  const InternalPower *internal_pwr,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
)
{
  return evalWhen(internal_pwr->when(), pin_states, port_name_to_idx_map);
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
