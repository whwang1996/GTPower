#pragma once
#include <vector>
#include <string>
#include <unordered_map>

#include "Types.hh"
#include "Enums.hh"
#include "StringUtil.hh"

namespace sta {
  class FuncExpr;
  class InternalPower;
  class Vertex;
}

namespace power::utils {

// NPeriodVal
// clkedWaveformIdx(EventTimeVal time, PeriodVal period);

NPeriodVal
clkedWaveformIdx(VcdEventTime time, EventTimeVal vcd_time_scale, PeriodVal period);
template<typename T>
NPeriodVal
clkedWaveformIdx(T, T, PeriodVal period) = delete;

std::string
pinStateAnnotation(char val, std::string pin_name);

NStateVal
getPinStatesIndex(const std::vector<VcdEventVal>& pin_states);

VcdEventVal
getVertexDefaultState(const sta::Vertex* vertex);

void
findLeakageVal(
  const std::vector<VcdEventVal>& pin_states,
  const std::vector<PowerVal>& leakage_power_values,
  PowerVal default_leakage_power_val,
  bool default_leakage_exists,
  // Return values
  PowerVal* leakage_val
);

// Convert a conjunction to per-pin requirements (0, 1, or unconstrained).
// A missing when is unconditional. Return false for an unsatisfiable condition;
// report an error if the expression cannot be represented as one conjunction.
bool
getWhenPinStates(
  const sta::FuncExpr* when,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  std::vector<VcdEventVal>& when_pin_states
);

bool
isInternalPowerMatchPinStates(
  const std::vector<VcdEventVal>& pin_states,
  const sta::InternalPower *internal_pwr,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map 
);

float
getNRise(
  VcdEventVal prev_val,
  VcdEventVal cur_val
);
template<typename T>
float
getNRise(T, T) = delete;

RISEFALL
getRiseFallEdge(
  VcdEventVal prev_val,
  VcdEventVal cur_val
);
template<typename T>
RISEFALL
getRiseFallEdge(T, T) = delete;

NToggleVal
getNToggle(
  VcdEventVal prev_val,
  VcdEventVal cur_val
);
template<typename T>
NToggleVal
getNToggle(T, T) = delete;

void
resetBoolVector(std::vector<bool>& arr);

NToggleVal
getTimeBasedGlitchScalingRatioH(EventTimeVal pulse_width, SlewVal sum_slew);

} // end of namespace power::utils
