#pragma once
#include <vector>
#include <string>
#include <unordered_map>

#include "Types.hh"
#include "Enums.hh"
#include "StringUtil.hh"

namespace sta {
  class LeakagePower;
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

void
findLeakageVal(
  const std::vector<std::string>& input_pin_states, 
  const std::unordered_map<std::string, const sta::LeakagePower*>& when_str_to_leakage_power,
  const bool default_leakage_exists,
  const PowerVal default_leakage_power_val,
  PowerVal* leakage_val,
  std::string* input_state_annotation
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
