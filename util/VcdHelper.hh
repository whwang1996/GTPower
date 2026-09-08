#pragma once
#include <algorithm>  // std::max

#include "Vcd.hh"

namespace utils {
int getVarBusWidth(const sta::VcdVar* var, const sta::Vcd& vcd) {
  const auto& values = vcd.values(var);
  if (values.size() == 0) {
    return 0;
  } else if (values.size() == 1) {
    return values[0].busWidth();
  } else {
    return std::max(values[0].busWidth(), values[1].busWidth());
  }
}

size_t getTotalBusWidthOfAllVars(const sta::Vcd& vcd) {
  size_t total_bus_width = 0;
  for (const sta::VcdVar* var: vcd.vars()) {
    total_bus_width += getVarBusWidth(var, vcd);
  }

  return total_bus_width;
}

}