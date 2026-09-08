// OpenSTA, Static Timing Analyzer
// Copyright (c) 2024, Parallax Software, Inc.
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "LeakagePower.hh"

#include "FuncExpr.hh"
#include "TableModel.hh"
#include "Liberty.hh"
#include "GlobalConfig.hh"

namespace sta {

LeakagePowerAttrs::LeakagePowerAttrs() :
  when_(nullptr),
  power_(0.0)
{
}

void
LeakagePowerAttrs::setPower(float power)
{
  power_ = power;
}

////////////////////////////////////////////////////////////////

LeakagePower::LeakagePower(LibertyCell *cell,
			   LeakagePowerAttrs *attrs) :
  cell_(cell),
  when_(attrs->when()),
  when_str_(),
  when_states_(),
  power_(attrs->power())
{
  split(attrs->whenStr(), G_CONFIG.strs.leakage_power_separator, when_states_);
  std::sort(when_states_.begin(), when_states_.end());
  for (std::string& when_state: when_states_) {
    trim(when_state);
  }
  when_str_ = strJoin(when_states_, G_CONFIG.strs.leakage_power_separator);

  cell->addLeakagePower(this);
}

LeakagePower::~LeakagePower()
{
  if (when_)
    when_->deleteSubexprs();
}

} // namespace
