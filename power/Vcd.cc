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

#include <cassert>

#include "Vcd.hh"

#include "Report.hh"

namespace sta {

Vcd::Vcd(StaState *sta) :
  StaState(sta),
  time_scale_(1.0),
  time_unit_scale_(1.0),
  max_var_name_length_(0),
  max_var_width_(0),
  min_delta_time_(0),
  time_max_(0),
  time_intervals_()
{
}

Vcd::Vcd(const Vcd &vcd) :
  StaState(vcd),
  date_(vcd.date_),
  comment_(vcd.comment_),
  version_(vcd.version_),
  time_scale_(vcd.time_scale_),
  time_unit_(vcd.time_unit_),
  time_unit_scale_(vcd.time_unit_scale_),
  vars_(vcd.vars_),
  var_name_map_(vcd.var_name_map_),
  max_var_name_length_(vcd.max_var_name_length_),
  max_var_width_(vcd.max_var_width_),
  id_values_map_(vcd.id_values_map_),
  min_delta_time_(vcd.min_delta_time_),
  time_max_(vcd.time_max_),
  time_intervals_(vcd.time_intervals_)
{
}

Vcd &
Vcd::operator=(Vcd &&vcd1)
{
  date_ = std::move(vcd1.date_);
  comment_ = std::move(vcd1.comment_);
  version_ = std::move(vcd1.version_);
  time_scale_ = vcd1.time_scale_;
  time_unit_ = std::move(vcd1.time_unit_);
  time_unit_scale_ = vcd1.time_unit_scale_;
  vars_ = std::move(vcd1.vars_);
  var_name_map_ = std::move(vcd1.var_name_map_);
  max_var_name_length_ = vcd1.max_var_name_length_;
  max_var_width_ = vcd1.max_var_width_;
  // id_values_map_ = vcd1.id_values_map_;
  std::unordered_map<std::string, sta::VcdValues>::iterator iter;
  for (iter = vcd1.id_values_map_.begin(); iter != vcd1.id_values_map_.end(); ++iter) {
    id_values_map_[iter->first] = std::move(vcd1.id_values_map_[iter->first]);
  }
  min_delta_time_ = vcd1.min_delta_time_;
  time_max_ = vcd1.time_max_;
  time_intervals_ = std::move(vcd1.time_intervals_);

  // vcd1.vars_.clear();
  return *this;
}

Vcd &
Vcd::operator=(const Vcd &vcd1)
{
  date_ = vcd1.date_;
  comment_ = vcd1.comment_;
  version_ = vcd1.version_;
  time_scale_ = vcd1.time_scale_;
  time_unit_ = vcd1.time_unit_;
  time_unit_scale_ = vcd1.time_unit_scale_;
  vars_ = vcd1.vars_;
  var_name_map_ = vcd1.var_name_map_;
  max_var_name_length_ = vcd1.max_var_name_length_;
  max_var_width_ = vcd1.max_var_width_;
  id_values_map_ = vcd1.id_values_map_;
  min_delta_time_ = vcd1.min_delta_time_;
  time_max_ = vcd1.time_max_;
  time_intervals_ = vcd1.time_intervals_;

  return *this;
}

Vcd::~Vcd()
{
  for (VcdVar *var : vars_)
    delete var;
}

void
Vcd::setTimeUnit(const string &time_unit,
                 double time_unit_scale)
{
  time_unit_ = time_unit;
  time_unit_scale_ = time_unit_scale;
}

void
Vcd::setDate(const string &date)
{
  date_ = date;
}

void
Vcd::setComment(const string &comment)
{
  comment_ = comment;
}

void
Vcd::setVersion(const string &version)
{
  version_ = version;
}

void
Vcd::setTimeScale(double time_scale)
{
  time_scale_ = time_scale;
}

void
Vcd::setMinDeltaTime(VcdTime min_delta_time)
{
  min_delta_time_ = min_delta_time;
}

void
Vcd::setTimeMax(VcdTime time_max)
{
  time_max_ = time_max;
}

void
Vcd::setTimeIntervals(const std::vector<std::pair<VcdEventTime, VcdEventTime>>& time_intervals)
{
  if (time_intervals_.size() != 0) {
    LOG_ERROR << "time_intervals_ has already been set before, only one should be set.";
  }
  time_intervals_ = time_intervals;
}

void
Vcd::makeVar(string &name,
             VcdVarType type,
             int width,
             string &id)
{
  VcdVar *var = new VcdVar(name, type, width, id);
  vars_.push_back(var);
  var_name_map_[name] = var;
  max_var_name_length_ = std::max(max_var_name_length_, name.size());
  max_var_width_ = std::max(max_var_width_, width);
  // Make entry for var ID.
  id_values_map_[id].clear();
}

void
Vcd::sortVarsById()
{
  std::stable_sort(vars_.begin(), vars_.end(),
    [](const VcdVar* a, const VcdVar* b) {
      // ids are created by std::to_string(var_id), so numeric.
      const long long ida = std::stoll(a->id());
      const long long idb = std::stoll(b->id());
      return ida < idb;
    });
}

VcdVar *
Vcd::var(const string name)
{
  return var_name_map_[name];
}

bool
Vcd::varIdValid(const string &id) const
{
  return id_values_map_.find(id) != id_values_map_.end();
}

void
Vcd::varAppendValue(const string &id,
                    VcdTime time,
                    char value)
{
  VcdValues &values = id_values_map_.at(id);
  if (values.empty() || (values.size() && values[values.size() - 1].value() != value)) {  // remove redundant event
    assert(values.empty() || (values.size() && values[values.size() - 1].time() < time && values[values.size() - 1].busWidth() == 1));
    if (values.empty() && time != 0) {  // insert an initial condition if no given
      values.emplace_back(0, 'X', 0, 1);
    }
    values.emplace_back(time, value, 0, 1);
  }
}

void
Vcd::varAppendBusValue(const string &id,
                       VcdTime time,
                       int64_t bus_value,
                       int bus_width)
{
  VcdValues &values = id_values_map_.at(id);
  if (values.empty() || (values.size() && values[values.size() - 1].busValue() != static_cast<uint64_t>(bus_value))) {  // remove redundant event
    assert(values.empty() || (values.size() && values[values.size() - 1].time() < time));
    if (values.empty() && time != 0) {  // insert an initial condition if no given
      values.emplace_back(0, '\0', 0, bus_width);
    }
    values.emplace_back(time, '\0', bus_value, bus_width);
  }
}

const VcdValues &
Vcd::values(const VcdVar *var) const
{
  if (id_values_map_.find(var->id()) == id_values_map_.end()) {
    report_->error(1360, "Unknown variable %s ID %s", var->name().c_str(), var->id().c_str());
    static VcdValues empty;
    return empty;
  }
  else
    return id_values_map_.at(var->id());
}

////////////////////////////////////////////////////////////////

VcdVar::VcdVar(string name,
               VcdVarType type,
               int width,
               string id) :
  name_(name),
  type_(type),
  width_(width),
  id_(id)
{
}

VcdValue::VcdValue(VcdTime time,
                   char value,
                   uint64_t bus_value,
                   int bus_width) :
  time_(time),
  value_(value),
  bus_value_(bus_value),
  bus_width_(bus_width)
{
}

char
VcdValue::value(int value_bit) const
{
  if (value_ == '\0')
    return ((bus_value_ >> value_bit) & 0x1) ? '1' : '0';
  else
    return value_;
}

}  // namespace sta
