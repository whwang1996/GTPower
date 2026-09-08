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

#include "ReadVcdActivities.hh"

#include <algorithm>
#include <inttypes.h>
#include <string>

#include "Debug.hh"
#include "GlobalConfig.hh"
#include "Network.hh"
#include "ParseBus.hh"
#include "Power.hh"
#include "Sdc.hh"
#include "Sta.hh"
#include "VcdReader.hh"
#ifdef GTP_ENABLE_FSDB
#include "FsdbReader.hh"
#endif
#include "VerilogNamespace.hh"
#include "Log.hh"
#include "ScopedTimer.hh"

namespace sta {

using std::abs;
using std::min;
using std::string;
using std::to_string;

typedef Set<const Pin *> ConstPinSet;

static bool
hasExtension(const char *filename,
             const string &extension)
{
  string file_name(filename ? filename : "");
  return file_name.size() >= extension.size()
      && file_name.compare(file_name.size() - extension.size(), extension.size(), extension) == 0;
}

class ReadVcdActivities : public StaState
{
public:
  ReadVcdActivities(const char *filename,
                    const char *scope,
                    Sta *sta);
  void readActivities();

private:
  void reportWaveformStat();
  void reportIntervalActivityFactor(const char *title,
                                    const char *prefix,
                                    size_t n_signal,
                                    const std::vector<double> &interval_n_event);
  void setActivities();
  void setVarActivity(VcdVar *var,
                      string &var_name,
                      const VcdValues &var_value);
  void setVarActivity(const char *pin_name,
                      const VcdValues &var_values,
                      int value_bit);
  void setPinActivity(const Pin *pin, 
                      const VcdValues &var_values, 
                      int value_bit);
  void findVarActivity(const VcdValues &var_values,
                       int value_bit,
                       // Return values.
                       double &transition_count,
                       double &activity,
                       double &duty);
  void checkClkPeriod(const Pin *pin,
                      double transition_count);
  bool isDigitalVar(const VcdVar *var) const;
  bool hasDesignSignal(const char *pin_name);
  bool isClockSignal(const char *pin_name);
  double eventCount(const VcdValues &var_values,
                    int value_bit) const;
  double eventCount(const VcdValues &var_values,
                    int value_bit,
                    VcdEventTime start_time,
                    VcdEventTime end_time) const;
  size_t eventIndexAtOrAfter(const VcdValues &var_values,
                             VcdEventTime time) const;
  double nCycle() const;
  double nCycle(VcdEventTime start_time,
                VcdEventTime end_time) const;

  const char *filename_;
  const char *scope_;
  Vcd vcd_;
  double clk_period_;
  Sta *sta_;
  Power *power_;
  ConstPinSet annotated_pins_;

  static constexpr double sim_clk_period_tolerance_ = .1;
};

void
readVcdActivities(const char *filename,
                  const char *scope,
                  Sta *sta)
{
  ReadVcdActivities reader(filename, scope, sta);
  reader.readActivities();
}

ReadVcdActivities::ReadVcdActivities(const char *filename,
                                     const char *scope,
                                     Sta *sta) :
  StaState(sta),
  filename_(filename),
  scope_(scope),
  vcd_(sta),
  clk_period_(0.0),
  sta_(sta),
  power_(sta->power())
{
}

void
ReadVcdActivities::readActivities()
{
  utils::ScopedTimer timer_read_act("Read Activities");
  TIMERSTART(READ_ACTIVITIES);

  clk_period_ = INF;
  for (Clock *clk : *sta_->sdc()->clocks())
    clk_period_ = min(static_cast<double>(clk->period()), clk_period_);

  if (hasExtension(filename_, ".fsdb")) {
    G_CONFIG.strs.activity_file_format = "fsdb";
  } else if (hasExtension(filename_, ".vcd") || hasExtension(filename_, ".vcd.gz")) {
    G_CONFIG.strs.activity_file_format = "vcd";
  } else {
    LOG_ERROR << "Unsupported activity file extension for "
              << (filename_ ? filename_ : "<null>")
              << ". Supported extensions: .vcd, .vcd.gz"
#ifdef GTP_ENABLE_FSDB
              << ", .fsdb"
#endif
              << ".";
  }

  if (G_CONFIG.strs.activity_file_format == "fsdb") {
#ifdef GTP_ENABLE_FSDB
    vcd_ = readFsdbFile(filename_, sta_, clk_period_);
#else
    LOG_ERROR << "FSDB support is disabled in this build. Reconfigure with "
              << "-DFSDB_READER_DIR=/path/to/verdi/share/FsdbReader "
              << "and rebuild GTPower to read .fsdb files.";
#endif
  } else if (G_CONFIG.strs.activity_file_format == "vcd") {
    vcd_ = readVcdFile(filename_, sta_);
  } else {
    LOG_ERROR << "Unknown activity file foramt: " << G_CONFIG.strs.activity_file_format;
  }
  if (G_CONFIG.flags.report_vcd_stat) {
    reportWaveformStat();
  }

  if (vcd_.timeMax() > 0) {
    setActivities();
    power_->vcd() = std::move(vcd_);
    power_->setClkPeriod(static_cast<PeriodVal>(clk_period_));
  }
  else
    report_->warn(1450, "Waveform max time is zero.");
  report_->reportLine("Annotated %zu pin activities.", annotated_pins_.size());

  TIMEREND(READ_ACTIVITIES);
  DURATION_ms(READ_ACTIVITIES);
}

bool
ReadVcdActivities::isDigitalVar(const VcdVar *var) const
{
  return var->type() == VcdVarType::wire
      || var->type() == VcdVarType::reg
      || var->type() == VcdVarType::tri
      || var->type() == VcdVarType::triand
      || var->type() == VcdVarType::trior
      || var->type() == VcdVarType::trireg
      || var->type() == VcdVarType::tri0
      || var->type() == VcdVarType::tri1
      || var->type() == VcdVarType::wand
      || var->type() == VcdVarType::wor;
}

bool
ReadVcdActivities::hasDesignSignal(const char *pin_name)
{
  if (sdc_network_->findNet(pin_name)
      || sdc_network_->findPin(pin_name))
    return true;

  string var_name(pin_name);
  replaceLast(var_name, "\\/", "/");
  return sdc_network_->findPin(var_name.c_str()) != nullptr;
}

bool
ReadVcdActivities::isClockSignal(const char *pin_name)
{
  if (const Net *net = sdc_network_->findNet(pin_name)) {
    NetPinIterator *it = network_->pinIterator(net);
    while (it->hasNext()) {
      const Pin *pin = it->next();
      if (sdc_->isLeafPinClock(pin)) {
        delete it;
        return true;
      }
    }
    delete it;
  }
  else if (const Pin *pin = sdc_network_->findPin(pin_name))
    return sdc_->isLeafPinClock(pin);

  string var_name(pin_name);
  replaceLast(var_name, "\\/", "/");
  if (const Pin *pin = sdc_network_->findPin(var_name.c_str()))
    return sdc_->isLeafPinClock(pin);

  return false;
}

double
ReadVcdActivities::eventCount(const VcdValues &var_values,
                              int value_bit) const
{
  if (var_values.empty())
    return 0.0;

  double event_count = 0.0;
  char prev_value = var_values[0].value(value_bit);
  for (const VcdValue &var_value : var_values) {
    char value = var_value.value(value_bit);
    if (value != prev_value)
      event_count += 1.0;
    prev_value = value;
  }
  return event_count;
}

double
ReadVcdActivities::eventCount(const VcdValues &var_values,
                              int value_bit,
                              VcdEventTime start_time,
                              VcdEventTime end_time) const
{
  if (var_values.empty() || end_time <= start_time)
    return 0.0;

  size_t event_idx = eventIndexAtOrAfter(var_values, start_time);
  const size_t end_idx = eventIndexAtOrAfter(var_values, end_time);
  if (event_idx >= end_idx)
    return 0.0;

  char prev_value = event_idx == 0
      ? var_values[event_idx].value(value_bit)
      : var_values[event_idx - 1].value(value_bit);
  double event_count = 0.0;
  for (; event_idx < end_idx; ++event_idx) {
    char value = var_values[event_idx].value(value_bit);
    if (value != prev_value)
      event_count += 1.0;
    prev_value = value;
  }
  return event_count;
}

size_t
ReadVcdActivities::eventIndexAtOrAfter(const VcdValues &var_values,
                                       VcdEventTime time) const
{
  auto event_itr = std::lower_bound(var_values.begin(), var_values.end(), time,
                                    [](const VcdValue &value,
                                       VcdEventTime time) {
                                      return value.time() < time;
                                    });
  return event_itr - var_values.begin();
}

double
ReadVcdActivities::nCycle() const
{
  if (clk_period_ <= 0.0 || clk_period_ == INF)
    return 0.0;
  return vcd_.timeMax() * vcd_.timeScale() / clk_period_;
}

double
ReadVcdActivities::nCycle(VcdEventTime start_time,
                          VcdEventTime end_time) const
{
  if (clk_period_ <= 0.0 || clk_period_ == INF || end_time <= start_time)
    return 0.0;

  VcdEventTime clamped_start = start_time < 0 ? 0 : start_time;
  VcdEventTime clamped_end = end_time < 0 ? 0 : end_time;
  const VcdEventTime time_max = vcd_.timeMax();
  clamped_start = min(clamped_start, time_max);
  clamped_end = min(clamped_end, time_max);
  if (clamped_end <= clamped_start)
    return 0.0;

  return (clamped_end - clamped_start) * vcd_.timeScale() / clk_period_;
}

void
ReadVcdActivities::reportWaveformStat()
{
  utils::ScopedTimer timer_report_waveform_activity_factor("Report Waveform Activity Factor");

  const auto& vars = vcd_.vars();
  const auto& time_intervals = vcd_.timeIntervals();

  // LOG_BEGIN(INFO, "Waveform size distribution");
  // size_t n_event = 0;
  // for (const VcdVar* var: vars) {
  //   assert(vcd_.varIdValid(var->id()));
  //   const auto& values = vcd_.values(var);
  //   n_event += values.size();
  //   LOG_INFO << var->id() << "\t" << values.size();
  // }
  // LOG_END(INFO, "Waveform size distribution");

  size_t n_rise = 0, n_fall = 0, n_x = 0, n_non_initial_x = 0;
  size_t waveform_signal_n_signal = 0;
  double waveform_signal_n_event = 0.0;
  size_t waveform_signal_n_clock_signal = 0;
  size_t waveform_signal_n_unmatched_signal = 0;
  std::vector<double> waveform_signal_interval_n_event(time_intervals.size(), 0.0);
  const size_t scope_length = strlen(scope_);
  for (const VcdVar* var: vars) {
    assert(vcd_.varIdValid(var->id()));
    const auto& values = vcd_.values(var);

    for (size_t event_idx = 0; event_idx < values.size(); ++event_idx) {
      const auto& event = values[event_idx];
      for (int bit = 0; bit < var->width(); ++bit) {
        char event_val = event.value(bit);
        if (event_val == '1') {
          ++n_rise;
        } else if (event_val == '0') {
          ++n_fall;
        }  else if (event_val == 'X') {
          ++n_x;
          if (event_idx != 0) {
            ++n_non_initial_x;
            // LOG_WARN << "Var id " << var->id() << " name " << var->name() << " time " << event.time() << " has non initial X transition.";
          }
        } else {
          LOG_ERROR << "Unexpected event value: " << event_val;
        }
      }
    }

    if (values.empty() || !isDigitalVar(var))
      continue;

    string var_name = var->name();
    if (scope_length) {
      if (var_name.substr(0, scope_length) != scope_)
        continue;
      var_name = var_name.substr(scope_length + 1);
    }

    if (var->width() == 1) {
      string sta_name = netVerilogToSta(var_name.c_str());
      if (isClockSignal(sta_name.c_str())) {
        waveform_signal_n_clock_signal++;
        continue;
      }
      if (!hasDesignSignal(sta_name.c_str())) {
        waveform_signal_n_unmatched_signal++;
        continue;
      }
      waveform_signal_n_signal++;
      waveform_signal_n_event += eventCount(values, 0);
      for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
        const auto& interval = time_intervals.at(interval_idx);
        waveform_signal_interval_n_event.at(interval_idx) += eventCount(values, 0, interval.first, interval.second);
      }
    }
    else {
      bool is_bus, is_range, subscript_wild;
      string bus_name;
      int from, to;
      parseBusName(var_name.c_str(), '[', ']', '\\', is_bus, is_range, bus_name, from, to, subscript_wild);
      if (!is_bus) {
        waveform_signal_n_unmatched_signal += var->width();
        continue;
      }

      string sta_bus_name = netVerilogToSta(bus_name.c_str());
      int value_bit = 0;
      auto count_bus_bit = [&](int bus_bit) {
        string pin_name = sta_bus_name;
        pin_name += '[';
        pin_name += to_string(bus_bit);
        pin_name += ']';

        if (isClockSignal(pin_name.c_str())) {
          waveform_signal_n_clock_signal++;
          return;
        }
        if (!hasDesignSignal(pin_name.c_str())) {
          waveform_signal_n_unmatched_signal++;
          return;
        }
        waveform_signal_n_signal++;
        waveform_signal_n_event += eventCount(values, value_bit);
        for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
          const auto& interval = time_intervals.at(interval_idx);
          waveform_signal_interval_n_event.at(interval_idx) += eventCount(values, value_bit, interval.first, interval.second);
        }
      };

      if (to < from) {
        for (int bus_bit = to; bus_bit <= from; bus_bit++) {
          count_bus_bit(bus_bit);
          value_bit++;
        }
      }
      else {
        for (int bus_bit = to; bus_bit >= from; bus_bit--) {
          count_bus_bit(bus_bit);
          value_bit++;
        }
      }
    }
  }

  double n_cycle = nCycle();
  double waveform_signal_activity_factor = 0.0;
  if (waveform_signal_n_signal != 0 && n_cycle > 0.0)
    waveform_signal_activity_factor = waveform_signal_n_event / (waveform_signal_n_signal * n_cycle);

  LOG_BEGIN(INFO, "Stat information of waveform");
  LOG_INFO << "Max time " << vcd_.timeMax();
  LOG_INFO << "Time unit " << vcd_.timeUnit();
  LOG_INFO << "ID number " << vcd_.idValuesMap().size();
  LOG_INFO << "Var number " << vars.size();
  // LOG_INFO << "Event number " << n_event;
  LOG_INFO << "n_rise: " << n_rise << " n_fall: " << n_fall << " n_x: " << n_x << " n_non_initial_x: " << n_non_initial_x;
  LOG_INFO << "Waveform signal activity factor N_signal: " << waveform_signal_n_signal;
  LOG_INFO << "Waveform signal activity factor N_event: " << waveform_signal_n_event;
  LOG_INFO << "Waveform signal activity factor N_cycle: " << n_cycle;
  LOG_INFO << "Waveform signal activity factor: " << waveform_signal_activity_factor;
  LOG_INFO << "Waveform signal activity factor skipped clock signals: " << waveform_signal_n_clock_signal;
  LOG_INFO << "Waveform signal activity factor skipped unmatched signals: " << waveform_signal_n_unmatched_signal;
  LOG_END(INFO, "Stat information of waveform");

  reportIntervalActivityFactor("Waveform signal activity factor by interval",
                               "Waveform signal activity factor",
                               waveform_signal_n_signal,
                               waveform_signal_interval_n_event);
}

void
ReadVcdActivities::reportIntervalActivityFactor(const char *title,
                                                const char *prefix,
                                                size_t n_signal,
                                                const std::vector<double> &interval_n_event)
{
  const auto& time_intervals = vcd_.timeIntervals();
  if (time_intervals.empty())
    return;

  LOG_BEGIN(INFO, title);
  for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
    const auto& interval = time_intervals.at(interval_idx);
    const double n_event = interval_idx < interval_n_event.size()
        ? interval_n_event.at(interval_idx)
        : 0.0;
    const double n_cycle = nCycle(interval.first, interval.second);
    double activity_factor = 0.0;
    if (n_signal != 0 && n_cycle > 0.0)
      activity_factor = n_event / (n_signal * n_cycle);
    LOG_INFO << prefix << " interval " << interval_idx
             << " start time: " << interval.first
             << " end time: " << interval.second
             << " N_signal: " << n_signal
             << " N_event: " << n_event
             << " N_cycle: " << n_cycle
             << " activity factor: " << activity_factor;
  }
  LOG_END(INFO, title);
}

void
ReadVcdActivities::setActivities()
{
  size_t scope_length = strlen(scope_);
  for (VcdVar *var : vcd_.vars()) {
    const VcdValues &var_values = vcd_.values(var);
    if (!var_values.empty() && (var->type() == VcdVarType::wire || var->type() == VcdVarType::reg)) {
      string var_name = var->name();
      // string::starts_with in c++20
      if (scope_length) {
        if (var_name.substr(0, scope_length) == scope_) {
          var_name = var_name.substr(scope_length + 1);  // remove scope prefix
          setVarActivity(var, var_name, var_values);
        }
      }
      else
        setVarActivity(var, var_name, var_values);
    }
  }
}

void
ReadVcdActivities::setVarActivity(VcdVar *var,
                                  string &var_name,
                                  const VcdValues &var_values)
{
  if (var->width() == 1) {
    string sta_name = netVerilogToSta(var_name.c_str());
    setVarActivity(sta_name.c_str(), var_values, 0);
  }
  else {
    bool is_bus, is_range, subscript_wild;
    string bus_name;
    int from, to;
    parseBusName(var_name.c_str(), '[', ']', '\\', is_bus, is_range, bus_name, from, to, subscript_wild);
    if (is_bus) {
      string sta_bus_name = netVerilogToSta(bus_name.c_str());
      int value_bit = 0;
      if (to < from) {
        for (int bus_bit = to; bus_bit <= from; bus_bit++) {
          string pin_name = sta_bus_name;
          pin_name += '[';
          pin_name += to_string(bus_bit);
          pin_name += ']';
          setVarActivity(pin_name.c_str(), var_values, value_bit);
          value_bit++;
        }
      }
      else {
        for (int bus_bit = to; bus_bit >= from; bus_bit--) {
          string pin_name = sta_bus_name;
          pin_name += '[';
          pin_name += to_string(bus_bit);
          pin_name += ']';
          setVarActivity(pin_name.c_str(), var_values, value_bit);
          value_bit++;
        }
      }
    }
    else
      report_->warn(1451, "problem parsing bus %s.", var_name.c_str());
  }
}

void
ReadVcdActivities::setVarActivity(const char *pin_name,
                                  const VcdValues &var_values,
                                  int value_bit)
{
  if (const Net *net = sdc_network_->findNet(pin_name); net) {  // annotate net first
    NetPinIterator *it = network_->pinIterator(net);
    while (it->hasNext()) {
      const Pin *pin = it->next();
      // if (LOG_DEBUG_FLAG) {
      //   LOG_DEBUG << "Var name: " << pin_name << " Net: " << network_->pathName(net) << " pin " << network_->pathName(pin);
      // }
      setPinActivity(pin, var_values, value_bit);
    }
    delete it;
  } else if (const Pin *pin = sdc_network_->findPin(pin_name); pin) {  // if net not found, use pin instead
    setPinActivity(pin, var_values, value_bit);
  } else {
    std::string var_name(pin_name);
    replaceLast(var_name, "\\/", "/");  // try to match cell_path/port_name pattern
    LOG_INFO << "Try to replace " << pin_name << " to " << var_name << " for pin matching";
    if (const Pin *pin = sdc_network_->findPin(var_name.c_str()); pin) {
      setPinActivity(pin, var_values, value_bit);
    } else {
      LOG_WARN << "No matched net or pin for " << pin_name;
    }
  }
}

void
ReadVcdActivities::setPinActivity(const Pin *pin, 
                                  const VcdValues &var_values, 
                                  int value_bit)
{
  double transition_count = 0.0, activity = 0.0, duty = 0.0;
  bool is_clock = sdc_->isLeafPinClock(pin);
  if (!G_CONFIG.flags.enable_time_based_analysis
      || G_CONFIG.flags.report_vcd_stat
      || is_clock) {
    findVarActivity(var_values, value_bit, transition_count, activity, duty);
  }
  if (is_clock && transition_count > 0.0)
    checkClkPeriod(pin, transition_count);
  power_->setUserActivity(pin, activity, duty, PwrActivityOrigin::vcd, &var_values, value_bit);
  annotated_pins_.insert(pin);
}

void
ReadVcdActivities::findVarActivity(const VcdValues &var_values,
                                   int value_bit,
                                   // Return values.
                                   double &transition_count,
                                   double &activity,
                                   double &duty)
{
  transition_count = 0.0;
  char prev_value = var_values[0].value(value_bit);
  VcdTime prev_time = var_values[0].time();
  VcdTime high_time = 0;
  for (const VcdValue &var_value : var_values) {
    VcdTime time = var_value.time();
    char value = var_value.value(value_bit);
    debugPrint(debug_, "read_vcd_activities", 3, " %" PRId64 " %c", time, value);
    if (prev_value == '1')
      high_time += time - prev_time;
    if (value != prev_value)
      transition_count += (value == 'X' || value == 'Z' || prev_value == 'X' || prev_value == 'Z')
          ? .5
          : 1.0;
    prev_time = time;
    prev_value = value;
  }
  VcdTime time_max = vcd_.timeMax();
  if (prev_value == '1')
    high_time += time_max - prev_time;
  duty = static_cast<double>(high_time) / time_max;
  activity = transition_count / (time_max * vcd_.timeScale() / clk_period_);
}

void
ReadVcdActivities::checkClkPeriod(const Pin *pin,
                                  double transition_count)
{
  VcdTime time_max = vcd_.timeMax();
  double sim_period = time_max * vcd_.timeScale() / (transition_count / 2.0);

  ClockSet *clks = sdc_->findLeafPinClocks(pin);
  if (clks) {
    for (Clock *clk : *clks) {
      double clk_period = clk->period();
      if (abs((clk_period - sim_period) / clk_period) > .1)
        // Warn if sim clock period differs from SDC by 10%.
        report_->warn(1452, "clock %s vcd period %s differs from SDC clock period %s", clk->name(), delayAsString(sim_period, this), delayAsString(clk_period, this));
    }
  }
}

}  // namespace sta
