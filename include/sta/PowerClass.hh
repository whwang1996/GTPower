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

#pragma once
#include <iostream>
#include <fstream>

#include "Types.hh"
#include "Network.hh"
#include "FileHelper.hh"

namespace sta {

class Power;
class VcdValue;
typedef vector<VcdValue> VcdValues;

enum class PwrActivityOrigin
{
 global,
 input,
 user,
 vcd,
 propagated,
 clock,
 constant,
 defaulted,
 unknown
};

class PwrActivity
{
public:
  PwrActivity();
  PwrActivity(float activity,
	      float duty,
	      PwrActivityOrigin origin,
        const VcdValue* vcd_values=nullptr,
        NEeventVal n_event=0,
        int value_bit=-1);
  float activity() const { return activity_; }
  void setActivity(float activity);
  float duty() const { return duty_; }
  void setDuty(float duty);
  PwrActivityOrigin origin() { return origin_; }
  void setOrigin(PwrActivityOrigin origin);
  const char *originName() const;
  const VcdValue* vcdValues() const { return vcd_values_; }
  NEeventVal nEvent() const { return n_event_; }
  int valueBit() const { return value_bit_; }
  void set(float activity,
	   float duty,
	   PwrActivityOrigin origin);
  bool isSet() const;

private:
  void check();

  // In general activity is per clock cycle, NOT per second.
  float activity_;
  float duty_;
  PwrActivityOrigin origin_;
  const VcdValue* vcd_values_;
  NEeventVal n_event_;
  int value_bit_;

  static constexpr float min_activity = 1E-10;
};

class PowerClkedWaveform
{
public:
  PowerClkedWaveform(PeriodVal period, NPeriodVal n_period);
  std::vector<PowerVal>& waveform () { return waveform_; }
  const std::vector<PowerVal>& waveform () const { return waveform_; }

private:
  PeriodVal period_; // clock period
  std::vector<PowerVal> waveform_; // average power of each period
};

class PowerResult
{
public:
  PowerResult();
  void clear();
  PowerVal &internal() { return internal_; }
  PowerVal &switching() { return switching_; }
  PowerVal &glitchSwitching() { return glitch_switching_; }
  PowerVal &glitchInternal() { return glitch_internal_; }
  PowerVal &leakage() { return leakage_; }
  PowerVal internal() const { return internal_; }
  PowerVal switching() const { return switching_; }
  PowerVal glitchSwitching() const { return glitch_switching_; }
  PowerVal glitchInternal() const { return glitch_internal_; }
  PowerVal leakage() const { return leakage_; }
  PowerVal total() const;
  void incr(const PowerResult &result);
  void initLeakagePowerClkedWaveform(const Instance* inst, PeriodVal clk_period, NPeriodVal n_period) { inst_to_leakage_power_clked_waveform_map_.emplace(inst, sta::PowerClkedWaveform{clk_period, n_period}); }
  sta::PowerClkedWaveform& findLeakagePowerClkedWaveform(const Instance* inst) { return inst_to_leakage_power_clked_waveform_map_.at(inst); }
  void initInternalPowerClkedWaveform(const Instance* inst, PeriodVal clk_period, NPeriodVal n_period) { inst_to_internal_power_clked_waveform_map_.emplace(inst, sta::PowerClkedWaveform{clk_period, n_period}); }
  sta::PowerClkedWaveform& findInternalPowerClkedWaveform(const Instance* inst) { return inst_to_internal_power_clked_waveform_map_.at(inst); }
  void initGlitchInternalPowerClkedWaveform(const Instance* inst, PeriodVal clk_period, NPeriodVal n_period) { inst_to_glitch_internal_power_clked_waveform_map_.emplace(inst, sta::PowerClkedWaveform{clk_period, n_period}); }
  sta::PowerClkedWaveform& findGlitchInternalPowerClkedWaveform(const Instance* inst) { return inst_to_glitch_internal_power_clked_waveform_map_.at(inst); }
  void initSwitchingPowerClkedWaveform(const Instance* inst, PeriodVal clk_period, NPeriodVal n_period) { inst_to_switching_power_clked_waveform_map_.emplace(inst, sta::PowerClkedWaveform{clk_period, n_period}); }
  sta::PowerClkedWaveform& findSwitchingPowerClkedWaveform(const Instance* inst) { return inst_to_switching_power_clked_waveform_map_.at(inst); }
  void initGlitchSwitchingPowerClkedWaveform(const Instance* inst, PeriodVal clk_period, NPeriodVal n_period) { inst_to_glitch_switching_power_clked_waveform_map_.emplace(inst, sta::PowerClkedWaveform{clk_period, n_period}); }
  sta::PowerClkedWaveform& findGlitchSwitchingPowerClkedWaveform(const Instance* inst) { return inst_to_glitch_switching_power_clked_waveform_map_.at(inst); }

  void printPerCycleResults(Network* network) const {
    namespace fs = std::filesystem;
    std::ofstream out_file;
    out_file.open(fs::path(utils::get_power_analysis_per_cycle_waveform_res_path()), std::ios::out);
    out_file << "Per cycle leakage power: " << std::endl;
    printSinglePerCycleResult(inst_to_leakage_power_clked_waveform_map_, network, out_file);
    out_file << "Per cycle internal power: " << std::endl;
    printSinglePerCycleResult(inst_to_internal_power_clked_waveform_map_, network, out_file);
    out_file << "Per cycle glitch internal power: " << std::endl;
    printSinglePerCycleResult(inst_to_glitch_internal_power_clked_waveform_map_, network, out_file);
    out_file << "Per cycle switching power: " << std::endl;
    printSinglePerCycleResult(inst_to_switching_power_clked_waveform_map_, network, out_file);
    out_file << "Per cycle glitch switching power: " << std::endl;
    printSinglePerCycleResult(inst_to_glitch_switching_power_clked_waveform_map_, network, out_file);
    out_file << "End of per cycle power" << std::endl;
  }

private:
  void printSinglePerCycleResult(const std::map<const Instance*, sta::PowerClkedWaveform>& InstToPowerClkedWaveformMap, Network* network, std::ofstream& out_file) const {
    for (auto it = InstToPowerClkedWaveformMap.begin(); it != InstToPowerClkedWaveformMap.end(); ++it) {
      out_file << network->pathName(it->first) << G_CONFIG.strs.power_analysis_res_file_separator;
      for (NPeriodVal i = 0; i < it->second.waveform().size(); ++i) {
        out_file << it->second.waveform()[i] << G_CONFIG.strs.power_analysis_res_file_separator;
      }
      out_file << std::endl;
    }
  }

  PowerVal internal_;
  PowerVal glitch_internal_;
  PowerVal switching_;
  PowerVal glitch_switching_;
  PowerVal leakage_;
  std::map<const Instance*, sta::PowerClkedWaveform> inst_to_leakage_power_clked_waveform_map_;
  std::map<const Instance*, sta::PowerClkedWaveform> inst_to_internal_power_clked_waveform_map_;
  std::map<const Instance*, sta::PowerClkedWaveform> inst_to_glitch_internal_power_clked_waveform_map_;
  std::map<const Instance*, sta::PowerClkedWaveform> inst_to_switching_power_clked_waveform_map_;
  std::map<const Instance*, sta::PowerClkedWaveform> inst_to_glitch_switching_power_clked_waveform_map_;
};

} // namespace
