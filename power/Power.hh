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

#include <utility>
#include <unordered_map>
#include <vector>

#include "Bdd.hh"
#include "Vcd.hh"
#include "Network.hh"
#include "PowerClass.hh"
#include "SdcClass.hh"
#include "StaConfig.hh"  // CUDD
#include "StaState.hh"
#include "UnorderedMap.hh"
#include "Enums.hh"
#include "Liberty.hh"

struct DdNode;
struct DdManager;

namespace sta {

class Sta;
class Corner;
class DcalcAnalysisPt;
class BfsFwdIterator;
class Vertex;

}  // namespace sta

namespace gtpower {

class PropActivityVisitor;

typedef std::pair<const sta::Instance *, sta::LibertyPort *> SeqPin;

class SeqPinHash
{
public:
  SeqPinHash(const sta::Network *network);
  size_t operator()(const SeqPin &pin) const;

private:
  const sta::Network *network_;
};

class SeqPinEqual
{
public:
  bool operator()(const SeqPin &pin1,
                  const SeqPin &pin2) const;
};

typedef sta::UnorderedMap<const sta::Pin *, PwrActivity> PwrActivityMap;
typedef sta::UnorderedMap<SeqPin, PwrActivity, SeqPinHash, SeqPinEqual> PwrSeqActivityMap;

// The Power class has access to Sta components directly for
// convenience but also requires access to the Sta class member functions.
class Power : public sta::StaState
{
public:
  Power(sta::StaState *sta);
  void getCircuitStat();
  virtual void power(const sta::Corner *corner,
             // Return values.
             PowerResult &total,
             PowerResult &sequential,
             PowerResult &combinational,
             PowerResult &clock,
             PowerResult &macro,
             PowerResult &pad);
  PowerResult power(const sta::Instance *inst,
                    const sta::Corner *corner);
  void setGlobalActivity(float activity,
                         float duty);
  void setInputActivity(float activity,
                        float duty);
  void setInputPortActivity(const sta::Port *input_port,
                            float activity,
                            float duty);
  PwrActivity &activity(const sta::Pin *pin);
  void setUserActivity(const sta::Pin *pin,
                       float activity,
                       float duty,
                       PwrActivityOrigin origin,
                       const VcdValues* var_values_ptr=nullptr,
                       int value_bit=-1);
  // Activity is toggles per second.
  PwrActivity findClkedActivity(const sta::Pin *pin);
  PeriodVal clkPeriod() const { return clk_period_; }
  void setClkPeriod(PeriodVal clk_period) { clk_period_ = clk_period; }
  Vcd& vcd() { return vcd_; }
  virtual void printCellRes() const;

protected:
  Vcd vcd_;
  PeriodVal clk_period_;
  const std::unordered_map<char, VcdEventVal> vcd_value_to_int_map_ = {{'0', 0}, {'1', 1}, {'X', 2}, {'Z', 3}};

  bool inClockNetwork(const sta::Instance *inst);
  void powerInside(const sta::Instance *hinst,
                   const sta::Corner *corner,
                   PowerResult &result);
  void ensureActivities();
  void reportResolvedTimeBasedPowerAnalysisWorkload();
  bool hasUserActivity(const sta::Pin *pin);
  PwrActivity &userActivity(const sta::Pin *pin);
  void setSeqActivity(const sta::Instance *reg,
                      sta::LibertyPort *output,
                      PwrActivity &activity);
  bool hasSeqActivity(const sta::Instance *reg,
                      sta::LibertyPort *output);
  PwrActivity &seqActivity(const sta::Instance *reg,
                           sta::LibertyPort *output);
  bool hasActivity(const sta::Pin *pin);
  void setActivity(const sta::Pin *pin,
                   PwrActivity &activity);

  PowerResult power(const sta::Instance *inst,
                    sta::LibertyCell *cell,
                    const sta::Corner *corner);
  void findInternalPower(const sta::Instance *inst,
                         sta::LibertyCell *cell,
                         const sta::Corner *corner,
                         const sta::Clock *inst_clk,
                         // Return values.
                         PowerResult &result);
  void findInputInternalPower(const sta::Pin *to_pin,
                              sta::LibertyPort *to_port,
                              const sta::Instance *inst,
                              sta::LibertyCell *cell,
                              PwrActivity &to_activity,
                              float load_cap,
                              const sta::Corner *corner,
                              // Return values.
                              PowerResult &result);
  void findOutputInternalPower(const sta::LibertyPort *to_port,
                               const sta::Instance *inst,
                               sta::LibertyCell *cell,
                               PwrActivity &to_activity,
                               float load_cap,
                               const sta::Corner *corner,
                               // Return values.
                               PowerResult &result);
  void findLeakagePower(const sta::Instance *inst,
                        sta::LibertyCell *cell,
                        const sta::Corner *corner,
                        // Return values.
                        PowerResult &result);
  void findSwitchingPower(const sta::Instance *inst,
                          sta::LibertyCell *cell,
                          const sta::Corner *corner,
                          const sta::Clock *inst_clk,
                          // Return values.
                          PowerResult &result);
  // -----------------------time based power analysis---------------------------
  PowerResult timeBasedPower(const sta::Instance *inst,
                  sta::LibertyCell *cell,
                  const sta::Corner *corner,
                  // Return values.
                  PowerResult& total_result);
  void findTimeBasedAllPower(const sta::Instance *inst,
                        sta::LibertyCell *cell,
                        const sta::Corner *corner,
                        // Return values.
                        PowerResult &result,
                        PowerResult &total_result);
  void calculatePerCycleLeakagePower(
                          const sta::Instance *inst,
                          const PowerVal leakage_val,
                          const VcdTime prev_time,
                          const VcdTime cur_time,
                          const double time_scale,
                          const double clk_period,
                          PowerResult& result) const;
  void getPinInformation(const sta::Instance *inst, std::vector<const sta::Pin *> *pins, NPinVal *n_pin, NPinVal *n_input_pin, NPinVal *n_output_pin) const;
  void getStateIdx(
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map, 
    const sta::FuncExpr* when,
    // Return values.
    std::vector<NStateVal>& matched_state_idxs
  ) const;
  void getEventClockCycleBasedGlitchFlags(
    const VcdValue *vcd_values,
    NEeventVal start_idx_in_vcd_values,
    NEeventVal end_idx_in_vcd_values,
    // Return values.
    std::vector<bool>& event_glitch_flags
  );
  NEeventVal getEventIdxByTime(const VcdValue* vcd_values, NEeventVal n_event, VcdEventTime time) const;
  NEeventVal getEventIdxByTime(const VcdValue* vcd_values, NEeventVal n_event, VcdEventTime time, NEeventVal left) const;
  NEeventVal getEventIdxByTime(const VcdValues& vcd_values, VcdEventTime time) const;
  void getLeakagePower(
    const sta::LibertyCell *cell, 
    const sta::LibertyCell *corner_cell, 
    const NPinVal n_pin,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    // Return values.
    std::vector<PowerVal>& leakage_power_values,
    PowerVal& default_leakage_power_val,
    bool& default_leakage_exists
  ) const;
  NToggleVal getGlitchScalingRatioClockCycleBasedH(
    NEeventVal cur_event_idx, const sta::Pin* pin,
    NEeventVal n_event, const VcdValue* vcd_values,
    const std::vector<bool>& event_glitch_flag,
    const sta::Corner *corner
  );
  void findInputInternalVal(
    const sta::LibertyCell* corner_cell,
    const std::vector<VcdEventVal>& pin_states,
    const sta::Pin* toggle_pin,
    RISEFALL rise_fall,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    const sta::Corner *corner,
    const sta::DcalcAnalysisPt* dcalc_ap,
    // Return values.
    EnergyVal* internal_energy
  );
  void findOutputInternalVal(
    const sta::Instance *inst,
    const sta::LibertyCell* corner_cell,
    const std::vector<VcdEventVal>& pin_states,
    VcdEventTime cur_time,
    const sta::Pin* toggle_pin,
    RISEFALL to_rf,
    NPinVal n_input_pin,
    const std::vector<const sta::Pin *>& pins,
    const std::vector<PwrActivity>& pin_activities,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    const sta::Corner *corner,
    const sta::DcalcAnalysisPt* dcalc_ap,
    // Return values.
    EnergyVal* internal_energy
  );
  void getDefaultOutputPinInternalEnergyVal(
    const sta::Instance *inst,
    const std::vector<const sta::Pin *>& pins,
    const sta::Pin* toggle_pin,
    RISEFALL to_rf,
    NPinVal n_input_pin,
    const sta::InternalPowerSeq& internal_pwrs,
    const sta::Corner *corner,
    const sta::DcalcAnalysisPt* dcalc_ap,
    // Return values.
    EnergyVal* internal_energy
  );
  // -----------------------end of time based power analysis---------------------------
  float getSlew(sta::Vertex *vertex,
                const sta::RiseFall *rf,
                const sta::Corner *corner);
  const sta::Clock *findInstClk(const sta::Instance *inst);
  const sta::Clock *findClk(const sta::Pin *to_pin);
  float clockDuty(const sta::Clock *clk);
  PwrActivity findClkedActivity(const sta::Pin *pin,
                                const sta::Clock *inst_clk);
  PwrActivity findActivity(const sta::Pin *pin);
  PwrActivity findSeqActivity(const sta::Instance *inst,
                              sta::LibertyPort *port);
  float portVoltage(const sta::LibertyCell *cell,
                    const sta::LibertyPort *port,
                    const sta::DcalcAnalysisPt *dcalc_ap) const;
  float pgNameVoltage(const sta::LibertyCell *cell,
                      const char *pg_port_name,
                      const sta::DcalcAnalysisPt *dcalc_ap) const;
  void seedActivities(sta::BfsFwdIterator &bfs);
  void seedRegOutputActivities(const sta::Instance *reg,
                               sta::Sequential *seq,
                               sta::LibertyPort *output,
                               bool invert);
  void seedRegOutputActivities(const sta::Instance *inst,
                               sta::BfsFwdIterator &bfs);
  PwrActivity evalActivity(sta::FuncExpr *expr,
                           const sta::Instance *inst);
  PwrActivity evalActivity(sta::FuncExpr *expr,
                           const sta::Instance *inst,
                           const sta::LibertyPort *cofactor_port,
                           bool cofactor_positive);
  sta::LibertyPort *findExprOutPort(sta::FuncExpr *expr);
  float findInputDuty(const sta::Instance *inst,
                      sta::FuncExpr *func,
                      sta::InternalPower *pwr);
  float evalDiffDuty(sta::FuncExpr *expr,
                     sta::LibertyPort *from_port,
                     const sta::Instance *inst);
  sta::LibertyPort *findLinkPort(const sta::LibertyCell *cell,
                            const sta::LibertyPort *corner_port);
  sta::Pin *findLinkPin(const sta::Instance *inst,
                   const sta::LibertyPort *corner_port);
  void clockGatePins(const sta::Instance *inst,
                     // Return values.
                     const sta::Pin *&enable,
                     const sta::Pin *&clk,
                     const sta::Pin *&gclk) const;
  float evalBddActivity(DdNode *bdd,
                        const sta::Instance *inst);
  float evalBddDuty(DdNode *bdd,
                    const sta::Instance *inst);

private:
  // Port/pin activities set by set_pin_activity.
  // set_pin_activity -global
  PwrActivity global_activity_;
  // set_pin_activity -input
  PwrActivity input_activity_;
  // set_pin_activity -input_ports -pins
  PwrActivityMap user_activity_map_;
  // Propagated activities.
  PwrActivityMap activity_map_;
  PwrSeqActivityMap seq_activity_map_;
  bool activities_valid_;
  sta::Bdd bdd_;
  std::vector<std::pair<const sta::Instance*, PowerResult>> cell_power_results_;

  static constexpr int max_activity_passes_ = 100;

  friend class PropActivityVisitor;
};

}  // namespace gtpower
