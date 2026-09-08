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
class PropActivityVisitor;
class BfsFwdIterator;
class Vertex;

typedef std::pair<const Instance *, LibertyPort *> SeqPin;

class SeqPinHash
{
public:
  SeqPinHash(const Network *network);
  size_t operator()(const SeqPin &pin) const;

private:
  const Network *network_;
};

class SeqPinEqual
{
public:
  bool operator()(const SeqPin &pin1,
                  const SeqPin &pin2) const;
};

typedef UnorderedMap<const Pin *, PwrActivity> PwrActivityMap;
typedef UnorderedMap<SeqPin, PwrActivity, SeqPinHash, SeqPinEqual> PwrSeqActivityMap;

// The Power class has access to Sta components directly for
// convenience but also requires access to the Sta class member functions.
class Power : public StaState
{
public:
  Power(StaState *sta);
  void getCircuitStat();
  virtual void power(const Corner *corner,
             // Return values.
             PowerResult &total,
             PowerResult &sequential,
             PowerResult &combinational,
             PowerResult &clock,
             PowerResult &macro,
             PowerResult &pad);
  PowerResult power(const Instance *inst,
                    const Corner *corner);
  void setGlobalActivity(float activity,
                         float duty);
  void setInputActivity(float activity,
                        float duty);
  void setInputPortActivity(const Port *input_port,
                            float activity,
                            float duty);
  PwrActivity &activity(const Pin *pin);
  void setUserActivity(const Pin *pin,
                       float activity,
                       float duty,
                       PwrActivityOrigin origin,
                       const VcdValues* var_values_ptr=nullptr,
                       int value_bit=-1);
  // Activity is toggles per second.
  PwrActivity findClkedActivity(const Pin *pin);
  PeriodVal clkPeriod() const { return clk_period_; }
  void setClkPeriod(PeriodVal clk_period) { clk_period_ = clk_period; }
  Vcd& vcd() { return vcd_; }
  virtual void printCellRes() const;

protected:
  Vcd vcd_;
  PeriodVal clk_period_;
  const std::unordered_map<char, VcdEventVal> vcd_value_to_int_map_ = {{'0', 0}, {'1', 1}, {'X', 2}, {'Z', 3}};

  bool inClockNetwork(const Instance *inst);
  void powerInside(const Instance *hinst,
                   const Corner *corner,
                   PowerResult &result);
  void ensureActivities();
  void reportResolvedTimeBasedPowerAnalysisWorkload();
  bool hasUserActivity(const Pin *pin);
  PwrActivity &userActivity(const Pin *pin);
  void setSeqActivity(const Instance *reg,
                      LibertyPort *output,
                      PwrActivity &activity);
  bool hasSeqActivity(const Instance *reg,
                      LibertyPort *output);
  PwrActivity &seqActivity(const Instance *reg,
                           LibertyPort *output);
  bool hasActivity(const Pin *pin);
  void setActivity(const Pin *pin,
                   PwrActivity &activity);

  PowerResult power(const Instance *inst,
                    LibertyCell *cell,
                    const Corner *corner);
  void findInternalPower(const Instance *inst,
                         LibertyCell *cell,
                         const Corner *corner,
                         const Clock *inst_clk,
                         // Return values.
                         PowerResult &result);
  void findInputInternalPower(const Pin *to_pin,
                              LibertyPort *to_port,
                              const Instance *inst,
                              LibertyCell *cell,
                              PwrActivity &to_activity,
                              float load_cap,
                              const Corner *corner,
                              // Return values.
                              PowerResult &result);
  void findOutputInternalPower(const LibertyPort *to_port,
                               const Instance *inst,
                               LibertyCell *cell,
                               PwrActivity &to_activity,
                               float load_cap,
                               const Corner *corner,
                               // Return values.
                               PowerResult &result);
  void findLeakagePower(const Instance *inst,
                        LibertyCell *cell,
                        const Corner *corner,
                        // Return values.
                        PowerResult &result);
  void findSwitchingPower(const Instance *inst,
                          LibertyCell *cell,
                          const Corner *corner,
                          const Clock *inst_clk,
                          // Return values.
                          PowerResult &result);
  // -----------------------time based power analysis---------------------------
  PowerResult timeBasedPower(const Instance *inst,
                  LibertyCell *cell,
                  const Corner *corner,
                  // Return values.
                  PowerResult& total_result);
  void findTimeBasedAllPower(const Instance *inst,
                        LibertyCell *cell,
                        const Corner *corner,
                        // Return values.
                        PowerResult &result,
                        PowerResult &total_result);
  void calculatePerCycleLeakagePower(
                          const Instance *inst,
                          const PowerVal leakage_val,
                          const VcdTime prev_time,
                          const VcdTime cur_time,
                          const double time_scale,
                          const double clk_period,
                          PowerResult& result) const;
  void getPinInformation(const Instance *inst, std::vector<const Pin *> *pins, NPinVal *n_pin, NPinVal *n_input_pin, NPinVal *n_output_pin) const;
  void getStateIdx(
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map, 
    const FuncExpr* when,
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
    const LibertyCell *cell, 
    const LibertyCell *corner_cell, 
    const NPinVal n_pin,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    // Return values.
    std::vector<PowerVal>& leakage_power_values,
    PowerVal& default_leakage_power_val,
    bool& default_leakage_exists
  ) const;
  NToggleVal getGlitchScalingRatioClockCycleBasedH(
    NEeventVal cur_event_idx, const Pin* pin,
    NEeventVal n_event, const VcdValue* vcd_values,
    const std::vector<bool>& event_glitch_flag,
    const Corner *corner
  );
  void findInputInternalVal(
    const LibertyCell* corner_cell,
    const std::vector<VcdEventVal>& pin_states,
    const Pin* toggle_pin,
    RISEFALL rise_fall,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    const Corner *corner,
    const DcalcAnalysisPt* dcalc_ap,
    // Return values.
    EnergyVal* internal_energy
  );
  void findOutputInternalVal(
    const Instance *inst,
    const LibertyCell* corner_cell,
    const std::vector<VcdEventVal>& pin_states,
    VcdEventTime cur_time,
    const Pin* toggle_pin,
    RISEFALL to_rf,
    NPinVal n_input_pin,
    const std::vector<const Pin *>& pins,
    const std::vector<PwrActivity>& pin_activities,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    const Corner *corner,
    const DcalcAnalysisPt* dcalc_ap,
    // Return values.
    EnergyVal* internal_energy
  );
  void getDefaultOutputPinInternalEnergyVal(
    const Instance *inst,
    const std::vector<const Pin *>& pins,
    const Pin* toggle_pin,
    RISEFALL to_rf,
    NPinVal n_input_pin,
    const InternalPowerSeq& internal_pwrs,
    const Corner *corner,
    const DcalcAnalysisPt* dcalc_ap,
    // Return values.
    EnergyVal* internal_energy
  );
  // -----------------------end of time based power analysis---------------------------
  float getSlew(Vertex *vertex,
                const RiseFall *rf,
                const Corner *corner);
  const Clock *findInstClk(const Instance *inst);
  const Clock *findClk(const Pin *to_pin);
  float clockDuty(const Clock *clk);
  PwrActivity findClkedActivity(const Pin *pin,
                                const Clock *inst_clk);
  PwrActivity findActivity(const Pin *pin);
  PwrActivity findSeqActivity(const Instance *inst,
                              LibertyPort *port);
  float portVoltage(const LibertyCell *cell,
                    const LibertyPort *port,
                    const DcalcAnalysisPt *dcalc_ap) const;
  float pgNameVoltage(const LibertyCell *cell,
                      const char *pg_port_name,
                      const DcalcAnalysisPt *dcalc_ap) const;
  void seedActivities(BfsFwdIterator &bfs);
  void seedRegOutputActivities(const Instance *reg,
                               Sequential *seq,
                               LibertyPort *output,
                               bool invert);
  void seedRegOutputActivities(const Instance *inst,
                               BfsFwdIterator &bfs);
  PwrActivity evalActivity(FuncExpr *expr,
                           const Instance *inst);
  PwrActivity evalActivity(FuncExpr *expr,
                           const Instance *inst,
                           const LibertyPort *cofactor_port,
                           bool cofactor_positive);
  LibertyPort *findExprOutPort(FuncExpr *expr);
  float findInputDuty(const Instance *inst,
                      FuncExpr *func,
                      InternalPower *pwr);
  float evalDiffDuty(FuncExpr *expr,
                     LibertyPort *from_port,
                     const Instance *inst);
  LibertyPort *findLinkPort(const LibertyCell *cell,
                            const LibertyPort *corner_port);
  Pin *findLinkPin(const Instance *inst,
                   const LibertyPort *corner_port);
  void clockGatePins(const Instance *inst,
                     // Return values.
                     const Pin *&enable,
                     const Pin *&clk,
                     const Pin *&gclk) const;
  float evalBddActivity(DdNode *bdd,
                        const Instance *inst);
  float evalBddDuty(DdNode *bdd,
                    const Instance *inst);

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
  Bdd bdd_;
  std::vector<std::pair<const Instance*, PowerResult>> cell_power_results_;

  static constexpr int max_activity_passes_ = 100;

  friend class PropActivityVisitor;
};

}  // namespace sta
