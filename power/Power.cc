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

// #include <gperftools/profiler.h>

#include "Power.hh"

#include <algorithm>  // max, shuffle
#include <cmath>      // abs
#include <fstream>    // std::ofstream
#include <random>   // std::random_device, std::mt19937
#include <string>

#include "Bfs.hh"
#include "ClkNetwork.hh"
#include "Clock.hh"
#include "Corner.hh"
#include "DcalcAnalysisPt.hh"
#include "Debug.hh"
#include "EnumNameMap.hh"
#include "FileHelper.hh"
#include "FuncExpr.hh"
#include "Graph.hh"
#include "GraphDelayCalc.hh"
#include "Hash.hh"
#include "InternalPower.hh"
#include "LeakagePower.hh"
#include "Liberty.hh"
#include "MinMax.hh"
#include "Network.hh"
#include "PathVertex.hh"
#include "PortDirection.hh"
#include "Sdc.hh"
#include "Search.hh"
#include "Sequential.hh"
#include "TimingArc.hh"
#include "TimingRole.hh"
#include "Transition.hh"
#include "ThreadPool.hh"
#include "Units.hh"
#include "cudd.h"
#include "search/Levelize.hh"
#include "search/Sim.hh"
#include "Log.hh"
#include "ScopedTimer.hh"

// Related liberty not supported:
// library
//  default_cell_leakage_power : 0;
//  output_voltage (default_VDD_VSS_output) {
// leakage_power
//  related_pg_pin : VDD;
// internal_power
//  input_voltage : default_VDD_VSS_input;
// pin
//  output_voltage : default_VDD_VSS_output;
//
// transition_density = activity / clock_period

namespace sta {

using std::abs;
using std::isnormal;
using std::max;

static bool
isPositiveUnate(const LibertyCell *cell,
                const LibertyPort *from,
                const LibertyPort *to);

static EnumNameMap<PwrActivityOrigin> pwr_activity_origin_map =
    {{PwrActivityOrigin::global, "global"},
     {PwrActivityOrigin::input, "input"},
     {PwrActivityOrigin::user, "user"},
     {PwrActivityOrigin::vcd, "vcd"},
     {PwrActivityOrigin::propagated, "propagated"},
     {PwrActivityOrigin::clock, "clock"},
     {PwrActivityOrigin::constant, "constant"},
     {PwrActivityOrigin::defaulted, "defaulted"},
     {PwrActivityOrigin::unknown, "unknown"}};

Power::Power(StaState *sta) :
  StaState(sta),
  vcd_(sta),
  global_activity_{0.0, 0.0, PwrActivityOrigin::unknown},
  input_activity_{0.1, 0.5, PwrActivityOrigin::input},
  seq_activity_map_(100, SeqPinHash(network_), SeqPinEqual()),
  activities_valid_(false),
  bdd_(sta)
{
}

void
Power::setGlobalActivity(float activity,
                         float duty)
{
  global_activity_.set(activity, duty, PwrActivityOrigin::global);
  activities_valid_ = false;
}

void
Power::setInputActivity(float activity,
                        float duty)
{
  input_activity_.set(activity, duty, PwrActivityOrigin::input);
  activities_valid_ = false;
}

void
Power::setInputPortActivity(const Port *input_port,
                            float activity,
                            float duty)
{
  Instance *top_inst = network_->topInstance();
  const Pin *pin = network_->findPin(top_inst, input_port);
  if (pin) {
    user_activity_map_[pin] = {activity, duty, PwrActivityOrigin::user};  // this function seems only called when interactive with tcl, so just use nullptr here temporally
    activities_valid_ = false;
  }
}

void
Power::setUserActivity(const Pin *pin,
                       float activity,
                       float duty,
                       PwrActivityOrigin origin,
                       const VcdValues* var_values_ptr,
                       int value_bit)
{
  user_activity_map_[pin] = {activity, duty, origin, &((*var_values_ptr)[0]), var_values_ptr->size(), value_bit};
  activities_valid_ = false;
}

PwrActivity &
Power::userActivity(const Pin *pin)
{
  return user_activity_map_[pin];
}

bool
Power::hasUserActivity(const Pin *pin)
{
  return user_activity_map_.hasKey(pin);
}

void
Power::setActivity(const Pin *pin,
                   PwrActivity &activity)
{
  debugPrint(debug_, "power_activity", 3, "set %s %.2e %.2f %s", network_->pathName(pin), activity.activity(), activity.duty(), pwr_activity_origin_map.find(activity.origin()));
  activity_map_[pin] = activity;
}

PwrActivity &
Power::activity(const Pin *pin)
{
  return activity_map_[pin];
}

bool
Power::hasActivity(const Pin *pin)
{
  return activity_map_.hasKey(pin);
}

// Sequential internal pins may not be in the netlist so their
// activities are stored by instance/liberty_port pairs.
void
Power::setSeqActivity(const Instance *reg,
                      LibertyPort *output,
                      PwrActivity &activity)
{
  seq_activity_map_[SeqPin(reg, output)] = activity;
  activities_valid_ = false;
}

bool
Power::hasSeqActivity(const Instance *reg,
                      LibertyPort *output)
{
  return seq_activity_map_.hasKey(SeqPin(reg, output));
}

PwrActivity &
Power::seqActivity(const Instance *reg,
                   LibertyPort *output)
{
  return seq_activity_map_[SeqPin(reg, output)];
}

SeqPinHash::SeqPinHash(const Network *network) :
  network_(network)
{
}

size_t
SeqPinHash::operator()(const SeqPin &pin) const
{
  return hashSum(network_->id(pin.first), pin.second->id());
}

bool
SeqPinEqual::operator()(const SeqPin &pin1,
                        const SeqPin &pin2) const
{
  return pin1.first == pin2.first && pin1.second == pin2.second;
}

////////////////////////////////////////////////////////////////

void
Power::power(const Corner *corner,
             // Return values.
             PowerResult &total,
             PowerResult &sequential,
             PowerResult &combinational,
             PowerResult &clock,
             PowerResult &macro,
             PowerResult &pad)
{
  total.clear();
  sequential.clear();
  combinational.clear();
  clock.clear();
  macro.clear();
  pad.clear();
  cell_power_results_.clear();

  ensureActivities();
  LeafInstanceIterator *inst_iter = network_->leafInstanceIterator();
  size_t inst_count = 0;
  const Instance* top_instance = network_->topInstance();
  NPeriodVal n_period = ceil((vcd_.timeMax() * vcd_.timeScale()) / clk_period_);
  total.initLeakagePowerClkedWaveform(top_instance, clk_period_, n_period);
  total.initInternalPowerClkedWaveform(top_instance, clk_period_, n_period);
  total.initGlitchInternalPowerClkedWaveform(top_instance, clk_period_, n_period);
  total.initSwitchingPowerClkedWaveform(top_instance, clk_period_, n_period);
  total.initGlitchSwitchingPowerClkedWaveform(top_instance, clk_period_, n_period);

  LOG_INFO << "CPU power analysis, enable_time_based_analysis: " << (G_CONFIG.flags.enable_time_based_analysis ? "true": "false")
    << " enable_multi_threaded_cpu: " << (G_CONFIG.flags.enable_multi_threaded_cpu ? "true": "false")
    << " multi_thread_number: " << G_CONFIG.nums.multi_thread_number;
  
  // ProfilerStart("./log/cpu_time_based.prof");
  utils::ScopedTimer cpu_power_analysis_timer("CPU Power Analysis");
  TIMERSTART(CPU_POWER_ANALYSIS);
  if (G_CONFIG.flags.enable_multi_threaded_cpu) {
    if (!G_CONFIG.flags.enable_time_based_analysis) {
      LOG_ERROR << "Only time based power analysis is supported in multi-threaded mode";
    }
    LOG_INFO << "Enable multi threaded time-based power analysis, thread number: " << G_CONFIG.nums.multi_thread_number;
    ThreadPool thread_pool(G_CONFIG.nums.multi_thread_number);
    // ---------------------------scheduling threads------------------------------
    std::vector<Instance*> instances;
    while (inst_iter->hasNext()) {
      Instance *inst = inst_iter->next();
      LibertyCell *cell = network_->libertyCell(inst);
      if (cell) {
        instances.push_back(inst);
      }
    }
    delete inst_iter;
    // -----shuffle instances-----
    std::random_device rd;
    std::mt19937 random_engine(rd());
    std::shuffle(instances.begin(), instances.end(), random_engine);
    // -----end of shuffle instances-----
    LOG_INFO << "Total gates number: " << instances.size();
    const size_t base = instances.size() / G_CONFIG.nums.multi_thread_number;
    const size_t remainder = instances.size() % G_CONFIG.nums.multi_thread_number;
    std::vector<std::vector<Instance*>> per_thread_instances(G_CONFIG.nums.multi_thread_number);
    for (size_t t_id = 0; t_id < G_CONFIG.nums.multi_thread_number; ++t_id) {
      size_t start = t_id * base + std::min(t_id, remainder);
      size_t end = start + base + (t_id < remainder ? 1 : 0);
      LOG_INFO << "Thread " << t_id << " handles gates in range [" << start << ", " << end << ").";
      for (size_t gate_idx = start; gate_idx < end; gate_idx++) {
        per_thread_instances.at(t_id).push_back(instances.at(gate_idx));
      }
    }
    // ---------------------------end of scheduling threads------------------------------
    // ---------------------------launching threads------------------------------
    std::vector<std::future<void>> job_futures;
    std::vector<PowerResult> per_thread_total_results(G_CONFIG.nums.multi_thread_number);  // record power waveform
    std::vector<std::vector<PowerResult>> per_thread_gate_results(G_CONFIG.nums.multi_thread_number);  // record per gate power
    for (int t_id = 0; t_id < G_CONFIG.nums.multi_thread_number; ++t_id) {
      const auto& cur_thread_instances = per_thread_instances.at(t_id);
      auto& cur_thread_total_result = per_thread_total_results.at(t_id);
      auto& cur_thread_gate_result = per_thread_gate_results.at(t_id);
      auto job = [this, &cur_thread_instances, &cur_thread_total_result, &cur_thread_gate_result, corner, n_period, top_instance](){
        cur_thread_total_result.initLeakagePowerClkedWaveform(top_instance, clk_period_, n_period);
        cur_thread_total_result.initInternalPowerClkedWaveform(top_instance, clk_period_, n_period);
        cur_thread_total_result.initGlitchInternalPowerClkedWaveform(top_instance, clk_period_, n_period);
        cur_thread_total_result.initSwitchingPowerClkedWaveform(top_instance, clk_period_, n_period);
        cur_thread_total_result.initGlitchSwitchingPowerClkedWaveform(top_instance, clk_period_, n_period);
        for (const Instance* inst: cur_thread_instances) {
          LibertyCell *cell = network_->libertyCell(inst);
          cur_thread_gate_result.push_back(timeBasedPower(inst, cell, corner, cur_thread_total_result));
        }
      };
      job_futures.push_back(thread_pool.enqueue(job));
    }
    for (const auto& future: job_futures) {
      future.wait();
    }
    // ---------------------------end of launching threads------------------------------
    // ---------------------------merge result------------------------------
    for (int t_id = 0; t_id < G_CONFIG.nums.multi_thread_number; ++t_id) {
      for (NPeriodVal cycle_idx = 0; cycle_idx < n_period; ++cycle_idx) {
        total.findLeakagePowerClkedWaveform(top_instance).waveform()[cycle_idx] += per_thread_total_results.at(t_id).findLeakagePowerClkedWaveform(top_instance).waveform()[cycle_idx];
        total.findInternalPowerClkedWaveform(top_instance).waveform()[cycle_idx] += per_thread_total_results.at(t_id).findInternalPowerClkedWaveform(top_instance).waveform()[cycle_idx];
        total.findGlitchInternalPowerClkedWaveform(top_instance).waveform()[cycle_idx] += per_thread_total_results.at(t_id).findGlitchInternalPowerClkedWaveform(top_instance).waveform()[cycle_idx];
        total.findSwitchingPowerClkedWaveform(top_instance).waveform()[cycle_idx] += per_thread_total_results.at(t_id).findSwitchingPowerClkedWaveform(top_instance).waveform()[cycle_idx];
        total.findGlitchSwitchingPowerClkedWaveform(top_instance).waveform()[cycle_idx] += per_thread_total_results.at(t_id).findGlitchSwitchingPowerClkedWaveform(top_instance).waveform()[cycle_idx];
      }
      if (per_thread_instances.at(t_id).size() != per_thread_gate_results.at(t_id).size()) {
        LOG_ERROR << "per_thread_instances.at(t_id).size() " << per_thread_instances.at(t_id).size() <<  " != per_thread_gate_results.at(t_id).size() " << per_thread_gate_results.at(t_id).size();
      }
      for (size_t gate_idx = 0; gate_idx < per_thread_instances.at(t_id).size(); ++gate_idx) {
        const Instance* inst = per_thread_instances.at(t_id).at(gate_idx);
        LibertyCell *cell = network_->libertyCell(inst);
        const auto& inst_power = per_thread_gate_results.at(t_id).at(gate_idx);
        cell_power_results_.emplace_back(inst, inst_power);
        if (cell->isMacro() || cell->isMemory() || cell->interfaceTiming())
          macro.incr(inst_power);
        else if (cell->isPad())
          pad.incr(inst_power);
        else if (inClockNetwork(inst))
          clock.incr(inst_power);
        else if (cell->hasSequentials())
          sequential.incr(inst_power);
        else
          combinational.incr(inst_power);
        total.incr(inst_power);
      }
    }
    // ---------------------------end of merge result------------------------------
  } else {
    while (inst_iter->hasNext()) {
      if (inst_count++ % 1000 == 0) {
        LOG_INFO << "Power analysis inst_count: " << inst_count;
      }
      Instance *inst = inst_iter->next();
      LibertyCell *cell = network_->libertyCell(inst);
      if (cell) {
        PowerResult inst_power;
        if (G_CONFIG.flags.enable_time_based_analysis) {
          inst_power = timeBasedPower(inst, cell, corner, total);
          cell_power_results_.emplace_back(inst, inst_power);
        } else {
          inst_power = power(inst, cell, corner);
        }
        if (cell->isMacro() || cell->isMemory() || cell->interfaceTiming())
          macro.incr(inst_power);
        else if (cell->isPad())
          pad.incr(inst_power);
        else if (inClockNetwork(inst))
          clock.incr(inst_power);
        else if (cell->hasSequentials())
          sequential.incr(inst_power);
        else
          combinational.incr(inst_power);
        total.incr(inst_power);
      }
    }
    delete inst_iter;
  }
  TIMEREND(CPU_POWER_ANALYSIS);
  DURATION_ms(CPU_POWER_ANALYSIS);

  // ProfilerStop();
}

void
Power::printCellRes() const
{
  std::ofstream out_file;
  const std::string cell_res_path = utils::get_power_analysis_cell_res_path();
  out_file.open(cell_res_path.c_str(), std::ios::out);
  out_file << "cell" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Internal_Power" << G_CONFIG.strs.power_analysis_res_file_separator << "Glitch_Internal_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Switching_Power" << G_CONFIG.strs.power_analysis_res_file_separator << "Glitch_Switching_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Leakage_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Total_Power" << G_CONFIG.strs.power_analysis_res_file_separator
    << "Standard_Cell"
    << "\n";

  for (const auto& cell_power_result : cell_power_results_) {
    const Instance* inst = cell_power_result.first;
    const PowerResult& inst_power_res = cell_power_result.second;
    LibertyCell* cell = network_->libertyCell(inst);

    PowerVal cur_gate_regular_internal = inst_power_res.internal() - inst_power_res.glitchInternal();
    PowerVal cur_gate_glitch_internal = inst_power_res.glitchInternal();
    PowerVal cur_gate_regular_switching = inst_power_res.switching() - inst_power_res.glitchSwitching();
    PowerVal cur_gate_glitch_switching = inst_power_res.glitchSwitching();
    PowerVal cur_gate_leakage = inst_power_res.leakage();

    out_file << network_->pathName(inst) << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_internal << G_CONFIG.strs.power_analysis_res_file_separator << cur_gate_glitch_internal << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_switching << G_CONFIG.strs.power_analysis_res_file_separator << cur_gate_glitch_switching << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_leakage << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_internal + cur_gate_glitch_internal + cur_gate_regular_switching + cur_gate_glitch_switching + cur_gate_leakage << G_CONFIG.strs.power_analysis_res_file_separator
      << (cell ? cell->name() : "")
      << "\n";
  }
}

bool
Power::inClockNetwork(const Instance *inst)
{
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {
    const Pin *pin = pin_iter->next();
    if (network_->direction(pin)->isAnyOutput() && !clk_network_->isClock(pin)) {
      delete pin_iter;
      return false;
    }
  }
  delete pin_iter;
  return true;
}

PowerResult
Power::power(const Instance *inst,
             const Corner *corner)
{
  if (network_->isHierarchical(inst)) {
    PowerResult result;
    powerInside(inst, corner, result);
    return result;
  }
  LibertyCell *cell = network_->libertyCell(inst);
  if (cell) {
    ensureActivities();
    return power(inst, cell, corner);
  }
  return PowerResult();
}

void
Power::powerInside(const Instance *hinst,
                   const Corner *corner,
                   PowerResult &result)
{
  InstanceChildIterator *child_iter = network_->childIterator(hinst);
  while (child_iter->hasNext()) {
    Instance *child = child_iter->next();
    if (network_->isHierarchical(child))
      powerInside(child, corner, result);
    else {
      LibertyCell *cell = network_->libertyCell(child);
      if (cell) {
        PowerResult inst_power = power(child, cell, corner);
        result.incr(inst_power);
      }
    }
  }
  delete child_iter;
}

////////////////////////////////////////////////////////////////

class ActivitySrchPred : public SearchPredNonLatch2
{
public:
  explicit ActivitySrchPred(const StaState *sta);
  virtual bool searchThru(Edge *edge);
};

ActivitySrchPred::ActivitySrchPred(const StaState *sta) :
  SearchPredNonLatch2(sta)
{
}

bool
ActivitySrchPred::searchThru(Edge *edge)
{
  TimingRole *role = edge->role();
  return SearchPredNonLatch2::searchThru(edge) && role != TimingRole::regClkToQ();
}

////////////////////////////////////////////////////////////////

class PropActivityVisitor : public VertexVisitor, StaState
{
public:
  PropActivityVisitor(Power *power,
                      BfsFwdIterator *bfs);
  virtual VertexVisitor *copy() const;
  virtual void visit(Vertex *vertex);
  InstanceSet &visitedRegs() { return visited_regs_; }
  void init();
  float maxChange() const { return max_change_; }

private:
  bool setActivityCheck(const Pin *pin,
                        PwrActivity &activity);

  static constexpr float change_tolerance_ = .001;
  InstanceSet visited_regs_;
  float max_change_;
  Power *power_;
  BfsFwdIterator *bfs_;
};

PropActivityVisitor::PropActivityVisitor(Power *power,
                                         BfsFwdIterator *bfs) :
  StaState(power),
  visited_regs_(network_),
  max_change_(0.0),
  power_(power),
  bfs_(bfs)
{
}

VertexVisitor *
PropActivityVisitor::copy() const
{
  return new PropActivityVisitor(power_, bfs_);
}

void
PropActivityVisitor::init()
{
  max_change_ = 0.0;
}

void
PropActivityVisitor::visit(Vertex *vertex)
{
  Pin *pin = vertex->pin();
  Instance *inst = network_->instance(pin);
  debugPrint(debug_, "power_activity", 3, "visit %s", vertex->name(network_));
  bool changed = false;
  if (power_->hasUserActivity(pin)) {  // TODO add assert all user annotated and not changed
    PwrActivity &activity = power_->userActivity(pin);
    changed = setActivityCheck(pin, activity);
  }
  else {
    if (network_->isLoad(pin)) {
      VertexInEdgeIterator edge_iter(vertex, graph_);
      if (edge_iter.hasNext()) {  // get activity from its from pin
        Edge *edge = edge_iter.next();
        if (edge->isWire()) {
          Vertex *from_vertex = edge->from(graph_);
          const Pin *from_pin = from_vertex->pin();
          PwrActivity &from_activity = power_->activity(from_pin);
          PwrActivity to_activity(from_activity.activity(),
                                  from_activity.duty(),
                                  PwrActivityOrigin::propagated,
                                  from_activity.vcdValues(),
                                  from_activity.nEvent(),
                                  from_activity.valueBit());
          changed = setActivityCheck(pin, to_activity);
        }
      }
    }
    if (network_->isDriver(pin)) {
      LibertyPort *port = network_->libertyPort(pin);
      if (port) {
        FuncExpr *func = port->function();
        if (func) {
          PwrActivity activity = power_->evalActivity(func, inst);
          changed = setActivityCheck(pin, activity);
        }
        if (port->isClockGateOut()) {
          const Pin *enable, *clk, *gclk;
          power_->clockGatePins(inst, enable, clk, gclk);
          if (enable && clk && gclk) {
            PwrActivity activity1 = power_->findActivity(clk);
            PwrActivity activity2 = power_->findActivity(enable);
            float p1 = activity1.duty();
            float p2 = activity2.duty();
            PwrActivity activity(activity1.activity() * p2 + activity2.activity() * p1,
                                 p1 * p2,
                                 PwrActivityOrigin::propagated);
            changed = setActivityCheck(gclk, activity);
            debugPrint(debug_, "power_activity", 3, "gated_clk %s %.2e %.2f", network_->pathName(gclk), activity.activity(), activity.duty());
          }
        }
      }
    }
  }
  if (changed) {
    LibertyCell *cell = network_->libertyCell(inst);
    if (network_->isLoad(pin) && cell) {
      if (cell->hasSequentials()) {
        debugPrint(debug_, "power_activity", 3, "pending seq %s", network_->pathName(inst));
        visited_regs_.insert(inst);
      }
      // Gated clock cells latch the enable so there is no EN->GCLK timing arc.
      if (cell->isClockGate()) {
        const Pin *enable, *clk, *gclk;
        power_->clockGatePins(inst, enable, clk, gclk);
        if (gclk) {
          Vertex *gclk_vertex = graph_->pinDrvrVertex(gclk);
          bfs_->enqueue(gclk_vertex);
        }
      }
    }
    bfs_->enqueueAdjacentVertices(vertex);
  }
}

// Return true if the activity changed.
bool
PropActivityVisitor::setActivityCheck(const Pin *pin,
                                      PwrActivity &activity)
{
  PwrActivity &prev_activity = power_->activity(pin);
  float activity_delta = abs(activity.activity() - prev_activity.activity());
  float duty_delta = abs(activity.duty() - prev_activity.duty());
  if (activity_delta > change_tolerance_ || duty_delta > change_tolerance_ || activity.origin() != prev_activity.origin()) {
    max_change_ = max(max_change_, activity_delta);
    max_change_ = max(max_change_, duty_delta);
    power_->setActivity(pin, activity);
    return true;
  }
  else
    return false;
}

void
Power::clockGatePins(const Instance *inst,
                     // Return values.
                     const Pin *&enable,
                     const Pin *&clk,
                     const Pin *&gclk) const
{
  enable = nullptr;
  clk = nullptr;
  gclk = nullptr;
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {
    const Pin *pin = pin_iter->next();
    const LibertyPort *port = network_->libertyPort(pin);
    if (port->isClockGateEnable())
      enable = pin;
    if (port->isClockGateClock())
      clk = pin;
    if (port->isClockGateOut())
      gclk = pin;
  }
  delete pin_iter;
}

////////////////////////////////////////////////////////////////

PwrActivity
Power::evalActivity(FuncExpr *expr,
                    const Instance *inst)
{
  LibertyPort *func_port = expr->port();
  if (func_port && func_port->direction()->isInternal())
    return findSeqActivity(inst, func_port);  // related to register or something
  else {
    DdNode *bdd = bdd_.funcBdd(expr);
    float duty = evalBddDuty(bdd, inst);
    float activity = evalBddActivity(bdd, inst);

    Cudd_RecursiveDeref(bdd_.cuddMgr(), bdd);
    bdd_.clearVarMap();
    return PwrActivity(activity, duty, PwrActivityOrigin::propagated);
  }
}

// Find duty when from_port is sensitized.
float
Power::evalDiffDuty(FuncExpr *expr,
                    LibertyPort *from_port,
                    const Instance *inst)
{
  DdNode *bdd = bdd_.funcBdd(expr);
  DdNode *var_node = bdd_.findNode(from_port);
  unsigned var_index = Cudd_NodeReadIndex(var_node);
  DdNode *diff = Cudd_bddBooleanDiff(bdd_.cuddMgr(), bdd, var_index);
  Cudd_Ref(diff);
  float duty = evalBddDuty(diff, inst);

  Cudd_RecursiveDeref(bdd_.cuddMgr(), diff);
  Cudd_RecursiveDeref(bdd_.cuddMgr(), bdd);
  bdd_.clearVarMap();
  return duty;
}

// As suggested by
// https://stackoverflow.com/questions/63326728/cudd-printminterm-accessing-the-individual-minterms-in-the-sum-of-products
float
Power::evalBddDuty(DdNode *bdd,
                   const Instance *inst)
{
  if (Cudd_IsConstant(bdd)) {
    if (bdd == Cudd_ReadOne(bdd_.cuddMgr()))
      return 1.0;
    else if (bdd == Cudd_ReadLogicZero(bdd_.cuddMgr()))
      return 0.0;
    else
      criticalError(1100, "unknown cudd constant");
  }
  else {
    float duty0 = evalBddDuty(Cudd_E(bdd), inst);  // duty of the else branch
    float duty1 = evalBddDuty(Cudd_T(bdd), inst);  // duty of the then branch
    unsigned int index = Cudd_NodeReadIndex(bdd);
    int var_index = Cudd_ReadPerm(bdd_.cuddMgr(), index);
    const LibertyPort *port = bdd_.varIndexPort(var_index);
    if (port->direction()->isInternal())
      return findSeqActivity(inst, const_cast<LibertyPort *>(port)).duty();
    else {
      const Pin *pin = findLinkPin(inst, port);
      if (pin) {
        PwrActivity var_activity = findActivity(pin);
        float var_duty = var_activity.duty();
        float duty = duty0 * (1.0 - var_duty) + duty1 * var_duty;  // duty is the static probability which is calculated in ReadVcdActivities::findVarActivity
        if (Cudd_IsComplement(bdd))
          duty = 1.0 - duty;
        return duty;
      }
    }
  }
  return 0.0;
}

// https://www.brown.edu/Departments/Engineering/Courses/engn2912/Lectures/LP-02-logic-power-est.pdf
// F(x0, x1, .. ) is sensitized when F(Xi=1) xor F(Xi=0)
// F(Xi=1), F(Xi=0) are the cofactors of F wrt Xi.
float
Power::evalBddActivity(DdNode *bdd,
                       const Instance *inst)
{
  float activity = 0.0;
  for (const auto [port, var_node] : bdd_.portVarMap()) {
    const Pin *pin = findLinkPin(inst, port);
    if (pin) {
      PwrActivity var_activity = findActivity(pin);
      unsigned int var_index = Cudd_NodeReadIndex(var_node);
      DdNode *diff = Cudd_bddBooleanDiff(bdd_.cuddMgr(), bdd, var_index);
      Cudd_Ref(diff);
      float diff_duty = evalBddDuty(diff, inst);
      Cudd_RecursiveDeref(bdd_.cuddMgr(), diff);
      float var_act = var_activity.activity() * diff_duty;
      activity += var_act;
      const Clock *clk = findClk(pin);
      float clk_period = clk ? clk->period() : 1.0;
      debugPrint(debug_, "power_activity", 3, "var %s %.3e * %.3f = %.3e", port->name(), var_activity.activity() / clk_period, diff_duty, var_act / clk_period);
    }
  }
  return activity;
}

////////////////////////////////////////////////////////////////

void
Power::ensureActivities()
{
  utils::ScopedTimer ensure_act_timer("Ensure Activities");
  TIMERSTART(ENSURE_ACTIVITIES);
  // No need to propagate activites if global activity is set.
  if (!global_activity_.isSet()) {
    if (!activities_valid_) {
      // Clear existing activities.
      activity_map_.clear();
      seq_activity_map_.clear();

      ActivitySrchPred activity_srch_pred(this);
      BfsFwdIterator bfs(BfsIndex::other, &activity_srch_pred, this);
      seedActivities(bfs);
      PropActivityVisitor visitor(this, &bfs);
      // Propagate activities through combinational logic.
      bfs.visit(levelize_->maxLevel(), &visitor);
      // Propagate activiities through registers.
      InstanceSet regs = std::move(visitor.visitedRegs());
      int pass = 1;
      while (!regs.empty() && pass < max_activity_passes_) {
        visitor.init();
        InstanceSet::Iterator reg_iter(regs);
        while (reg_iter.hasNext()) {
          const Instance *reg = reg_iter.next();
          // Propagate activiities across register D->Q.
          seedRegOutputActivities(reg, bfs);
        }
        // Propagate register output activities through
        // combinational logic.
        bfs.visit(levelize_->maxLevel(), &visitor);
        regs = std::move(visitor.visitedRegs());
        debugPrint(debug_, "power_activity", 1, "Pass %d change %.2f", pass, visitor.maxChange());
        pass++;
      }
      activities_valid_ = true;
      if (G_CONFIG.flags.report_vcd_stat)
        reportResolvedTimeBasedPowerAnalysisWorkload();
    }
  }

  TIMEREND(ENSURE_ACTIVITIES);
  DURATION_ms(ENSURE_ACTIVITIES);
}

void
Power::reportResolvedTimeBasedPowerAnalysisWorkload()
{
  utils::ScopedTimer timer_report_resolved_time_based_power_analysis_workload(
      "Report Resolved Time-Based Power Analysis Workload");

  size_t n_waveform_pin = 0;
  size_t n_non_clock_waveform_pin = 0;
  const auto& time_intervals = vcd_.timeIntervals();
  std::vector<double> interval_n_cycles(time_intervals.size(), 0.0);
  std::vector<double> interval_waveform_n_event(time_intervals.size(), 0.0);
  std::vector<double> non_clock_interval_waveform_n_event(time_intervals.size(), 0.0);
  struct WaveformActivityUse {
    NEeventVal n_event = 0;
    size_t n_pin = 0;
    size_t n_non_clock_pin = 0;
  };
  std::unordered_map<const VcdValue *, std::unordered_map<int, WaveformActivityUse>> waveform_activity_uses;

  auto n_cycle_in_range = [this](VcdEventTime start_time,
                                 VcdEventTime end_time) {
    if (clk_period_ <= 0.0 || end_time <= start_time)
      return 0.0;

    VcdEventTime clamped_start = start_time < 0 ? 0 : start_time;
    VcdEventTime clamped_end = end_time < 0 ? 0 : end_time;
    const VcdEventTime time_max = vcd_.timeMax();
    clamped_start = std::min(clamped_start, time_max);
    clamped_end = std::min(clamped_end, time_max);
    if (clamped_end <= clamped_start)
      return 0.0;

    return (clamped_end - clamped_start) * vcd_.timeScale() / clk_period_;
  };

  for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
    const auto& interval = time_intervals.at(interval_idx);
    interval_n_cycles.at(interval_idx) = n_cycle_in_range(interval.first, interval.second);
  }

  auto event_index_at_or_after = [](const VcdValue *vcd_values,
                                    NEeventVal n_event,
                                    VcdEventTime time,
                                    NEeventVal left) -> NEeventVal {
    if (vcd_values == nullptr || left >= n_event)
      return n_event;

    const VcdValue *event_itr = std::lower_bound(vcd_values + left, vcd_values + n_event, time,
                                                 [](const VcdValue &value,
                                                    VcdEventTime time) {
                                                   return value.time() < time;
                                                 });
    return static_cast<NEeventVal>(event_itr - vcd_values);
  };

  LeafInstanceIterator *inst_iter = network_->leafInstanceIterator();
  while (inst_iter->hasNext()) {
    const Instance *inst = inst_iter->next();
    LibertyCell *cell = network_->libertyCell(inst);
    if (!cell)
      continue;

    std::vector<const Pin *> pins;
    NPinVal inst_n_pin = 0;
    NPinVal inst_n_input_pin = 0;
    NPinVal inst_n_output_pin = 0;
    getPinInformation(inst, &pins, &inst_n_pin, &inst_n_input_pin, &inst_n_output_pin);
    for (const Pin *pin : pins) {
      LibertyPort *port = network_->libertyPort(pin);
      if (!port)
        continue;

      PwrActivity activity = findActivity(pin);
      const bool is_non_clock = !port->isClock();
      if (activity.vcdValues()) {
        n_waveform_pin++;
        if (is_non_clock)
          n_non_clock_waveform_pin++;
        WaveformActivityUse &use = waveform_activity_uses[activity.vcdValues()][activity.valueBit()];
        use.n_event = activity.nEvent();
        use.n_pin++;
        if (is_non_clock)
          use.n_non_clock_pin++;
      }
    }
  }
  delete inst_iter;

  double n_cycle = 0.0;
  if (clk_period_ > 0.0)
    n_cycle = vcd_.timeMax() * vcd_.timeScale() / clk_period_;

  double workload_n_cycle = n_cycle;
  if (!time_intervals.empty()) {
    workload_n_cycle = 0.0;
    for (double interval_n_cycle : interval_n_cycles)
      workload_n_cycle += interval_n_cycle;
  }

  double waveform_n_event = 0.0;
  double non_clock_waveform_n_event = 0.0;
  for (const auto& [vcd_values, value_bit_to_use] : waveform_activity_uses) {
    for (const auto& value_bit_use : value_bit_to_use) {
      const WaveformActivityUse &use = value_bit_use.second;
      if (time_intervals.empty()) {
        const double n_event = use.n_event;
        waveform_n_event += n_event * use.n_pin;
        non_clock_waveform_n_event += n_event * use.n_non_clock_pin;
      }

      // Match CUDA event buffer construction: non-first intervals carry one
      // previous event record for the starting state.
      NEeventVal prev_right_bound = 0;
      for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
        const auto& interval = time_intervals.at(interval_idx);
        const bool is_first_time_interval = interval_idx == 0;
        const NEeventVal start_idx = is_first_time_interval
            ? 0
            : (prev_right_bound > 0 ? prev_right_bound - 1 : 0);
        const NEeventVal end_idx = interval.second >= vcd_.timeMax()
            ? use.n_event
            : event_index_at_or_after(vcd_values,
                                      use.n_event,
                                      interval.second,
                                      is_first_time_interval ? 0 : prev_right_bound);
        const double interval_n_event = end_idx >= start_idx ? end_idx - start_idx : 0;
        prev_right_bound = end_idx;
        interval_waveform_n_event.at(interval_idx) += interval_n_event * use.n_pin;
        non_clock_interval_waveform_n_event.at(interval_idx) += interval_n_event * use.n_non_clock_pin;
        waveform_n_event += interval_n_event * use.n_pin;
        non_clock_waveform_n_event += interval_n_event * use.n_non_clock_pin;
      }
    }
  }

  double time_based_workload_activity_factor = 0.0;
  if (n_waveform_pin != 0 && workload_n_cycle > 0.0)
    time_based_workload_activity_factor = waveform_n_event / (n_waveform_pin * workload_n_cycle);

  double non_clock_time_based_workload_activity_factor = 0.0;
  if (n_non_clock_waveform_pin != 0 && workload_n_cycle > 0.0)
    non_clock_time_based_workload_activity_factor =
        non_clock_waveform_n_event / (n_non_clock_waveform_pin * workload_n_cycle);

  LOG_BEGIN(INFO, "Resolved time-based power analysis workload activity");
  LOG_INFO << "Resolved time-based power analysis workload activity N_pin: " << n_waveform_pin;
  LOG_INFO << "Resolved time-based power analysis workload activity N_event: " << waveform_n_event;
  LOG_INFO << "Resolved time-based power analysis workload activity N_cycle: " << workload_n_cycle;
  LOG_INFO << "Resolved time-based power analysis workload activity factor: "
           << time_based_workload_activity_factor;
  LOG_INFO << "Resolved time-based power analysis workload activity non-clock N_pin: "
           << n_non_clock_waveform_pin;
  LOG_INFO << "Resolved time-based power analysis workload activity non-clock N_event: "
           << non_clock_waveform_n_event;
  LOG_INFO << "Resolved time-based power analysis workload activity non-clock N_cycle: "
           << workload_n_cycle;
  LOG_INFO << "Resolved time-based power analysis workload activity non-clock activity factor: "
           << non_clock_time_based_workload_activity_factor;
  LOG_END(INFO, "Resolved time-based power analysis workload activity");

  if (!time_intervals.empty()) {
    LOG_BEGIN(INFO, "Resolved time-based power analysis workload activity by interval");
    for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
      const auto& interval = time_intervals.at(interval_idx);
      const double interval_n_cycle = interval_n_cycles.at(interval_idx);
      const double interval_n_event = interval_waveform_n_event.at(interval_idx);
      double interval_activity_factor = 0.0;
      if (n_waveform_pin != 0 && interval_n_cycle > 0.0)
        interval_activity_factor = interval_n_event / (n_waveform_pin * interval_n_cycle);
      LOG_INFO << "Resolved time-based power analysis workload activity interval " << interval_idx
               << " start time: " << interval.first
               << " end time: " << interval.second
               << " N_pin: " << n_waveform_pin
               << " N_event: " << interval_n_event
               << " N_cycle: " << interval_n_cycle
               << " activity factor: " << interval_activity_factor;
    }
    LOG_END(INFO, "Resolved time-based power analysis workload activity by interval");

    LOG_BEGIN(INFO, "Resolved time-based power analysis workload non-clock activity by interval");
    for (size_t interval_idx = 0; interval_idx < time_intervals.size(); ++interval_idx) {
      const auto& interval = time_intervals.at(interval_idx);
      const double interval_n_cycle = interval_n_cycles.at(interval_idx);
      const double interval_n_event = non_clock_interval_waveform_n_event.at(interval_idx);
      double interval_activity_factor = 0.0;
      if (n_non_clock_waveform_pin != 0 && interval_n_cycle > 0.0)
        interval_activity_factor = interval_n_event / (n_non_clock_waveform_pin * interval_n_cycle);
      LOG_INFO << "Resolved time-based power analysis workload non-clock activity interval " << interval_idx
               << " start time: " << interval.first
               << " end time: " << interval.second
               << " N_pin: " << n_non_clock_waveform_pin
               << " N_event: " << interval_n_event
               << " N_cycle: " << interval_n_cycle
               << " activity factor: " << interval_activity_factor;
    }
    LOG_END(INFO, "Resolved time-based power analysis workload non-clock activity by interval");
  }
}

void
Power::seedActivities(BfsFwdIterator &bfs)
{
  for (Vertex *vertex : *levelize_->roots()) {
    const Pin *pin = vertex->pin();
    // We also set clock activity here, since the vcd events given by user will be used to evaluate power
    if (!network_->direction(pin)->isInternal()) {
      debugPrint(debug_, "power_activity", 3, "seed %s", vertex->name(network_));
      if (hasUserActivity(pin))
        setActivity(pin, userActivity(pin));
      else
        // Default inputs without explicit activities to the input default.
        setActivity(pin, input_activity_);
      Vertex *vertex = graph_->pinDrvrVertex(pin);
      bfs.enqueueAdjacentVertices(vertex);
    }
  }
}

void
Power::seedRegOutputActivities(const Instance *inst,
                               BfsFwdIterator &bfs)
{
  LibertyCell *cell = network_->libertyCell(inst);
  for (Sequential *seq : cell->sequentials()) {
    seedRegOutputActivities(inst, seq, seq->output(), false);
    seedRegOutputActivities(inst, seq, seq->outputInv(), true);
    // Enqueue register output pins with functions that reference
    // the sequential internal pins (IQ, IQN).
    InstancePinIterator *pin_iter = network_->pinIterator(inst);
    while (pin_iter->hasNext()) {
      Pin *pin = pin_iter->next();
      LibertyPort *port = network_->libertyPort(pin);
      if (port) {
        FuncExpr *func = port->function();
        Vertex *vertex = graph_->pinDrvrVertex(pin);
        if (vertex && func && (func->port() == seq->output() || func->port() == seq->outputInv())) {
          debugPrint(debug_, "power_reg", 1, "enqueue reg output %s", vertex->name(network_));
          bfs.enqueue(vertex);
        }
      }
    }
    delete pin_iter;
  }
}

void
Power::seedRegOutputActivities(const Instance *reg,
                               Sequential *seq,
                               LibertyPort *output,
                               bool invert)
{
  const Pin *out_pin = network_->findPin(reg, output);
  if (!hasUserActivity(out_pin)) {
    PwrActivity activity = evalActivity(seq->data(), reg);
    // Register output activity cannnot exceed one transition per clock cycle,
    // but latch output can.
    if (seq->isRegister() && activity.activity() > 1.0)
      activity.setActivity(1.0);
    if (invert)
      activity.setDuty(1.0 - activity.duty());
    activity.setOrigin(PwrActivityOrigin::propagated);
    setSeqActivity(reg, output, activity);
  }
}

////////////////////////////////////////////////////////////////

PowerResult
Power::power(const Instance *inst,
             LibertyCell *cell,
             const Corner *corner)
{
  PowerResult result;
  const Clock *inst_clk = findInstClk(inst);
  findInternalPower(inst, cell, corner, inst_clk, result);
  findSwitchingPower(inst, cell, corner, inst_clk, result);
  findLeakagePower(inst, cell, corner, result);
  return result;
}

const Clock *
Power::findInstClk(const Instance *inst)
{
  const Clock *inst_clk = nullptr;
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {
    const Pin *pin = pin_iter->next();
    const Clock *clk = findClk(pin);
    if (clk) {
      inst_clk = clk;
      break;
    }
  }
  delete pin_iter;
  return inst_clk;
}

void
Power::findInternalPower(const Instance *inst,
                         LibertyCell *cell,
                         const Corner *corner,
                         const Clock *inst_clk,
                         // Return values.
                         PowerResult &result)
{
  const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {
    const Pin *to_pin = pin_iter->next();
    LibertyPort *to_port = network_->libertyPort(to_pin);
    if (to_port) {
      float load_cap = to_port->direction()->isAnyOutput()
          ? graph_delay_calc_->loadCap(to_pin, dcalc_ap)
          : 0.0;
      PwrActivity activity = findClkedActivity(to_pin, inst_clk);
      if (to_port->direction()->isAnyOutput())
        findOutputInternalPower(to_port, inst, cell, activity, load_cap, corner, result);
      if (to_port->direction()->isAnyInput())
        findInputInternalPower(to_pin, to_port, inst, cell, activity, load_cap, corner, result);
    }
  }
  delete pin_iter;
}

void
Power::findInputInternalPower(const Pin *pin,
                              LibertyPort *port,
                              const Instance *inst,
                              LibertyCell *cell,
                              PwrActivity &activity,
                              float load_cap,
                              const Corner *corner,
                              // Return values.
                              PowerResult &result)
{
  const MinMax *min_max = MinMax::max();
  LibertyCell *corner_cell = cell->cornerCell(corner, min_max);
  const LibertyPort *corner_port = port->cornerPort(corner, min_max);
  if (corner_cell && corner_port) {
    const InternalPowerSeq &internal_pwrs = corner_cell->internalPowers(corner_port);
    if (!internal_pwrs.empty()) {
      debugPrint(debug_, "power", 2, "internal input %s/%s cap %s", network_->pathName(inst), port->name(), units_->capacitanceUnit()->asString(load_cap));
      debugPrint(debug_, "power", 2, "       when  act/ns duty  energy    power");
      const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
      const Pvt *pvt = dcalc_ap->operatingConditions();
      Vertex *vertex = graph_->pinLoadVertex(pin);
      float internal = 0.0;
      for (InternalPower *pwr : internal_pwrs) {
        const char *related_pg_pin = pwr->relatedPgPin();
        float energy = 0.0;
        int rf_count = 0;
        for (RiseFall *rf : RiseFall::range()) {
          float slew = getSlew(vertex, rf, corner);
          if (!delayInf(slew)) {
            float table_energy = pwr->power(rf, pvt, slew, load_cap);  // the energy is decided by the rf and slew of current pin
            energy += table_energy;
            rf_count++;
          }
        }
        if (rf_count)
          energy /= rf_count;  // average non-inf energies
        float duty = 1.0;      // fallback default
        FuncExpr *when = pwr->when();
        if (when) {
          const LibertyPort *out_corner_port = findExprOutPort(when);
          if (out_corner_port) {
            LibertyPort *out_port = findLinkPort(cell, out_corner_port);
            if (out_port) {
              FuncExpr *func = out_port->function();
              if (func && func->hasPort(port))
                duty = evalDiffDuty(func, port, inst);
              else
                duty = evalActivity(when, inst).duty();
            }
          }
          else
            duty = evalActivity(when, inst).duty();
        }
        float port_internal = energy * duty * activity.activity();
        debugPrint(debug_, "power", 2, " %3s %6s  %.2f  %.2f %9.2e %9.2e %s", port->name(), when ? when->asString() : "", activity.activity() * 1e-9, duty, energy, port_internal, related_pg_pin ? related_pg_pin : "no pg_pin");
        internal += port_internal;
      }
      result.internal() += internal;
    }
  }
}

// TODO add query slew by dcalc_ap?
float
Power::getSlew(Vertex *vertex,
               const RiseFall *rf,
               const Corner *corner)
{
  const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
  const Pin *pin = vertex->pin();
  if (clk_network_->isIdealClock(pin))
    return clk_network_->idealClkSlew(pin, rf, MinMax::max());
  else
    return delayAsFloat(graph_->slew(vertex, rf, dcalc_ap->index()));
}

LibertyPort *
Power::findExprOutPort(FuncExpr *expr)
{
  LibertyPort *port;
  switch (expr->op()) {
    case FuncExpr::op_port:
      port = expr->port();
      if (port && port->direction()->isAnyOutput())
        return expr->port();
      return nullptr;
    case FuncExpr::op_not:
      port = findExprOutPort(expr->left());
      if (port)
        return port;
      return nullptr;
    case FuncExpr::op_or:
    case FuncExpr::op_and:
    case FuncExpr::op_xor:
      port = findExprOutPort(expr->left());
      if (port)
        return port;
      port = findExprOutPort(expr->right());
      if (port)
        return port;
      return nullptr;
    case FuncExpr::op_one:
    case FuncExpr::op_zero:
      return nullptr;
  }
  return nullptr;
}

void
Power::findOutputInternalPower(const LibertyPort *to_port,
                               const Instance *inst,
                               LibertyCell *cell,
                               PwrActivity &to_activity,
                               float load_cap,
                               const Corner *corner,
                               // Return values.
                               PowerResult &result)
{
  debugPrint(debug_, "power", 2, "internal output %s/%s cap %s", network_->pathName(inst), to_port->name(), units_->capacitanceUnit()->asString(load_cap));
  const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
  const Pvt *pvt = dcalc_ap->operatingConditions();
  LibertyCell *corner_cell = cell->cornerCell(dcalc_ap);
  const LibertyPort *to_corner_port = to_port->cornerPort(dcalc_ap);
  FuncExpr *func = to_port->function();

  map<const char *, float, StringLessIf> pg_duty_sum;
  for (InternalPower *pwr : corner_cell->internalPowers(to_corner_port)) {
    const LibertyPort *from_corner_port = pwr->relatedPort();
    if (from_corner_port) {
      const Pin *from_pin = findLinkPin(inst, from_corner_port);
      float from_activity = findActivity(from_pin).activity();
      float duty = findInputDuty(inst, func, pwr);
      const char *related_pg_pin = pwr->relatedPgPin();
      // Note related_pg_pin may be null.
      pg_duty_sum[related_pg_pin] += from_activity * duty;
    }
  }

  debugPrint(debug_, "power", 2, "             when act/ns  duty  wgt   energy    power");
  float internal = 0.0;
  for (InternalPower *pwr : corner_cell->internalPowers(to_corner_port)) {
    FuncExpr *when = pwr->when();
    const char *related_pg_pin = pwr->relatedPgPin();
    float duty = findInputDuty(inst, func, pwr);
    Vertex *from_vertex = nullptr;
    bool positive_unate = true;
    const LibertyPort *from_corner_port = pwr->relatedPort();
    const Pin *from_pin = nullptr;
    if (from_corner_port) {
      positive_unate = isPositiveUnate(corner_cell, from_corner_port, to_corner_port);
      from_pin = findLinkPin(inst, from_corner_port);
      if (from_pin)
        from_vertex = graph_->pinLoadVertex(from_pin);
    }
    float energy = 0.0;
    int rf_count = 0;
    for (RiseFall *to_rf : RiseFall::range()) {
      // Use unateness to find from_rf.
      RiseFall *from_rf = positive_unate ? to_rf : to_rf->opposite();
      float slew = from_vertex
          ? getSlew(from_vertex, from_rf, corner)
          : 0.0;
      if (!delayInf(slew)) {
        float table_energy = pwr->power(to_rf, pvt, slew, load_cap);  // the energy is decided by the to_rf, input slew and load_cap, the to_rf is used to decide the rise/fall lut
        energy += table_energy;
        rf_count++;
      }
    }
    if (rf_count)
      energy /= rf_count;  // average non-inf energies
    auto duty_sum_iter = pg_duty_sum.find(related_pg_pin);
    float weight = 0.0;
    if (duty_sum_iter != pg_duty_sum.end()) {
      float duty_sum = duty_sum_iter->second;
      if (duty_sum != 0.0 && from_pin) {
        float from_activity = findActivity(from_pin).activity();
        weight = from_activity * duty / duty_sum;
      }
    }
    float port_internal = weight * energy * to_activity.activity();
    debugPrint(debug_, "power", 2, "%3s -> %-3s %6s  %.3f %.3f %.3f %9.2e %9.2e %s", from_corner_port ? from_corner_port->name() : "-", to_port->name(), when ? when->asString() : "", to_activity.activity() * 1e-9, duty, weight, energy, port_internal, related_pg_pin ? related_pg_pin : "no pg_pin");
    internal += port_internal;
  }
  result.internal() += internal;
}

float
Power::findInputDuty(const Instance *inst,
                     FuncExpr *func,
                     InternalPower *pwr)

{
  const LibertyPort *from_corner_port = pwr->relatedPort();
  if (from_corner_port) {
    LibertyPort *from_port = findLinkPort(network_->libertyCell(inst),
                                          from_corner_port);
    const Pin *from_pin = network_->findPin(inst, from_port);
    if (from_pin) {
      FuncExpr *when = pwr->when();
      Vertex *from_vertex = graph_->pinLoadVertex(from_pin);
      if (func && func->hasPort(from_port)) {
        float duty = evalDiffDuty(func, from_port, inst);
        return duty;
      }
      else if (when)
        return evalActivity(when, inst).duty();
      else if (search_->isClock(from_vertex))
        return 1.0;
      return 0.5;
    }
  }
  return 0.0;
}

// Hack to find cell port that corresponds to corner_port.
LibertyPort *
Power::findLinkPort(const LibertyCell *cell,
                    const LibertyPort *corner_port)
{
  return cell->findLibertyPort(corner_port->name());
}

Pin *
Power::findLinkPin(const Instance *inst,
                   const LibertyPort *corner_port)
{
  const LibertyCell *cell = network_->libertyCell(inst);
  LibertyPort *port = findLinkPort(cell, corner_port);
  return network_->findPin(inst, port);
}

static bool
isPositiveUnate(const LibertyCell *cell,
                const LibertyPort *from,
                const LibertyPort *to)
{
  const TimingArcSetSeq &arc_sets = cell->timingArcSets(from, to);
  if (!arc_sets.empty()) {
    TimingSense sense = arc_sets[0]->sense();
    return sense == TimingSense::positive_unate || sense == TimingSense::non_unate;
  }
  // default
  return true;
}

////////////////////////////////////////////////////////////////

void
Power::findSwitchingPower(const Instance *inst,
                          LibertyCell *cell,
                          const Corner *corner,
                          const Clock *inst_clk,
                          // Return values.
                          PowerResult &result)
{
  const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
  LibertyCell *corner_cell = cell->cornerCell(dcalc_ap);
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {
    const Pin *to_pin = pin_iter->next();
    const LibertyPort *to_port = network_->libertyPort(to_pin);
    if (to_port) {
      float load_cap = to_port->direction()->isAnyOutput()
          ? graph_delay_calc_->loadCap(to_pin, dcalc_ap)
          : 0.0;
      PwrActivity activity = findClkedActivity(to_pin, inst_clk);
      if (to_port->direction()->isAnyOutput()) {
        float volt = portVoltage(corner_cell, to_port, dcalc_ap);
        float switching = .5 * load_cap * volt * volt * activity.activity();
        debugPrint(debug_, "power", 2, "switching %s/%s activity = %.2e volt = %.2f %.3e", cell->name(), to_port->name(), activity.activity(), volt, switching);
        result.switching() += switching;
      }
    }
  }
  delete pin_iter;
}

////////////////////////////////////////////////////////////////

void
Power::findLeakagePower(const Instance *inst,
                        LibertyCell *cell,
                        const Corner *corner,
                        // Return values.
                        PowerResult &result)
{
  LibertyCell *corner_cell = cell->cornerCell(corner, MinMax::max());
  float cond_leakage = 0.0;
  bool found_cond = false;
  float uncond_leakage = 0.0;
  bool found_uncond = false;
  float cond_duty_sum = 0.0;
  for (LeakagePower *leak : *corner_cell->leakagePowers()) {
    FuncExpr *when = leak->when();
    if (when) {
      PwrActivity cond_activity = evalActivity(when, inst);
      float cond_duty = cond_activity.duty();
      debugPrint(debug_, "power", 2, "leakage %s %s %.3e * %.2f", cell->name(), when->asString(), leak->power(), cond_duty);
      cond_leakage += leak->power() * cond_duty;
      if (leak->power() > 0.0)
        cond_duty_sum += cond_duty;
      found_cond = true;
    }
    else {  // no when condition, which means it will definitely happen, so just add it to the leakage power
      debugPrint(debug_, "power", 2, "leakage -- %s %.3e", cell->name(), leak->power());
      uncond_leakage += leak->power();
      found_uncond = true;
    }
  }
  float leakage = 0.0;
  float cell_leakage;
  bool cell_leakage_exists;
  cell->leakagePower(cell_leakage, cell_leakage_exists);
  if (cell_leakage_exists) {  // default condition, which means the duty sum of condition when is less than 1.0
    float duty = 1.0 - cond_duty_sum;
    debugPrint(debug_, "power", 2, "leakage cell %s %.3e * %.2f", cell->name(), cell_leakage, duty);
    cell_leakage *= duty;
  }
  // Ignore unconditional leakage unless there are no conditional leakage groups.
  if (found_cond)
    leakage = cond_leakage;
  else if (found_uncond)
    leakage = uncond_leakage;
  if (cell_leakage_exists)
    leakage += cell_leakage;
  debugPrint(debug_, "power", 2, "leakage %s %.3e", cell->name(), leakage);
  result.leakage() += leakage;
}

PwrActivity
Power::findClkedActivity(const Pin *pin)
{
  const Instance *inst = network_->instance(pin);
  const Clock *inst_clk = findInstClk(inst);
  ensureActivities();
  return findClkedActivity(pin, inst_clk);
}

PwrActivity
Power::findClkedActivity(const Pin *pin,
                         const Clock *inst_clk)
{
  PwrActivity activity = findActivity(pin);
  const Clock *clk = findClk(pin);
  if (clk == nullptr)
    clk = inst_clk;
  if (clk) {
    float period = clk->period();
    if (period > 0.0)
      return PwrActivity(activity.activity() / period,
                         activity.duty(),
                         activity.origin(),
                         activity.vcdValues(),
                         activity.nEvent(),
                         activity.valueBit());
  }
  return activity;
}

PwrActivity
Power::findActivity(const Pin *pin)
{
  Vertex *vertex = graph_->pinLoadVertex(pin);
  if (vertex && vertex->isConstant())
    return PwrActivity(0.0, 0.0, PwrActivityOrigin::constant);
  else if (vertex && search_->isClock(vertex)) {
    if (activity_map_.hasKey(pin)) {
      PwrActivity &activity = activity_map_[pin];
      if (activity.origin() != PwrActivityOrigin::unknown)
        return activity;
    } else if (hasUserActivity(pin)) {
      if (LOG_DEBUG_FLAG) {
        LOG_DEBUG << "No estimated activity found for pin " << network_->pathName(pin) << ". Use user annotated activity instead.";
      }
      PwrActivity &activity = user_activity_map_.at(pin);
      if (activity.origin() != PwrActivityOrigin::unknown)
        return activity;
    }
    const Clock *clk = findClk(pin);
    float duty = clockDuty(clk);
    return PwrActivity(2.0, duty, PwrActivityOrigin::clock);
  }
  else if (global_activity_.isSet())
    return global_activity_;
  else if (activity_map_.hasKey(pin)) {
    PwrActivity &activity = activity_map_[pin];
    if (activity.origin() != PwrActivityOrigin::unknown)
      return activity;
  } 
  else if (hasUserActivity(pin)) {
    if (LOG_DEBUG_FLAG) {
      LOG_DEBUG << "No estimated activity found for pin " << network_->pathName(pin) << ". Use user annotated activity instead.";
    }
    PwrActivity &activity = user_activity_map_.at(pin);
    if (activity.origin() != PwrActivityOrigin::unknown)
      return activity;
  }
  return PwrActivity(0.0, 0.0, PwrActivityOrigin::unknown);
}

float
Power::clockDuty(const Clock *clk)
{
  if (clk->isGenerated()) {
    const Clock *master = clk->masterClk();
    if (master == nullptr)
      return 0.5;  // punt
    else
      return clockDuty(master);
  }
  else {
    const FloatSeq *waveform = clk->waveform();
    float rise_time = (*waveform)[0];
    float fall_time = (*waveform)[1];
    float duty = (fall_time - rise_time) / clk->period();
    return duty;
  }
}

PwrActivity
Power::findSeqActivity(const Instance *inst,
                       LibertyPort *port)
{
  if (global_activity_.isSet())
    return global_activity_;
  else if (hasSeqActivity(inst, port)) {
    PwrActivity &activity = seqActivity(inst, port);
    if (activity.origin() != PwrActivityOrigin::unknown)
      return activity;
  }
  return PwrActivity(0.0, 0.0, PwrActivityOrigin::unknown);
}

float
Power::portVoltage(const LibertyCell *cell,
                   const LibertyPort *port,
                   const DcalcAnalysisPt *dcalc_ap) const
{
  return pgNameVoltage(cell, port->relatedPowerPin(), dcalc_ap);
}

float
Power::pgNameVoltage(const LibertyCell *cell,
                     const char *pg_port_name,
                     const DcalcAnalysisPt *dcalc_ap) const
{
  if (pg_port_name) {
    LibertyPgPort *pg_port = cell->findPgPort(pg_port_name);
    if (pg_port) {
      const char *volt_name = pg_port->voltageName();
      LibertyLibrary *library = cell->libertyLibrary();
      float voltage;
      bool exists;
      library->supplyVoltage(volt_name, voltage, exists);
      if (exists)
        return voltage;
    }
  }

  const Pvt *pvt = dcalc_ap->operatingConditions();
  if (pvt == nullptr)
    pvt = cell->libertyLibrary()->defaultOperatingConditions();
  if (pvt)
    return pvt->voltage();
  else
    return 0.0;
}

const Clock *
Power::findClk(const Pin *to_pin)
{
  const Clock *clk = nullptr;
  Vertex *to_vertex = graph_->pinDrvrVertex(to_pin);
  if (to_vertex) {
    VertexPathIterator path_iter(to_vertex, this);
    while (path_iter.hasNext()) {
      PathVertex *path = path_iter.next();
      const Clock *path_clk = path->clock(this);
      if (path_clk && (clk == nullptr || path_clk->period() < clk->period()))
        clk = path_clk;
    }
  }
  return clk;
}

////////////////////////////////////////////////////////////////
PowerClkedWaveform::PowerClkedWaveform(PeriodVal period, NPeriodVal n_period) :
  period_(period),
  waveform_(n_period, 0.0)
{
}
////////////////////////////////////////////////////////////////

PowerResult::PowerResult() :
  internal_(0.0),
  glitch_internal_(0.0),
  switching_(0.0),
  glitch_switching_(0.0),
  leakage_(0.0),
  inst_to_leakage_power_clked_waveform_map_(),
  inst_to_internal_power_clked_waveform_map_(),
  inst_to_glitch_internal_power_clked_waveform_map_(),
  inst_to_switching_power_clked_waveform_map_(),
  inst_to_glitch_switching_power_clked_waveform_map_()
{
}

void
PowerResult::clear()
{
  internal_ = 0.0;
  switching_ = 0.0;
  leakage_ = 0.0;
}

PowerVal
PowerResult::total() const
{
  return internal_ + switching_ + leakage_;
}

void
PowerResult::incr(const PowerResult &result)
{
  internal_ += result.internal_;
  glitch_internal_ += result.glitch_internal_;
  switching_ += result.switching_;
  glitch_switching_ += result.glitch_switching_;
  leakage_ += result.leakage_;

  inst_to_leakage_power_clked_waveform_map_.insert(result.inst_to_leakage_power_clked_waveform_map_.begin(), result.inst_to_leakage_power_clked_waveform_map_.end());
  inst_to_internal_power_clked_waveform_map_.insert(result.inst_to_internal_power_clked_waveform_map_.begin(), result.inst_to_internal_power_clked_waveform_map_.end());
  inst_to_glitch_internal_power_clked_waveform_map_.insert(result.inst_to_glitch_internal_power_clked_waveform_map_.begin(), result.inst_to_glitch_internal_power_clked_waveform_map_.end());
  inst_to_switching_power_clked_waveform_map_.insert(result.inst_to_switching_power_clked_waveform_map_.begin(), result.inst_to_switching_power_clked_waveform_map_.end());
  inst_to_glitch_switching_power_clked_waveform_map_.insert(result.inst_to_glitch_switching_power_clked_waveform_map_.begin(), result.inst_to_glitch_switching_power_clked_waveform_map_.end());
}

////////////////////////////////////////////////////////////////

PwrActivity::PwrActivity(float activity,
                         float duty,
                         PwrActivityOrigin origin,
                         const VcdValue* vcd_values,
                         NEeventVal n_event,
                         int value_bit) :
  activity_(activity),
  duty_(duty),
  origin_(origin),
  vcd_values_(vcd_values),
  n_event_(n_event),
  value_bit_(value_bit)
{
}

PwrActivity::PwrActivity() :
  activity_(0.0),
  duty_(0.0),
  origin_(PwrActivityOrigin::unknown),
  vcd_values_(nullptr),
  n_event_(0),
  value_bit_(-1)
{
  check();
}

void
PwrActivity::setActivity(float activity)
{
  activity_ = activity;
}

void
PwrActivity::setDuty(float duty)
{
  duty_ = duty;
}

void
PwrActivity::setOrigin(PwrActivityOrigin origin)
{
  origin_ = origin;
}

void
PwrActivity::set(float activity,
                 float duty,
                 PwrActivityOrigin origin)
{
  activity_ = activity;
  duty_ = duty;
  origin_ = origin;
  check();
}

void
PwrActivity::check()
{
  // Activities can get very small from multiplying probabilities
  // through deep chains of logic. Clip them to prevent floating
  // point anomalies.
  if (abs(activity_) < min_activity)
    activity_ = 0.0;
}

bool
PwrActivity::isSet() const
{
  return origin_ != PwrActivityOrigin::unknown;
}

const char *
PwrActivity::originName() const
{
  return pwr_activity_origin_map.find(origin_);
}

}  // namespace sta
