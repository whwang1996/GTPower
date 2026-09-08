#include <algorithm>  // std::min
#include <cassert>
#include <limits>  // std::numeric_limits
#include <cmath>  // floor, ceil
#include <map>
#include <unordered_set>

#include "Power.hh"

#include "Clock.hh"
#include "Corner.hh"
#include "Graph.hh"
#include "GraphDelayCalc.hh"
#include "InternalPower.hh"
#include "Liberty.hh"
#include "Network.hh"
#include "PortDirection.hh"
#include "StringUtil.hh"
#include "LeakagePower.hh"
#include "PowerUtils.hh"

#include "Log.hh"
#include "ScopedTimer.hh"
#include "Types.hh"

using namespace power::utils;

namespace sta {

void Power::getCircuitStat()
{
  utils::ScopedTimer timer_get_circuit_stat("Report Circuit Stats");

  ensureActivities();

  LeafInstanceIterator *inst_iter = network_->leafInstanceIterator();
  size_t n_gate = 0, n_gate_of_macro = 0, n_gate_of_pad = 0, n_gate_of_clock = 0, n_gate_of_sequential = 0, n_gate_of_combinational = 0;
  size_t n_fallback_gate = 0;
  size_t n_output_pin = 0, n_input_pin = 0, n_other_pin = 0, n_total_pin = 0;
  size_t n_min_pin = std::numeric_limits<size_t>::max(), n_max_pin = std::numeric_limits<size_t>::min();  // input plus output 
  size_t n_min_input_pin = std::numeric_limits<size_t>::max(), n_max_input_pin = std::numeric_limits<size_t>::min();
  size_t n_min_output_pin = std::numeric_limits<size_t>::max(), n_max_output_pin = std::numeric_limits<size_t>::min();
  size_t n_event = 0, n_event_of_input_pin = 0, n_event_of_output_pin = 0, n_event_of_clock_pin = 0, n_event_of_macro = 0, n_event_of_pad = 0, n_event_of_clock = 0, n_event_of_sequential = 0, n_event_of_combinational = 0;
  size_t n_min_gate_event = std::numeric_limits<size_t>::max(), n_max_gate_event = std::numeric_limits<size_t>::min();
  size_t n_min_input_pin_event = std::numeric_limits<size_t>::max(), n_max_input_pin_event = std::numeric_limits<size_t>::min();
  size_t n_min_output_pin_event = std::numeric_limits<size_t>::max(), n_max_output_pin_event = std::numeric_limits<size_t>::min();
  std::map<NPinVal, size_t> gate_with_n_pin_to_n_gate;
  std::map<NPinVal, size_t> gate_with_n_pin_to_n_event;
  while (inst_iter->hasNext()) {
    Instance *inst = inst_iter->next();
    LibertyCell *cell = network_->libertyCell(inst);
    if (!cell) {
      continue;
    }

    InstancePinIterator *pin_iter = network_->pinIterator(inst);
    size_t n_cur_gate_input_pin = 0, n_cur_gate_output_pin = 0;
    size_t n_cur_gate_event = 0;
    while (pin_iter->hasNext()) {
      const Pin *cur_pin = pin_iter->next();
      size_t n_cur_pin_event = findActivity(cur_pin).nEvent();
      LibertyPort *cur_port = network_->libertyPort(cur_pin);
      if (!cur_port) {
        continue;
      }

      // ---------gate information---------
      ++n_total_pin;
      if (cur_port->direction()->isAnyOutput()) {
        ++n_output_pin;
        ++n_cur_gate_output_pin;
      } else if (cur_port->direction()->isAnyInput()) {
        ++n_input_pin;
        ++n_cur_gate_input_pin;
      } else {
        ++n_other_pin;
      }
      // ---------end of gate information---------

      // ---------event information---------
      n_cur_gate_event += n_cur_pin_event;
      if (cur_port->direction()->isAnyOutput()) {
        n_event_of_output_pin += n_cur_pin_event;
        n_min_output_pin_event = std::min(n_min_output_pin_event, n_cur_pin_event);
        n_max_output_pin_event = std::max(n_max_output_pin_event, n_cur_pin_event);
      } else if (cur_port->direction()->isAnyInput()) {
        n_event_of_input_pin += n_cur_pin_event;
        n_min_input_pin_event = std::min(n_min_input_pin_event, n_cur_pin_event);
        n_max_input_pin_event = std::max(n_max_input_pin_event, n_cur_pin_event);
      }
      if (cur_port->isClock()) {
        n_event_of_clock_pin += n_cur_pin_event;
      }
      // ---------end of event information---------
    }
    delete pin_iter;

    size_t n_cur_gate_pin = n_cur_gate_input_pin + n_cur_gate_output_pin;
    if (n_cur_gate_pin > G_CONFIG.nums.max_n_pin_for_leakage_power || n_cur_gate_pin > G_CONFIG.nums.max_n_pin_for_internal_power) {
      ++n_fallback_gate;
    }
    n_min_pin = std::min(n_min_pin, n_cur_gate_pin);
    n_max_pin = std::max(n_max_pin, n_cur_gate_pin);
    n_min_input_pin = std::min(n_min_input_pin, n_cur_gate_input_pin);
    n_max_input_pin = std::max(n_max_input_pin, n_cur_gate_input_pin);
    n_min_output_pin = std::min(n_min_output_pin, n_cur_gate_output_pin);
    n_max_output_pin = std::max(n_max_output_pin, n_cur_gate_output_pin);
    n_min_gate_event = std::min(n_min_gate_event, n_cur_gate_event);
    n_max_gate_event = std::max(n_max_gate_event, n_cur_gate_event);
    if (cell->isMacro() || cell->isMemory() || cell->interfaceTiming()) {
      ++n_gate_of_macro;
      n_event_of_macro += n_cur_gate_event;
    } else if (cell->isPad()) {
      ++n_gate_of_pad;
      n_event_of_pad += n_cur_gate_event;
    } else if (inClockNetwork(inst)) {
      ++n_gate_of_clock;
      n_event_of_clock += n_cur_gate_event;
    } else if (cell->hasSequentials()) {
      ++n_gate_of_sequential;
      n_event_of_sequential += n_cur_gate_event;
    } else {
      ++n_gate_of_combinational;
      n_event_of_combinational += n_cur_gate_event;
    }
    ++n_gate;
    n_event += n_cur_gate_event;
    gate_with_n_pin_to_n_gate[n_cur_gate_pin] += 1;
    gate_with_n_pin_to_n_event[n_cur_gate_pin] += n_cur_gate_event;
  }
  delete inst_iter;

  if (n_gate_of_macro + n_gate_of_pad + n_gate_of_clock + n_gate_of_sequential + n_gate_of_combinational != n_gate) {
    LOG_ERROR << "n_gate_of_macro + n_gate_of_pad + n_gate_of_clock + n_gate_of_sequential + n_gate_of_combinational != n_gate in circuit stat";
  }
  if (n_event_of_macro + n_event_of_pad + n_event_of_clock + n_event_of_sequential + n_event_of_combinational != n_event) {
    LOG_ERROR << "n_event_of_macro + n_event_of_pad + n_event_of_clock + n_event_of_sequential + n_event_of_combinational != n_event in circuit stat";
  }

  LOG_BEGIN(INFO, "Circuit Stat");
  LOG_INFO << "n_gate: " << n_gate
    << " n_gate_of_macro: " << n_gate_of_macro << " n_gate_of_pad: " << n_gate_of_pad
    << " n_gate_of_clock: " << n_gate_of_clock << " n_gate_of_sequential: " << n_gate_of_sequential
    << " n_gate_of_combinational: " << n_gate_of_combinational;
  LOG_INFO << "n_fallback_gate: " << n_fallback_gate << " of " << n_gate;
  LOG_INFO << "n_input_pin: " << n_input_pin << " n_output_pin: " << n_output_pin 
    << " n_other_pin: " << n_other_pin << " n_total_pin: " << n_total_pin;
  LOG_INFO << "netCount: " << network_->netCount() << " pinCount: " << network_->pinCount() << " network_: " << network_->instanceCount();
  LOG_INFO << "n_min_pin: " << n_min_pin << " n_max_pin: " << n_max_pin
    << " n_min_input_pin: " << n_min_input_pin << " n_max_input_pin: " << n_max_input_pin 
    << " n_min_output_pin: " << n_min_output_pin << " n_max_output_pin: " << n_max_output_pin;
  LOG_INFO << "n_event: " << n_event 
    << " n_event_of_input_pin: " << n_event_of_input_pin << " n_event_of_output_pin: " << n_event_of_output_pin << " n_event_of_clock_pin: " << n_event_of_clock_pin
    << " n_event_of_macro: " << n_event_of_macro << " n_event_of_pad: " << n_event_of_pad
    << " n_event_of_clock: " << n_event_of_clock << " n_event_of_sequential: " << n_event_of_sequential
    << " n_event_of_combinational: " << n_event_of_combinational
    << " n_min_gate_event: " << n_min_gate_event << " n_max_gate_event:" << n_max_gate_event
    << " n_min_input_pin_event: " << n_min_input_pin_event << " n_max_input_pin_event: " << n_max_input_pin_event
    << " n_min_output_pin_event: " << n_min_output_pin_event << " n_max_output_pin_event: " << n_max_output_pin_event;
  for (const auto& gate_event: gate_with_n_pin_to_n_event) {
    LOG_INFO << "Gate with n_pin: " << gate_event.first
      << " gate number: " << gate_with_n_pin_to_n_gate.at(gate_event.first)
      << " event number: " << gate_event.second;
  }
  static constexpr NPinVal bsim_thresholds[] = {4, 8, 16, 32};
  for (NPinVal bsim_threshold: bsim_thresholds) {
    size_t n_filtered_gate = 0;
    size_t n_filtered_event = 0;
    for (auto iter = gate_with_n_pin_to_n_gate.begin(); iter != gate_with_n_pin_to_n_gate.end(); ++iter) {
      if (iter->first <= bsim_threshold) {
        continue;
      }
      n_filtered_gate += iter->second;
      n_filtered_event += gate_with_n_pin_to_n_event.at(iter->first);
    }

    const double filtered_gate_ratio = n_gate == 0
      ? 0.0
      : static_cast<double>(n_filtered_gate) / static_cast<double>(n_gate);
    const double filtered_event_ratio = n_event == 0
      ? 0.0
      : static_cast<double>(n_filtered_event) / static_cast<double>(n_event);
    LOG_INFO << "BSIM threshold stat T=" << bsim_threshold
      << " filtered_gate: " << n_filtered_gate << " of " << n_gate
      << " filtered_gate_ratio: " << filtered_gate_ratio
      << " filtered_event: " << n_filtered_event << " of " << n_event
      << " filtered_event_ratio: " << filtered_event_ratio
      << " dense_event_coverage: " << (1.0 - filtered_event_ratio);
  }
  LOG_END(INFO, "Circuit Stat");
}

PowerResult
Power::timeBasedPower(const Instance *inst,
                      LibertyCell *cell,
                      const Corner *corner,
                      PowerResult& total_result)
{
  PowerResult result;

  findTimeBasedAllPower(inst, cell, corner, result, total_result);
  result.internal() += result.glitchInternal();
  result.switching() += result.glitchSwitching();

  if (LOG_DEBUG_FLAG) {
    PowerVal cur_gate_regular_internal = result.internal() - result.glitchInternal();
    PowerVal cur_gate_glitch_internal = result.glitchInternal();
    PowerVal cur_gate_regular_switching = result.switching() - result.glitchSwitching();
    PowerVal cur_gate_glitch_switching = result.glitchSwitching();
    PowerVal cur_gate_leakage = result.leakage();

    LOG_DEBUG << network_->pathName(inst) << G_CONFIG.strs.power_analysis_res_file_separator 
      << cur_gate_regular_internal << G_CONFIG.strs.power_analysis_res_file_separator << cur_gate_glitch_internal << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_switching << G_CONFIG.strs.power_analysis_res_file_separator << cur_gate_glitch_switching << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_leakage << G_CONFIG.strs.power_analysis_res_file_separator
      << cur_gate_regular_internal + cur_gate_glitch_internal + cur_gate_regular_switching + cur_gate_glitch_switching + cur_gate_leakage;
  }

  return result;
}

void
Power::getPinInformation(const Instance *inst, std::vector<const Pin *> *pins, NPinVal *n_pin, NPinVal *n_input_pin, NPinVal *n_output_pin) const
{
  InstancePinIterator *pin_iter = network_->pinIterator(inst);
  *n_pin = *n_input_pin = *n_output_pin = 0;
  while (pin_iter->hasNext()) {  // push input pin first
    const Pin *cur_pin = pin_iter->next();
    LibertyPort *cur_port = network_->libertyPort(cur_pin);
    if (cur_port) {
      if (cur_port->direction()->isAnyInput()) {
        pins->push_back(cur_pin);
        ++(*n_pin);
        ++(*n_input_pin); 
      }
    }
  }
  delete pin_iter;

  pin_iter = network_->pinIterator(inst);
  while (pin_iter->hasNext()) {  // then push output pin
    const Pin *cur_pin = pin_iter->next();
    LibertyPort *cur_port = network_->libertyPort(cur_pin);
    if (cur_port) {
      if (cur_port->direction()->isAnyOutput()) {
        pins->push_back(cur_pin);
        ++(*n_pin);
        ++(*n_output_pin); 
      }
    }
  }
  delete pin_iter;
  pin_iter = nullptr;
  assert(*n_pin == (*n_input_pin + *n_output_pin) && *n_pin == pins->size());
}

void 
Power::getLeakagePower(
  const LibertyCell *cell, 
  const LibertyCell *corner_cell, 
  const NPinVal n_pin,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  // Return values.
  std::vector<PowerVal>& leakage_power_values,
  PowerVal& default_leakage_power_val,
  bool& default_leakage_exists
) const
{
  const LeakagePowerSeq& leakage_powers = corner_cell->leakagePowers();
  cell->leakagePower(default_leakage_power_val, default_leakage_exists);
  NStateVal n_state = INVALID_N_STATE;
  if (n_pin > G_CONFIG.nums.max_n_pin_for_leakage_power) {  // too many pins will cause memory overflow
    assert(leakage_powers.size() == 0 || leakage_powers.size() == 1);
    if (leakage_powers.size() == 1) {
      default_leakage_power_val = leakage_powers[0]->power();
      default_leakage_exists = true;
    }
  } else {
    if (leakage_powers.size() != 0) {
      n_state = 1 << n_pin;
    }
  }

  leakage_power_values.resize(n_state == INVALID_N_STATE ? 0 : n_state);
  if (n_state != INVALID_N_STATE) {
    utils::ScopedTimer timer_bsim_construction("BSIM/state-index mapping construction");
    assert(leakage_power_values.size() != 0);
    for (NStateVal i = 0; i < n_state; ++i) {
      leakage_power_values[i] = default_leakage_exists ? default_leakage_power_val : 0;
    }

    for (const LeakagePower* const pwr: leakage_powers) { // TODO sort leakage power to make unconditioned one ahead
      std::vector<NStateVal> matched_state_idxs;
      getStateIdx(port_name_to_idx_map, pwr->whenStr(), G_CONFIG.strs.leakage_power_separator, matched_state_idxs);
      for (const NStateVal matched_state_idx: matched_state_idxs) {
        leakage_power_values[matched_state_idx] = pwr->power();
      }
    }
  }
}

NToggleVal
Power::getGlitchScalingRatioClockCycleBasedH(
  NEeventVal cur_event_idx, const Pin* pin,
  NEeventVal n_event, const VcdValue* vcd_values,
  const std::vector<bool>& event_glitch_flag,
  const Corner *corner
) {
  if (event_glitch_flag.at(cur_event_idx)) {
    VcdEventTime prev_glitch_pulse_width = INVALID_PULSE_WIDTH;
    if (cur_event_idx - 1 >= 0 && event_glitch_flag.at(cur_event_idx - 1)) {
      prev_glitch_pulse_width = vcd_values[cur_event_idx].time() - vcd_values[cur_event_idx - 1].time();
    }

    VcdEventTime next_glitch_pulse_width = INVALID_PULSE_WIDTH;
    if (cur_event_idx + 1 < n_event && event_glitch_flag.at(cur_event_idx + 1)) {  // next event
      next_glitch_pulse_width = vcd_values[cur_event_idx + 1].time() - vcd_values[cur_event_idx].time();
    }

    VcdEventTime selected_glitch_pulse_width = std::max(prev_glitch_pulse_width, next_glitch_pulse_width);
    if (selected_glitch_pulse_width != INVALID_PULSE_WIDTH) {
      Vertex* vertex = graph_->pinLoadVertex(pin);
      const SlewVal rise_slew = getSlew(vertex, RiseFall::rise(), corner);
      const SlewVal fall_slew = getSlew(vertex, RiseFall::fall(), corner);
      if (delayInf(rise_slew) || delayInf(fall_slew)) {
        LOG_ERROR << "Invalid slew value of " << network_->pathName(pin) 
          << " rise slew: " << rise_slew
          << " fall slew: " << fall_slew;
      }
      const SlewVal sum_slew = rise_slew + fall_slew;
      return getTimeBasedGlitchScalingRatioH(selected_glitch_pulse_width * vcd_.timeScale(), sum_slew);
    } else {
      return INVALID_N_TOGGLE_VAL;
    }
  } else {
    return INVALID_N_TOGGLE_VAL;
  }
}

void
Power::findInputInternalVal(
  const LibertyCell* corner_cell,
  const std::vector<VcdEventVal>& pin_states,
  const Pin* toggle_pin,
  RISEFALL rise_fall,
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
  const Corner *corner,
  const DcalcAnalysisPt* dcalc_ap,
  // Return values.
  EnergyVal* internal_energy
)
{
  //-------------------------------init------------------------------------
  const LibertyPort* toggle_port = network_->libertyPort(toggle_pin);
  const LibertyPort* toggle_corner_port = toggle_port->cornerPort(dcalc_ap);
  if (!corner_cell || !toggle_corner_port) {
    *internal_energy = 0.0;
    return;
  }
  InternalPowerSeq internal_pwrs;
  corner_cell->internalPowers(toggle_corner_port, internal_pwrs);

  Vertex* toggle_vertex = graph_->pinLoadVertex(toggle_pin);
  RiseFall* rf = rise_fall == RISE ? RiseFall::rise() : RiseFall::fall();
  SlewVal slew = getSlew(toggle_vertex, rf, corner);
  if (delayInf(slew)) {
    *internal_energy = 0.0;
    return;
  }
  //-------------------------------end of init------------------------------------

  for (const InternalPower *pwr: internal_pwrs) {
    if (isInternalPowerMatchPinStates(pin_states, pwr, port_name_to_idx_map)) {
      *internal_energy = pwr->power(rf, dcalc_ap->operatingConditions(), slew, 0.0);
      return;
    }
  }

  *internal_energy = 0.0;
}

void
Power::findOutputInternalVal(
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
)
{
  //-------------------------------init------------------------------------
  const LibertyPort* toggle_port = network_->libertyPort(toggle_pin);
  const LibertyPort* toggle_corner_port = toggle_port->cornerPort(dcalc_ap);
  if (!corner_cell || !toggle_corner_port) {
    *internal_energy = 0.0;
    return;
  }
  InternalPowerSeq internal_pwrs;
  corner_cell->internalPowers(toggle_corner_port, internal_pwrs);
  //-------------------------------end of init------------------------------------

  // ----------find related pin-----------
  EventTimeVal min_diff_time = vcd_.timeMax() * vcd_.timeScale();
  NPinVal related_pin_idx = INVALID_PIN_IDX;
  NEeventVal related_pin_event_idx = INVALID_WAVEFORM_PTR;
  for (NPinVal input_pin_idx = 0; input_pin_idx < n_input_pin; ++input_pin_idx) {
    if (pin_activities.at(input_pin_idx).nEvent() == 0) {
      continue;
    }
    const VcdValue* cur_pin_vcd_values = pin_activities.at(input_pin_idx).vcdValues();
    // const int cur_pin_value_bit = pin_activities.at(input_pin_idx).valueBit();
    NEeventVal cur_pin_related_event_idx = std::max(
      static_cast<NEeventVal>(0),
      getEventIdxByTime(cur_pin_vcd_values, pin_activities.at(input_pin_idx).nEvent(), cur_time) - 1 // minus one to get the largest event idx that the time is smaller than cur_time
    );

    // ----------delay-----------
    NEeventVal cur_pin_prev_event_idx = std::max(
      static_cast<NEeventVal>(0),
      cur_pin_related_event_idx - 1
    );
    Edge *edge;
    const TimingArc *arc;
    graph_->gateEdgeArc(
      pins.at(input_pin_idx), 
      getRiseFallEdge(
        vcd_value_to_int_map_.at(cur_pin_vcd_values[cur_pin_prev_event_idx].value(pin_activities.at(input_pin_idx).valueBit())), 
        vcd_value_to_int_map_.at(cur_pin_vcd_values[cur_pin_related_event_idx].value(pin_activities.at(input_pin_idx).valueBit()))
      ) == RISE ? RiseFall::rise() : RiseFall::fall(), 
      toggle_pin, 
      to_rf == RISE ? RiseFall::rise() : RiseFall::fall(), edge, arc
    );
    if (!edge || !arc) {
      continue;
    }
    ArcDelay cell_arc_delay = graph_->arcDelay(edge, arc, dcalc_ap->index());
    // ----------end of delay-----------

    EventTimeVal cur_pin_diff_time = std::abs(cur_time * vcd_.timeScale() - cell_arc_delay - cur_pin_vcd_values[cur_pin_related_event_idx].time() * vcd_.timeScale());
    if (LOG_DEBUG_FLAG) {
      LOG_DEBUG << network_->pathName(inst) 
        << " cur_time: " << cur_time 
        << " input_pin_idx: " << input_pin_idx 
        << " cur_pin_vcd_values[cur_pin_related_event_idx].time(): " << cur_pin_vcd_values[cur_pin_related_event_idx].time() 
        << " cell_arc_delay: " << cell_arc_delay
        << " cur_pin_diff_time: " << cur_pin_diff_time;
    }
    if (cur_pin_diff_time < min_diff_time && 
        clkedWaveformIdx(cur_pin_vcd_values[cur_pin_related_event_idx].time(), vcd_.timeScale(), clk_period_) == clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)) {
      min_diff_time = cur_pin_diff_time;
      related_pin_idx = input_pin_idx;
      related_pin_event_idx = cur_pin_related_event_idx;
    }
  }
  if (LOG_DEBUG_FLAG) {
    LOG_DEBUG << "related_pin_idx: " << related_pin_idx << " min_diff_time: " << min_diff_time;
  }
  // ----------end of find related pin-----------

  if (related_pin_idx == INVALID_PIN_IDX) {
    getDefaultOutputPinInternalEnergyVal(inst, pins, toggle_pin, to_rf, n_input_pin, internal_pwrs, corner, dcalc_ap, internal_energy);
    return;
  }

  const Pin* related_pin = pins.at(related_pin_idx);
  Vertex* related_vertex = graph_->pinLoadVertex(related_pin);
  const VcdValue* related_pin_vcd_values = pin_activities.at(related_pin_idx).vcdValues();
  const int related_pin_value_bit = pin_activities.at(related_pin_idx).valueBit();
  NEeventVal related_pin_prev_event_idx = std::max(
    static_cast<NEeventVal>(0),
    related_pin_event_idx - 1
  );
  RISEFALL from_rf = getRiseFallEdge(
    vcd_value_to_int_map_.at(related_pin_vcd_values[related_pin_prev_event_idx].value(related_pin_value_bit)), 
    vcd_value_to_int_map_.at(related_pin_vcd_values[related_pin_event_idx].value(related_pin_value_bit))
  );
  SlewVal input_slew = getSlew(related_vertex, from_rf == RISE ? RiseFall::rise() : RiseFall::fall(), corner);
  if (!delayInf(input_slew)) {
    for (const InternalPower* pwr: internal_pwrs) {
      const LibertyPort *from_corner_port = pwr->relatedPort();
      if (from_corner_port && related_pin == findLinkPin(inst, from_corner_port)) {
        if (isInternalPowerMatchPinStates(pin_states, pwr, port_name_to_idx_map)) {
          *internal_energy = pwr->power(
            to_rf == RISE ? RiseFall::rise() : RiseFall::fall(), 
            dcalc_ap->operatingConditions(), 
            input_slew, 
            graph_delay_calc_->loadCap(toggle_pin, dcalc_ap)
          );
          return;
        }
      }
    }
    *internal_energy = 0.0;
    return;
  } else {
    *internal_energy = 0.0;
    return;
  }
}

void
Power::getDefaultOutputPinInternalEnergyVal(
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
)
{
  int n_table = 0;
  EnergyVal total_energy = 0.0;

  for (NPinVal input_pin_idx = 0; input_pin_idx < n_input_pin; ++input_pin_idx) {
    const Pin* input_pin = pins.at(input_pin_idx);
    Vertex* input_vertex = graph_->pinLoadVertex(input_pin);
    for (RiseFall *from_rf : RiseFall::range()) {
      SlewVal input_slew = getSlew(input_vertex, from_rf, corner);
      if (!delayInf(input_slew)) {
        for (const InternalPower* pwr: internal_pwrs) {
          const LibertyPort *from_corner_port = pwr->relatedPort();
          if (from_corner_port && input_pin == findLinkPin(inst, from_corner_port)) {
            total_energy += pwr->power(
              to_rf == RISE ? RiseFall::rise() : RiseFall::fall(), 
              dcalc_ap->operatingConditions(), 
              input_slew, 
              graph_delay_calc_->loadCap(toggle_pin, dcalc_ap)
            );
            n_table++;
          }
        }
      }
    }
  }

  if (n_table == 0) {
    *internal_energy = 0;
  } else {
    *internal_energy = total_energy / n_table;
  }
}

void
Power::getStateIdx(
  const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map, 
  const std::string& when,
  const std::string& when_str_separator,
  // Return values.
  std::vector<NStateVal>& matched_state_idxs
) const 
{
  StringVector when_pin_states;
  split(when, when_str_separator, when_pin_states);

  NStateVal matched_state_idx = 0;
  std::unordered_set<std::string> used_ports;
  for (auto& pin_state: when_pin_states) {
    trim(pin_state);
    VcdEventVal cur_pin_state = 1;
    if (pin_state.substr(0, 1) == "!") {
      cur_pin_state = 0;
      pin_state = pin_state.substr(1);
    }

    // if (LOG_DEBUG_FLAG) {
    //   LOG_DEBUG << pin_state << " idx: " << port_name_to_idx_map.at(pin_state);
    // }
    used_ports.insert(pin_state);
    matched_state_idx += cur_pin_state * (1 << port_name_to_idx_map.at(pin_state));
  }

  // std::vector<NStateVal> matched_state_idxs(1, matched_state_idx);
  matched_state_idxs.push_back(matched_state_idx);
  // like https://leetcode.cn/problems/letter-combinations-of-a-phone-number/
  for (auto it = port_name_to_idx_map.begin(); it != port_name_to_idx_map.end(); ++it) {
    if (!used_ports.count(it->first)) {
      std::vector<NStateVal> tmp_state_idxs;
      for (NStateVal tmp_state: matched_state_idxs) {
        tmp_state_idxs.push_back(tmp_state);
        tmp_state_idxs.push_back(tmp_state + (1 << it->second));
      }
      matched_state_idxs = std::move(tmp_state_idxs);
    }
  }
}

void
Power::getEventClockCycleBasedGlitchFlags(
  const VcdValue *vcd_values,
  NEeventVal start_idx_in_vcd_values,
  NEeventVal end_idx_in_vcd_values,
  // Return values.
  std::vector<bool>& event_glitch_flags
) {
  if (vcd_values == nullptr || start_idx_in_vcd_values == end_idx_in_vcd_values) {
    return;
  }

  event_glitch_flags.resize(end_idx_in_vcd_values - start_idx_in_vcd_values, false);
  std::vector<NEeventVal> cur_period_event_idxes;
  NPeriodVal cur_period_idx = clkedWaveformIdx(vcd_values[start_idx_in_vcd_values].time(), vcd_.timeScale(), clk_period_);
  if (start_idx_in_vcd_values != 0) {  // do not consider the initial condition as transition event
    cur_period_event_idxes.push_back(start_idx_in_vcd_values);
  }
  for (NEeventVal pos = start_idx_in_vcd_values + 1; pos < end_idx_in_vcd_values; ++pos) {
    NPeriodVal cur_event_period_idx = clkedWaveformIdx(vcd_values[pos].time(), vcd_.timeScale(), clk_period_);
    if (cur_event_period_idx != cur_period_idx) {
      size_t n_event_in_prev_period = cur_period_event_idxes.size();
      if (n_event_in_prev_period >= 2) {
        for (size_t idx = 0; idx < ((n_event_in_prev_period % 2 == 0) ? n_event_in_prev_period : (n_event_in_prev_period - 1)); ++idx) {
          event_glitch_flags.at(cur_period_event_idxes.at(idx) - start_idx_in_vcd_values) = true;
        }
      }

      // for (auto idx: cur_period_event_idxes) {
      //   LOG_INFO << vcd_values[idx].time() << " is glitch: " << event_glitch_flags.at(idx - start_idx_in_vcd_values);
      // }
      cur_period_idx = cur_event_period_idx;
      cur_period_event_idxes.clear();
    }
    cur_period_event_idxes.push_back(pos);
  }

  //---------------wrap up---------------
  size_t n_event_in_prev_period = cur_period_event_idxes.size();
  if (n_event_in_prev_period >= 2) {
    for (size_t idx = 0; idx < ((n_event_in_prev_period % 2 == 0) ? n_event_in_prev_period : (n_event_in_prev_period - 1)); ++idx) {
      event_glitch_flags.at(cur_period_event_idxes.at(idx) - start_idx_in_vcd_values) = true;
    }
  }
  // for (auto idx: cur_period_event_idxes) {
  //   LOG_INFO << vcd_values[idx].time() << " is glitch: " << event_glitch_flags.at(idx - start_idx_in_vcd_values);
  // }
  //---------------end of wrap up---------------
}

NEeventVal 
Power::getEventIdxByTime(const VcdValue* vcd_values, NEeventVal n_event, VcdEventTime time) const {
  NEeventVal left = 0, right = n_event;
  while (left < right) {  // [left, right)
    NEeventVal mid = left + (right - left) / 2;  // avoid overflow
    if (vcd_values[mid].time() < time) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  return right;
}

NEeventVal 
Power::getEventIdxByTime(const VcdValue* vcd_values, NEeventVal n_event, VcdEventTime time, NEeventVal left) const {
  NEeventVal right = n_event;
  while (left < right) {  // [left, right)
    NEeventVal mid = left + (right - left) / 2;  // avoid overflow
    if (vcd_values[mid].time() < time) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  assert(left == right);
  return right;
}

NEeventVal 
Power::getEventIdxByTime(const VcdValues& vcd_values, VcdEventTime time) const
{
  return getEventIdxByTime(vcd_values.data(), vcd_values.size(), time);
}

void 
Power::calculatePerCycleLeakagePower(
  const Instance *inst,
  const PowerVal leakage_val,
  const VcdTime prev_time,
  const VcdTime cur_time,
  const double time_scale,
  const double clk_period,
  PowerResult& result) const
{
  // a two pointers algorithm is adopted to calculate the per cycle leakage power
  EventTimeVal left = prev_time * time_scale;
  NPeriodVal cur_cycle_idx = clkedWaveformIdx(prev_time, time_scale, clk_period);
  while (cur_cycle_idx * clk_period <= cur_time * time_scale) {
    EventTimeVal right = std::min((cur_cycle_idx + 1) * clk_period, cur_time * time_scale);
    EventTimeVal duration = right - left;
    result.findLeakagePowerClkedWaveform(network_->topInstance()).waveform()[cur_cycle_idx] += leakage_val * duration / clk_period;

    // if (LOG_DEBUG_FLAG) {
    //   LOG_DEBUG << "per cycle leakage power " << network_->pathName(inst) << " prev_time: " << prev_time << " cur_time: " << cur_time << " state: " << input_state_annotation << " left: " << left << " right: " << right << " leakage_val: " << leakage_val;
    // }
    left = (cur_cycle_idx + 1) * clk_period;
    ++cur_cycle_idx;
  }
}

void
Power::findTimeBasedAllPower(const Instance *inst,
                        LibertyCell *cell,
                        const Corner *corner,
                        // Return values.
                        PowerResult &result,
                        PowerResult &total_result)
{
  VcdTime MAX_TIME = vcd_.timeMax() + 10;
  const DcalcAnalysisPt *dcalc_ap = corner->findDcalcAnalysisPt(MinMax::max());
  const LibertyCell *corner_cell = cell->cornerCell(corner, MinMax::max());

  // ---------------------------init pin information------------------------------
  std::vector<const Pin *> pins;
  NPinVal n_pin = 0, n_input_pin = 0, n_output_pin = 0;
  getPinInformation(inst, &pins, &n_pin, &n_input_pin, &n_output_pin);

  std::vector<PwrActivity> pin_activities;
  std::unordered_map<std::string, NPinVal> port_name_to_idx_map;  // input pin with small index, output pin wi big index
  for (NPinVal pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
    const Pin *cur_pin = pins.at(pin_idx);
    LibertyPort *cur_port = network_->libertyPort(cur_pin);
    if (cur_port) {
      PwrActivity activity = findActivity(cur_pin);
      pin_activities.emplace_back(activity.activity(),
                         activity.duty(),
                         activity.origin(),
                         activity.vcdValues(),
                         activity.nEvent(),
                         activity.valueBit());
      port_name_to_idx_map[cur_port->name()] = pin_idx;
    }
  }
  // ------------------------------init pin information------------------------------

  // ------------------------------get clock cycle based glitch flags------------------------------
  std::vector<std::vector<bool>> pin_event_glitch_flags(n_pin);
  for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    pin_event_glitch_flags.at(pin_idx).resize(pin_activities.at(pin_idx).nEvent(), false);
    const LibertyPort *cur_port = network_->libertyPort(pins.at(pin_idx));
    if (!cur_port->isClock()) {
      getEventClockCycleBasedGlitchFlags(pin_activities.at(pin_idx).vcdValues(), 0, pin_activities.at(pin_idx).nEvent(), pin_event_glitch_flags.at(pin_idx));
    }
  }
  // ------------------------------end of get clock cycle based glitch flags------------------------------

  // ------------------------------get pin_single_switching_energies------------------------------
  std::vector<EnergyVal> pin_single_switching_energies(n_pin, 0);
  for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
    const Pin* cur_pin = pins.at(pin_idx);
    const CapacitanceVal load_cap = graph_delay_calc_->loadCap(cur_pin, dcalc_ap);
    LibertyPort *cur_port = network_->libertyPort(cur_pin);
    if (cur_port && cur_port->direction()->isAnyOutput()) {
      const VoltageVal volt = portVoltage(corner_cell, cur_port, dcalc_ap);
      pin_single_switching_energies.at(pin_idx) = load_cap * volt * volt;
    }
  }
  // ------------------------------end of get pin_single_switching_energies------------------------------

  // ---------------------------get leakage power------------------------------
  std::vector<PowerVal> leakage_power_values;
  PowerVal default_leakage_power_val = 0.0;
  bool default_leakage_exists = false;
  getLeakagePower(cell, corner_cell, n_pin, port_name_to_idx_map, leakage_power_values, default_leakage_power_val, default_leakage_exists);
  // ---------------------------end of get leakage power------------------------------

  // ------------------------------loop initial------------------------------
  std::vector<NEeventVal> pin_frontier_event_idxes(n_pin, 0);
  std::vector<VcdEventVal> prev_pin_states(n_pin, INVALID_VCD_EVENT_VAL);
  for (NPinVal pin_idx = 0; pin_idx < pins.size(); ++pin_idx) {
    const PwrActivity& cur_pin_activity = pin_activities.at(pin_idx);
    if (cur_pin_activity.nEvent() == 0) {
      LOG_WARN << network_->pathName(pins.at(pin_idx)) << " has no events";
      prev_pin_states.at(pin_idx) = getVertexDefaultState(graph_->pinLoadVertex(pins.at(pin_idx)));
      continue;
    }
    int cur_pin_value_bit = cur_pin_activity.valueBit();
    const auto& cur_pin_first_event = cur_pin_activity.vcdValues()[0];

    if (cur_pin_first_event.time() == 0) {
      prev_pin_states.at(pin_idx) = vcd_value_to_int_map_.at(cur_pin_first_event.value(cur_pin_value_bit));
      pin_frontier_event_idxes.at(pin_idx) = 1;
    } else {  // take X state as the unknown initial state
      prev_pin_states.at(pin_idx) = 2;
    }
  }
  VcdTime prev_time = 0;
  std::vector<VcdEventVal> pin_states = prev_pin_states;
  // ------------------------------end of loop initial------------------------------

  double cur_gate_leakage_power_val = 0.0;
  std::vector<bool> cur_time_pin_triggereds(n_pin, false);
  while (true) {  // will exit until no unprocessed input event left
    // ------------------------------get current min time------------------------------
    VcdTime cur_time = MAX_TIME;
    for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
      const PwrActivity& cur_pin_activity = pin_activities.at(pin_idx);
      const NEeventVal cur_pin_frontier = pin_frontier_event_idxes.at(pin_idx);
      if (cur_pin_frontier >= cur_pin_activity.nEvent() || cur_pin_activity.vcdValues() == nullptr) {
        continue;
      }

      cur_time = std::min(cur_pin_activity.vcdValues()[cur_pin_frontier].time(), cur_time);
    }

    if (cur_time == MAX_TIME) {  // no unprocessed input events left 
      break;
    }
    // ------------------------------end of get current min time------------------------------

    // ------------------------------get new state and advance the frontier------------------------------
    for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
      const PwrActivity& cur_pin_activity = pin_activities.at(pin_idx);
      NEeventVal& cur_pin_frontier = pin_frontier_event_idxes.at(pin_idx);
      if (cur_pin_frontier >= cur_pin_activity.nEvent() || cur_pin_activity.vcdValues() == nullptr) {
        continue;
      }
      int cur_input_value_bit = cur_pin_activity.valueBit();
      const auto& cur_input_first_event = cur_pin_activity.vcdValues()[cur_pin_frontier];

      if (cur_pin_activity.vcdValues()[cur_pin_frontier].time() == cur_time) {
        pin_states.at(pin_idx) = vcd_value_to_int_map_.at(cur_input_first_event.value(cur_input_value_bit));
        cur_time_pin_triggereds[pin_idx] = true;
        ++cur_pin_frontier;
      }
    }
    // ------------------------------end of get new state and advance the frontier------------------------------

    // ------------------------------find leakage power value according to previous state------------------------------
    PowerVal leakage_val = 0.0;
    findLeakageVal(prev_pin_states, leakage_power_values, default_leakage_power_val, default_leakage_exists, &leakage_val);
    // ------------------------------end of find leakage power value according to previous state------------------------------

    // ------------------------------calculate leakage power------------------------------
    calculatePerCycleLeakagePower(inst, leakage_val, prev_time, cur_time, vcd_.timeScale(), clk_period_, total_result);
    if (LOG_DEBUG_FLAG) {
      LOG_DEBUG << "total leakage power " << network_->pathName(inst) << " prev_time: " << prev_time << " cur_time: " << cur_time << " leakage_val: " << leakage_val;
    }
    result.leakage() += leakage_val * (cur_time - prev_time) / vcd_.timeMax();  // the time scale is reduced
    cur_gate_leakage_power_val += leakage_val * (cur_time - prev_time) / vcd_.timeMax();
    // ------------------------------end of calculate leakage power------------------------------
    
    // ------------------------------calculate internal power------------------------------
    for (NPinVal pin_idx = 0; pin_idx < n_pin; ++pin_idx) {
      if (cur_time_pin_triggereds[pin_idx]) {
        const float n_toggle = getNToggle(prev_pin_states[pin_idx], pin_states[pin_idx]);
        NToggleVal glitch_scaling_ratio = getGlitchScalingRatioClockCycleBasedH(pin_frontier_event_idxes.at(pin_idx) - 1, pins.at(pin_idx), pin_activities.at(pin_idx).nEvent(), pin_activities.at(pin_idx).vcdValues(), pin_event_glitch_flags.at(pin_idx), corner);

        EnergyVal cur_internal_energy = 0.0;
        if (pin_idx < n_input_pin) {
          findInputInternalVal(corner_cell, prev_pin_states, pins.at(pin_idx), getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]), port_name_to_idx_map, corner, dcalc_ap, &cur_internal_energy);
        } else {
          findOutputInternalVal(inst, corner_cell, prev_pin_states, cur_time, pins.at(pin_idx), getRiseFallEdge(prev_pin_states[pin_idx], pin_states[pin_idx]), n_input_pin, pins, pin_activities, port_name_to_idx_map, corner, dcalc_ap, &cur_internal_energy);
          //-----switching-----
          NToggleVal n_cur_transition_rise = getNRise(prev_pin_states[pin_idx], pin_states[pin_idx]);
          if (n_cur_transition_rise > 0) {
            const EnergyVal cur_switching_energy = n_cur_transition_rise * pin_single_switching_energies.at(pin_idx);
            if (glitch_scaling_ratio > 0) {
              result.glitchSwitching() += (glitch_scaling_ratio * cur_switching_energy) / (vcd_.timeMax() * vcd_.timeScale());
              total_result.findSwitchingPowerClkedWaveform(network_->topInstance()).waveform()[clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)] += (glitch_scaling_ratio * cur_switching_energy) / clk_period_;
              total_result.findGlitchSwitchingPowerClkedWaveform(network_->topInstance()).waveform()[clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)] += (glitch_scaling_ratio * cur_switching_energy) / clk_period_;
            } else {
              result.switching() += cur_switching_energy / (vcd_.timeMax() * vcd_.timeScale());
              total_result.findSwitchingPowerClkedWaveform(network_->topInstance()).waveform()[clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)] += cur_switching_energy / clk_period_;
            }
          }
          //-----end of switching-----
        }
        cur_internal_energy *= n_toggle;

        if (glitch_scaling_ratio > 0) {
          result.glitchInternal() += glitch_scaling_ratio * cur_internal_energy / (vcd_.timeMax() * vcd_.timeScale());
          total_result.findInternalPowerClkedWaveform(network_->topInstance()).waveform()[clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)] += glitch_scaling_ratio * cur_internal_energy / clk_period_;
          total_result.findGlitchInternalPowerClkedWaveform(network_->topInstance()).waveform()[clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)] += glitch_scaling_ratio * cur_internal_energy / clk_period_;
        } else {
          result.internal() += cur_internal_energy / (vcd_.timeMax() * vcd_.timeScale());
          total_result.findInternalPowerClkedWaveform(network_->topInstance()).waveform()[clkedWaveformIdx(cur_time, vcd_.timeScale(), clk_period_)] += cur_internal_energy / clk_period_;
        }
      }
    }
    // ------------------------------end of calculate internal power------------------------------

    // ------------------------------wrap up------------------------------
    prev_time = cur_time;
    prev_pin_states = pin_states;
    resetBoolVector(cur_time_pin_triggereds);
    // ------------------------------end of wrap up------------------------------
  }

  // final wrap up
  PowerVal final_leakage_val = 0.0;
  findLeakageVal(prev_pin_states, leakage_power_values, default_leakage_power_val, default_leakage_exists, &final_leakage_val);
  calculatePerCycleLeakagePower(inst, final_leakage_val, prev_time, vcd_.timeMax(), vcd_.timeScale(), clk_period_, total_result);
  result.leakage() += final_leakage_val * (vcd_.timeMax() - prev_time) / vcd_.timeMax();
  cur_gate_leakage_power_val += final_leakage_val * (vcd_.timeMax() - prev_time) / vcd_.timeMax();
  if (LOG_DEBUG_FLAG) {
    LOG_DEBUG << network_->pathName(inst) << " cur_gate_leakage_power_val: " << cur_gate_leakage_power_val;
  }
}
}  // end of namespace sta
