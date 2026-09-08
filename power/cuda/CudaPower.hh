#pragma once
#include <utility>
#include <unordered_map>
#include "Power.hh"

#include "Types.hh"
#include "Types.cuh"

namespace utils::cuda {
  class OneDimensionalLUTPair;
  class TwoDimensionalLUTPair;
}

namespace sta {
namespace power {
  class Gate;
  class Event;
}

class CudaPower : public Power
{
public:
  CudaPower(StaState *sta);
  void power(const Corner *corner,
             // Return values.
             PowerResult &total,
             PowerResult &sequential,
             PowerResult &combinational,
             PowerResult &clock,
             PowerResult &macro,
             PowerResult &pad) override;
  void printCellRes() const override;
  ~CudaPower();

protected:
  void printSettings();

  // ------------------------------time inteval scheduling------------------------------
  void getTimeIntervals(
    // Return values.
    std::vector<std::pair<VcdEventTime, VcdEventTime>>& intervals
  ) const;
  void getTimeIntervalBoudary(
    NEeventVal accu_event_count, 
    // Return values.
    VcdEventTime *boundary
  ) const;
  void getTimeIntervalsByCycle(
    // Return values.
    std::vector<std::pair<VcdEventTime, VcdEventTime>>& intervals
  ) const;
  void getTimeIntervalsByCycleIterateAllEvents(
    // Return values.
    std::vector<std::pair<VcdEventTime, VcdEventTime>>& intervals
  );
  void getCycleIntervalBoudary(
    NEeventVal accu_event_count, 
    // Return values.
    NPeriodVal *cycle_boundary
  ) const;
  void getCycleIntervalBoudary(
    NEeventVal accu_event_count, 
    NPeriodVal left,
    // Return values.
    NPeriodVal *cycle_boundary
  ) const;
  void getCycleIntervalBoudaryByCycleIdxToAccuEvents(
    NEeventVal accu_event_count, 
    const std::unordered_map<NPeriodVal, NEeventVal>& cycle_idx_to_accu_n_event,
    // Return values.
    NPeriodVal *cycle_boundary
  ) const;
  NEeventVal getAccuEventCountOfAllGates(VcdEventTime time) const;
  void initGateData(VcdEventTime start_time, VcdEventTime end_time, const Corner *corner, const DcalcAnalysisPt *dcalc_ap);
  void copyGateDataToDeviceSide();
  void getGateWaveformRangeSingleThread(VcdEventTime start_time, VcdEventTime end_time, std::vector<NEeventVal>& h_global_pin_waveform_starts, std::vector<NEeventVal>& h_global_pin_waveform_ends);
  void getGateWaveformRangeMultiThread(VcdEventTime start_time, VcdEventTime end_time, std::vector<NEeventVal>& h_global_pin_waveform_starts, std::vector<NEeventVal>& h_global_pin_waveform_ends);
  void renewGateWaveform(const std::vector<NEeventVal>& h_global_pin_waveform_starts, const std::vector<NEeventVal>& h_global_pin_waveform_ends);
  void copyWaveformToDeviceSide();
  // ------------------------------end of time inteval scheduling------------------------------
  void getWaveform(
    const Instance* inst,
    const std::vector<const Pin *>& pins,
    VcdEventTime start_time, 
    VcdEventTime end_time,
    // Return values.
    NEeventVal& accu_event_count,
    std::unordered_map<const VcdValue*, std::unordered_map<int, std::pair<NEeventVal, NEeventVal>>>& pin_vcd_values_ptr_to_value_bit_to_range_map,
    std::vector<NEeventVal>& pin_waveform_starts, 
    std::vector<NEeventVal>& pin_waveform_ends
  );
  void getWhenStateVec(
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map, 
    const std::string& when,
    const std::string& when_str_separator,
    const NPinVal n_pin,
    // Return values.
    std::vector<VcdEventVal>& when_state_vec
  ) const;
  struct InternalPowerData
  {
    std::vector<utils::cuda::OneDimensionalLUTPair**> input_ports_internal_power_LUTs;
    std::vector<NStateVal> input_ports_n_internal_power_LUTs_index_by_order;
    std::vector<utils::cuda::OneDimensionalLUTPair**> input_ports_internal_power_LUTs_index_by_order;
    std::vector<utils::cuda::TwoDimensionalLUTPair***> output_ports_internal_power_LUTs;
    std::vector<NStateVal*> output_ports_n_internal_power_LUTs_index_by_order;
    std::vector<utils::cuda::TwoDimensionalLUTPair***> output_ports_internal_power_LUTs_index_by_order;
  };
  struct LeakagePowerData
  {
    std::vector<PowerVal> leakage_power_values;
    PowerVal default_leakage_power_val = 0.0;
    bool default_leakage_exists = false;
  };
  const LeakagePowerData& getLeakagePowerData(
    const LibertyCell *cell,
    const LibertyCell *corner_cell,
    NPinVal n_pin,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
  );
  const InternalPowerData& getInternalPower(
    const Instance *inst, 
    const LibertyCell *corner_cell, 
    const DcalcAnalysisPt *dcalc_ap, 
    const NPinVal n_pin,
    const std::vector<const Pin *>& pins,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map
  );
  void getInputInternalPower(
    const LibertyCell *corner_cell, 
    const LibertyPort *port, 
    const DcalcAnalysisPt *dcalc_ap, 
    const NPinVal n_pin,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    // Return values.
    utils::cuda::OneDimensionalLUTPair**& d_cur_port_internal_power_LUTs,
    NStateVal& n_cur_port_internal_power_LUTs_indexed_by_order,
    utils::cuda::OneDimensionalLUTPair**& d_cur_port_internal_power_LUTs_indexed_by_order
  ) const;
  void getOutputInternalPower(
    const Instance *inst,
    const LibertyCell *corner_cell, 
    const LibertyPort *port, 
    const DcalcAnalysisPt *dcalc_ap,
    const NPinVal n_pin,
    const std::unordered_map<std::string, NPinVal>& port_name_to_idx_map,
    // Return values.
    utils::cuda::TwoDimensionalLUTPair***& d_cur_port_internal_power_LUTs,
    NStateVal*& d_cur_port_n_internal_power_LUTs_indexed_by_order,
    utils::cuda::TwoDimensionalLUTPair***& d_cur_port_internal_power_LUTs_indexed_by_order
  ) const;
  void getDelay(
    const std::vector<const Pin *>& pins, 
    NPinVal n_pin, 
    NPinVal n_input_pin, 
    const DcalcAnalysisPt *dcalc_ap,
    // Return values.
    std::vector<DelayVal>& global_cell_arc_delays
  ) const;
  void setEvent(VcdEventTime time, VcdEventVal val, bool is_glitch, NEeventVal pos);
  void initPerGatePower();
  void setPeriodAndInitPerCyclePower(VcdEventTime _max_event_time, EventTimeVal _vcd_time_scale);
  void scheduleKernel(VcdEventTime interval_start_time, VcdEventTime interval_end_time);
  void scheduleKernelForEventAndCycleBasedPartition(VcdEventTime interval_start_time, VcdEventTime interval_end_time);
  void runCudaPowerAnalysis(VcdEventTime interval_start_time, VcdEventTime interval_end_time, const std::pair<VcdEventTime, VcdEventTime>* next_interval = nullptr);
  template <class Kernel>
  void logKernelResourceUsage(const char* kernel_name, Kernel* kernel, NBlockVal grid_block_count, int block_thread_count) const;
  void reportPowerKernelResourceUsage(NBlockVal kernel1_all_grid_block_count) const;
  void mergeResult();
  void recordResult();
  void finalizeResult(
    PowerResult &total,
    PowerResult &sequential,
    PowerResult &combinational,
    PowerResult &clock,
    PowerResult &macro,
    PowerResult &pad
  );
  void getThreadAllocationStats(const VcdEventTime interval_start_time, const VcdEventTime interval_end_time) const;
  void releaseMemory();

private:
  float power_computation_kernel_total_time_;
  float k11_kernel_total_time_;
  float merge_result_kernel_total_time_;
  NPeriodVal n_period_;
  VcdEventTime max_event_time_;
  EventTimeVal vcd_time_scale_;
  VcdEventTime vcd_time_unit_per_cycle_;
  std::unordered_map<const Instance*, std::vector<const Pin *>> inst_to_pins_;
  std::unordered_map<const VcdValue*, NEeventVal> vcd_values_ptr_to_n_event_;
  std::vector<std::pair<const VcdValue*, int>> vcd_values_bit_pair_list_;
  std::unordered_map<const Instance*, PowerResult> inst_to_res_;

  //--------------------events and gates-----------------------
  sta::power::Event* events_;  // array of event
  sta::power::Event* d_events_;
  std::vector<sta::power::Gate*> h_multiple_output_gates_;
  sta::power::Gate* d_multiple_output_gates_;
  //--------------------end of events and gates-----------------------

  //--------------------for leakage and internal-----------------------
  NGateVal n_multiple_output_gate_;
  NBlockVal n_block_for_event_partition_;
  NBlockVal n_block_for_cycle_partition_;

  // -----auxiliary arrays for gates-----
  NEeventVal* global_pin_waveform_starts_;
  NEeventVal* global_pin_waveform_ends_;
  VcdEventVal* global_pin_default_states_;
  VoltageVal* global_pin_voltages_;
  SlewVal* global_pin_rise_slews_;
  SlewVal* global_pin_fall_slews_;
  CapacitanceVal* global_pin_load_capacitances_;
  bool* global_pin_is_clocks_;
  DelayVal* global_cell_arc_delays_;
  // -----end of auxiliary arrays for gates-----
  // -----auxiliary arrays to record res-----
  PowerVal* global_per_tile_leakage_res_;
  PowerVal* global_per_tile_internal_res_;
  PowerVal* global_per_tile_glitch_internal_res_;
  PowerVal* global_per_tile_switching_res_;
  PowerVal* global_per_tile_glitch_switching_res_;
  // -----end of auxiliary arrays to record res-----
  // -----end of auxiliary arrays-----
  std::unordered_map<const VcdValue*, std::unordered_map<int, NEeventVal>> var_val_ptr_to_value_bit_to_prev_right_bound_;

  NGateVal* block_corr_gate_idxes_for_event_partition_;
  NGateVal* block_corr_gate_idxes_for_cycle_partition_;
  std::unordered_map<const LibertyPort*, utils::cuda::OneDimensionalLUTPair**> input_port_to_internal_luts_map_;
  std::unordered_map<const LibertyPort*, NStateVal> input_port_to_n_internal_LUTs_indexed_by_order_map_;
  std::unordered_map<const LibertyPort*, utils::cuda::OneDimensionalLUTPair**> input_port_to_internal_luts_indexed_by_order_map_;
  std::unordered_map<const LibertyPort*, utils::cuda::TwoDimensionalLUTPair***> output_port_to_internal_luts_map_;
  std::unordered_map<const LibertyPort*, NStateVal*> output_port_to_n_internal_LUTs_indexed_by_order_map_;
  std::unordered_map<const LibertyPort*, utils::cuda::TwoDimensionalLUTPair***> output_port_to_internal_luts_indexed_by_order_map_;
  std::unordered_map<const LibertyCell*, LeakagePowerData> cell_to_leakage_power_data_;
  std::unordered_map<const LibertyCell*, InternalPowerData> cell_to_internal_power_data_;

  PowerVal *gate_leakage_powers_;
  PowerVal *gate_internal_powers_;
  PowerVal *gate_glitch_internal_powers_;
  PowerVal *per_cycle_leakage_powers_;
  PowerVal *per_cycle_internal_powers_;
  PowerVal *per_cycle_glitch_internal_powers_;
  PowerVal *h_gate_leakage_powers_;
  PowerVal *h_gate_internal_powers_;
  PowerVal *h_gate_glitch_internal_powers_;
  PowerVal *h_per_cycle_leakage_powers_;
  PowerVal *h_per_cycle_internal_powers_;
  PowerVal *h_per_cycle_glitch_internal_powers_;
  //--------------------end of for leakage and internal-----------------------

  //--------------------for switching-----------------------
  PowerVal *gate_switching_powers_;
  PowerVal *gate_glitch_switching_powers_;
  PowerVal *per_cycle_switching_powers_;
  PowerVal *per_cycle_glitch_switching_powers_;
  PowerVal *h_gate_switching_powers_;
  PowerVal *h_gate_glitch_switching_powers_;
  PowerVal *h_per_cycle_switching_powers_;
  PowerVal *h_per_cycle_glitch_switching_powers_;
  //--------------------end of for switching-----------------------

};
}  // end of namespace sta
