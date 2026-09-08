#pragma once
#include <string>
#include <unordered_map>
#include <filesystem>

#include "Singleton.hh"
#include "Types.hh"
#include "Defines.hh"

namespace common {
class GlobalConfigs : public utils::Singleton<common::GlobalConfigs> {
  public:
  GlobalConfigs(token) { }

  public:
  struct Flags {
    bool enable_cuda_power_analysis = true;
    const bool enable_time_based_analysis = true;
    const bool enable_multi_threaded_cpu = true;
    const bool enable_multiple_rounds_power_analysis = true;
    const bool iterate_all_events_to_do_time_slicing = false;
    const bool partition_unit_is_cycle = true;  // workload partition unit (event-equalized chunking and GPU thread workload assignment), whether partition by cycle (true) or VCD time unit (false)
    const bool time_slicing_by_activity_file_reading = true;
    const bool overlapping_get_gate_waveform_range = true;
    bool report_vcd_stat = false;
    bool report_circuit_stat = false;
    bool report_cuda_power_thread_alloc_stat = false;
    bool cuda_power_separate_kernels_for_dyn_and_leak = false;  // true: naive baseline, 1T1E dynamic kernel + 1T1C leakage kernel, false: our proposed fusion kernel
    bool enable_auto_select_n_cycle_per_thread_for_each_gate = true;
  };

  struct Nums {
    NEeventVal max_event_num = 1000000000; // 1e9 events for about 10GB memory
    int n_event_per_thread_for_all_pins = 32;
    int n_cycle_per_thread = 8;
    const int n_thread_per_block_for_all_pins = 128;
    //--------auto n_cycle selection for each gate--------
    int n_cycle_auto_selection_parallelism_floor = n_thread_per_block_for_all_pins * 4;  // 稀疏时希望大约这么多线程, n_thread_per_block * 4
    int n_cycle_auto_selection_e_target= 8;
    //--------end of auto n_cycle selection for each gate--------
    const int power_res_length_for_each_gate = n_thread_per_block_for_all_pins;
    int max_n_pin_for_leakage_power = 16;
    int max_n_pin_for_internal_power = 16;
    int multi_thread_number = 16;
    int cuda_device_id = INVALID_DEVICE_ID;
  };

  struct Strs {
    std::string activity_file_format = "fsdb";  // fsdb or vcd
    const std::string leakage_power_separator = " ";
    const std::string internal_power_separator = "&";
    const std::string power_analysis_res_file_separator = "\t";
    std::string cuda_thread_partition_basis = "cycle";  // IMPORTANT!!! whether partition workload by cycle or event
  };

  struct Paths {
    std::filesystem::path result_dir = "./res";  // relative path of tcl script
    const std::filesystem::path log_dir = "./log";  // relative path of tcl script
    const std::filesystem::path cuda_power_analysis_per_cycle_waveform_res_file = "cuda_per_cycle_waveform.txt";
    const std::filesystem::path cuda_power_analysis_cell_res_file = "cuda_cell_power.txt";
    const std::filesystem::path cpu_power_analysis_per_cycle_waveform_res_file = "cpu_per_cycle_waveform.txt";
    const std::filesystem::path cpu_power_analysis_cell_res_file = "cpu_cell_power.txt";
  };

  Flags flags;
  Nums nums;
  Strs strs;
  Paths paths;
};

#define G_CONFIG common::GlobalConfigs::instance()
}
