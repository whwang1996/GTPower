#pragma once
#include <filesystem>
#include "GlobalConfig.hh"

namespace utils {

void inline create_log_and_res_dir() {
  std::filesystem::create_directories(G_CONFIG.paths.log_dir);
  std::filesystem::create_directories(G_CONFIG.paths.result_dir);
}

std::string inline get_power_analysis_per_cycle_waveform_res_path() {
  return G_CONFIG.paths.result_dir / std::filesystem::path(
    G_CONFIG.flags.enable_cuda_power_analysis ? G_CONFIG.paths.cuda_power_analysis_per_cycle_waveform_res_file : G_CONFIG.paths.cpu_power_analysis_per_cycle_waveform_res_file
  );
}

std::string inline get_power_analysis_cell_res_path() {
  return G_CONFIG.paths.result_dir / std::filesystem::path(
    G_CONFIG.flags.enable_cuda_power_analysis ? G_CONFIG.paths.cuda_power_analysis_cell_res_file : G_CONFIG.paths.cpu_power_analysis_cell_res_file
  );
}

}  // namespace utils
