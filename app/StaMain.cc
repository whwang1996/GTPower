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

#include "StaMain.hh"

#include <tcl.h>
#include <cstdlib>
#include <sys/stat.h>

#include "Machine.hh"
#include "StringUtil.hh"
#include "Vector.hh"
#include "Sta.hh"
#include "Log.hh"
#include "GlobalConfig.hh"

namespace sta {

void
PrintArguments(int argc, char* argv[]) 
{
  LOG_BEGIN(INFO, "Program Arguments");

  LOG_INFO << "Program: " << argv[0];
  LOG_INFO << "Number of arguments: " << argc - 1;
  for (int i = 1; i < argc; ++i) {
    LOG_INFO << "Argument " << i << ": " << argv[i];
  }

  LOG_END(INFO, "Program Arguments");
}

void
parseMyPowerArgs(int &argc,
		char *argv[])
{
  LOG_BEGIN(INFO, "Setting my power analysis args");
  //--------------------------------------------result_dir--------------------------------------------
  char* result_dir_arg = findCmdLineKey(argc, argv, "-result_dir");
  if (result_dir_arg) {
    G_CONFIG.paths.result_dir = result_dir_arg;
  }
  //--------------------------------------------end of result_dir--------------------------------------------

  //--------------------------------------------disable_cuda_power_analysis--------------------------------------------
  if (findCmdLineFlag(argc, argv, "-disable_cuda_power_analysis")) {
    G_CONFIG.flags.enable_cuda_power_analysis = false;
    LOG_INFO << "Disable CUDA power analysis.";
  }
  //--------------------------------------------end of disable_cuda_power_analysis--------------------------------------------

  //--------------------------------------------cuda_thread_partition_basis--------------------------------------------
  char* cuda_thread_partition_basis_arg = findCmdLineKey(argc, argv, "-cuda_thread_partition_basis");
  if (cuda_thread_partition_basis_arg) {
    std::string cuda_thread_partition_basis = std::string(cuda_thread_partition_basis_arg);
    if (cuda_thread_partition_basis != "event" && cuda_thread_partition_basis != "cycle") {
      LOG_ERROR << "Unknown cuda_thread_partition_basis: " << cuda_thread_partition_basis;
    } else {
      G_CONFIG.strs.cuda_thread_partition_basis = cuda_thread_partition_basis;
      LOG_INFO << "Setting cuda_thread_partition_basis: " << G_CONFIG.strs.cuda_thread_partition_basis;
    }
  }
  //--------------------------------------------end of cuda_thread_partition_basis--------------------------------------------

  //--------------------------------------------n_cycle_per_thread--------------------------------------------
  char* n_cycle_per_thread_arg = findCmdLineKey(argc, argv, "-n_cycle_per_thread");
  if (n_cycle_per_thread_arg) {
    if (isDigits(n_cycle_per_thread_arg)) {
      G_CONFIG.nums.n_cycle_per_thread = atoi(n_cycle_per_thread_arg);
      LOG_INFO << "Setting n_cycle_per_thread: " << G_CONFIG.nums.n_cycle_per_thread;
    } else {
      LOG_ERROR << "Unknown n_cycle_per_thread_arg: " << n_cycle_per_thread_arg;
    }
  }
  //--------------------------------------------end of n_cycle_per_thread--------------------------------------------

  //--------------------------------------------n_event_per_thread_for_all_pins--------------------------------------------
  char* n_event_per_thread_for_all_pins_arg = findCmdLineKey(argc, argv, "-n_event_per_thread_for_all_pins");
  if (n_event_per_thread_for_all_pins_arg) {
    if (isDigits(n_event_per_thread_for_all_pins_arg)) {
      G_CONFIG.nums.n_event_per_thread_for_all_pins = atoi(n_event_per_thread_for_all_pins_arg);
      LOG_INFO << "Setting n_event_per_thread_for_all_pins: " << G_CONFIG.nums.n_event_per_thread_for_all_pins;
    } else {
      LOG_ERROR << "Unknown n_event_per_thread_for_all_pins_arg: " << n_event_per_thread_for_all_pins_arg;
    }
  }
  //--------------------------------------------end of n_event_per_thread_for_all_pins--------------------------------------------

  //--------------------------------------------max_event_num--------------------------------------------
  char* max_event_num_arg = findCmdLineKey(argc, argv, "-max_event_num");
  if (max_event_num_arg) {
    if (isDigits(max_event_num_arg)) {
      G_CONFIG.nums.max_event_num = atoi(max_event_num_arg);
      LOG_INFO << "Setting max_event_num: " << G_CONFIG.nums.max_event_num;
    } else {
      LOG_ERROR << "Unknown max_event_num_arg: " << max_event_num_arg;
    }
  } else {
    LOG_ERROR << "No max_event_num_arg";
  }
  //--------------------------------------------end of max_event_num--------------------------------------------

  //--------------------------------------------bsim_pin_threshold--------------------------------------------
  char* bsim_pin_threshold_arg = findCmdLineKey(argc, argv, "-bsim_pin_threshold");
  if (bsim_pin_threshold_arg) {
    if (isDigits(bsim_pin_threshold_arg)) {
      const int bsim_pin_threshold = atoi(bsim_pin_threshold_arg);
      G_CONFIG.nums.max_n_pin_for_leakage_power = bsim_pin_threshold;
      G_CONFIG.nums.max_n_pin_for_internal_power = bsim_pin_threshold;
      LOG_INFO << "Setting bsim_pin_threshold: " << bsim_pin_threshold;
    } else {
      LOG_ERROR << "Unknown bsim_pin_threshold_arg: " << bsim_pin_threshold_arg;
    }
  }
  //--------------------------------------------end of bsim_pin_threshold--------------------------------------------
  
  //--------------------------------------------multi_thread_number--------------------------------------------
  char* multi_thread_number_arg = findCmdLineKey(argc, argv, "-multi_thread_number");
  if (multi_thread_number_arg) {
    if (isDigits(multi_thread_number_arg)) {
      G_CONFIG.nums.multi_thread_number = atoi(multi_thread_number_arg);
      LOG_INFO << "Setting multi_thread_number: " << G_CONFIG.nums.multi_thread_number;
    } else {
      LOG_ERROR << "Unknown multi_thread_number_arg: " << multi_thread_number_arg;
    }
  }
  //--------------------------------------------end of multi_thread_number--------------------------------------------

  //--------------------------------------------cuda_device_id--------------------------------------------
  char* cuda_device_id_arg = findCmdLineKey(argc, argv, "-cuda_device_id");
  if (cuda_device_id_arg) {
    if (isDigits(cuda_device_id_arg)) {
      G_CONFIG.nums.cuda_device_id = atoi(cuda_device_id_arg);
      LOG_INFO << "Setting cuda_device_id: " << G_CONFIG.nums.cuda_device_id;
    } else {
      LOG_ERROR << "Unknown cuda_device_id_arg: " << cuda_device_id_arg;
    }
  }
  //--------------------------------------------end of cuda_device_id--------------------------------------------

  //--------------------------------------------n_cycle_auto_selection_e_target--------------------------------------------
  char* n_cycle_auto_selection_e_target_arg = findCmdLineKey(argc, argv, "-n_cycle_auto_selection_e_target");
  if (n_cycle_auto_selection_e_target_arg) {
    if (isDigits(n_cycle_auto_selection_e_target_arg)) {
      G_CONFIG.nums.n_cycle_auto_selection_e_target = atoi(n_cycle_auto_selection_e_target_arg);
      LOG_INFO << "Setting n_cycle_auto_selection_e_target: " << G_CONFIG.nums.n_cycle_auto_selection_e_target;
    } else {
      LOG_ERROR << "Unknown n_cycle_auto_selection_e_target_arg: " << n_cycle_auto_selection_e_target_arg;
    }
  }
  //--------------------------------------------end of n_cycle_auto_selection_e_target--------------------------------------------

  //--------------------------------------------n_cycle_auto_selection_parallelism_floor--------------------------------------------
  char* n_cycle_auto_selection_parallelism_floor_arg = findCmdLineKey(argc, argv, "-n_cycle_auto_selection_parallelism_floor");
  if (n_cycle_auto_selection_parallelism_floor_arg) {
    if (isDigits(n_cycle_auto_selection_parallelism_floor_arg)) {
      const int n_cycle_auto_selection_parallelism_floor = atoi(n_cycle_auto_selection_parallelism_floor_arg);
      if (n_cycle_auto_selection_parallelism_floor > 0) {
        G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor = n_cycle_auto_selection_parallelism_floor;
        LOG_INFO << "Setting n_cycle_auto_selection_parallelism_floor: " << G_CONFIG.nums.n_cycle_auto_selection_parallelism_floor;
      } else {
        LOG_ERROR << "n_cycle_auto_selection_parallelism_floor must be greater than 0: " << n_cycle_auto_selection_parallelism_floor_arg;
      }
    } else {
      LOG_ERROR << "Unknown n_cycle_auto_selection_parallelism_floor_arg: " << n_cycle_auto_selection_parallelism_floor_arg;
    }
  }
  //--------------------------------------------end of n_cycle_auto_selection_parallelism_floor--------------------------------------------

  //--------------------------------------------disable n_cycle auto selection--------------------------------------------
  if (findCmdLineFlag(argc, argv, "-disable_n_cycle_auto_selection")) {
    G_CONFIG.flags.enable_auto_select_n_cycle_per_thread_for_each_gate = false;
    LOG_INFO << "Disable n_cycle_per_thread auto selection.";
  }
  //--------------------------------------------end of disable n_cycle auto selection--------------------------------------------

  //--------------------------------------------disable fusion--------------------------------------------
  if (findCmdLineFlag(argc, argv, "-disable_fusion")) {
    G_CONFIG.flags.cuda_power_separate_kernels_for_dyn_and_leak = true;
    LOG_INFO << "Disable kernel fusion.";
  }
  //--------------------------------------------end of disable fusion--------------------------------------------

  //--------------------------------------------report vcd stat--------------------------------------------
  if (findCmdLineFlag(argc, argv, "-report_vcd_stat")) {
    G_CONFIG.flags.report_vcd_stat = true;
    LOG_INFO << "Report VCD statistics.";
  }
  //--------------------------------------------end of report vcd stat--------------------------------------------

  //--------------------------------------------report circuit stat--------------------------------------------
  if (findCmdLineFlag(argc, argv, "-report_circuit_stat")) {
    G_CONFIG.flags.report_circuit_stat = true;
    LOG_INFO << "Report circuit statistics.";
  }
  //--------------------------------------------end of report circuit stat--------------------------------------------

  //--------------------------------------------report cuda power thread alloc stat--------------------------------------------
  if (findCmdLineFlag(argc, argv, "-report_cuda_power_thread_alloc_stat")) {
    G_CONFIG.flags.report_cuda_power_thread_alloc_stat = true;
    LOG_INFO << "Report CUDA power thread allocation statistics.";
  }
  //--------------------------------------------end of report cuda power thread alloc stat--------------------------------------------

  LOG_END(INFO, "Setting my power analysis args");
}

int
parseThreadsArg(int &argc,
		char *argv[])
{
  char *thread_arg = findCmdLineKey(argc, argv, "-threads");
  if (thread_arg) {
    if (stringEqual(thread_arg, "max"))
      return processorCount();
    else if (isDigits(thread_arg))
      return atoi(thread_arg);
    else
      fprintf(stderr,"Warning: -threads must be max or a positive integer.\n");
  }
  return 1;
}

bool
findCmdLineFlag(int &argc,
		char *argv[],
		const char *flag)
{
  for (int i = 1; i < argc; i++) {
    char *arg = argv[i];
    if (stringEq(arg, flag)) {
      // Remove flag from argv.
      for (int j = i + 1; j < argc; j++, i++)
	argv[i] = argv[j];
      argc--;
      argv[argc] = nullptr;
      return true;
    }
  }
  return false;
}

char *
findCmdLineKey(int &argc,
	       char *argv[],
	       const char *key)
{
  for (int i = 1; i < argc; i++) {
    char *arg = argv[i];
    if (stringEq(arg, key) && i + 1 < argc) {
      char *value = argv[i + 1];
      // Remove key and value from argv.
      for (int j = i + 2; j < argc; j++, i++)
	argv[i] = argv[j];
      argc -= 2;
      argv[argc] = nullptr;
      return value;
    }
  }
  return nullptr;
}

// Use overridden version of source to echo cmds and results.
int
sourceTclFile(const char *filename,
	      bool echo,
	      bool verbose,
	      Tcl_Interp *interp)
{
  string cmd;
  stringPrint(cmd, "source %s%s%s",
	      echo ? "-echo " : "",
	      verbose ? "-verbose " : "",
	      filename);
  int code = Tcl_Eval(interp, cmd.c_str());
  const char *result = Tcl_GetStringResult(interp);
  if (result[0] != '\0')
    printf("%s\n", result);
  return code;
}

void
evalTclInit(Tcl_Interp *interp,
	    const char *inits[])
{
  char *unencoded = unencode(inits);
  if (Tcl_Eval(interp, unencoded) != TCL_OK) {
    // Get a backtrace for the error.
    Tcl_Eval(interp, "$errorInfo");
    const char *tcl_err = Tcl_GetStringResult(interp);
    fprintf(stderr, "Error: TCL init script: %s.\n", tcl_err);
    fprintf(stderr, "       Try deleting TclInitVar.cc and rebuilding.\n");
    exit(0);
  }
  delete [] unencoded;
}

char *
unencode(const char *inits[])
{
  size_t length = 0;
  for (const char **e = inits; *e; e++) {
    const char *init = *e;
    length += strlen(init);
  }
  char *unencoded = new char[length / 3 + 1];
  char *u = unencoded;
  for (const char **e = inits; *e; e++) {
    const char *init = *e;
    size_t init_length = strlen(init);
    for (const char *s = init; s < &init[init_length]; s += 3) {
      char code[4] = {s[0], s[1], s[2], '\0'};
      char ch = atoi(code);
      *u++ = ch;
    }
  }
  *u = '\0';
  return unencoded;
}

// Hack until c++17 filesystem is better supported.
bool
is_regular_file(const char *filename)
{
  struct stat sb;
  return stat(filename, &sb) == 0 && S_ISREG(sb.st_mode);
}

} // namespace
