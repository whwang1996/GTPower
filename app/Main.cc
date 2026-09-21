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
#include "StaConfig.hh"  // STA_VERSION

#include <stdio.h>
#include <cstdlib>              // exit
#include <tcl.h>
#if TCL_READLINE
  #include <tclreadline.h>
#endif

#include "Sta.hh"

namespace sta {
extern const char *tcl_inits[];
}

using std::string;
using sta::stringEq;
using sta::findCmdLineFlag;
using sta::Sta;
using sta::initSta;
using sta::evalTclInit;
using sta::sourceTclFile;
using sta::parseThreadsArg;
using sta::tcl_inits;
using sta::is_regular_file;

// Swig uses C linkage for init functions.
extern "C" {
extern int Sta_Init(Tcl_Interp *interp);
}

static int cmd_argc;
static char **cmd_argv;
static const char *init_filename = ".sta";

static void
showUsage(const char *prog,
	  const char *init_filename);
static int
tclAppInit(Tcl_Interp *interp);
static int
staTclAppInit(int argc,
	      char *argv[],
	      const char *init_filename,
	      Tcl_Interp *interp);
static void
initStaApp(int &argc,
	   char *argv[],
	   Tcl_Interp *interp);

int
main(int argc,
     char *argv[])
{
  if (argc == 2 && stringEq(argv[1], "-help")) {
    showUsage(argv[0], init_filename);
    return 0;
  }
  else if (argc == 2 && stringEq(argv[1], "-version")) {
    printf("GTPower (based on OpenSTA %s)\n", STA_VERSION);
    return 0;
  }
  else {
    // Set argc to 1 so Tcl_Main doesn't source any files.
    // Tcl_Main never returns.
#if 0
    // It should be possible to pass argc/argv to staTclAppInit with
    // a closure but I couldn't get the signature to match Tcl_AppInitProc.
    Tcl_Main(1, argv, [=](Tcl_Interp *interp)
		      { sta::staTclAppInit(argc, argv, interp);
			return 1;
		      });
#else
    // Workaround.
    cmd_argc = argc;
    cmd_argv = argv;
    Tcl_Main(1, argv, tclAppInit);
#endif
    return 0;
  }
}

static int
tclAppInit(Tcl_Interp *interp)
{
  return staTclAppInit(cmd_argc, cmd_argv, init_filename, interp);
}

// Tcl init executed inside Tcl_Main.
static int
staTclAppInit(int argc,
	      char *argv[],
	      const char *init_filename,
	      Tcl_Interp *interp)
{
  // source init.tcl
  if (Tcl_Init(interp) == TCL_ERROR)
    return TCL_ERROR;

#if TCL_READLINE
  if (Tclreadline_Init(interp) == TCL_ERROR)
    return TCL_ERROR;
  Tcl_StaticPackage(interp, "tclreadline", Tclreadline_Init, Tclreadline_SafeInit);
  if (Tcl_EvalFile(interp, TCLRL_LIBRARY "/tclreadlineInit.tcl") != TCL_OK)
    printf("Failed to load tclreadline.tcl\n");
#endif

  // Preserve the original arguments for PrintArguments.
  bool show_splash = true;
  for (int i = 1; i < argc; i++) {
    if (stringEq(argv[i], "-no_splash")) {
      show_splash = false;
      break;
    }
  }
  if (show_splash) {
    printf("%s\n", sta::splashMessage());
    fflush(stdout);
  }

  initStaApp(argc, argv, interp);
  findCmdLineFlag(argc, argv, "-no_splash");

  if (!findCmdLineFlag(argc, argv, "-no_init")) {
    const char *home = getenv("HOME");
    if (home) {
      string init_path = home;
      init_path += "/";
      init_path += init_filename;
      if (is_regular_file(init_path.c_str()))
        sourceTclFile(init_path.c_str(), true, true, interp);
    }
  }

  bool exit_after_cmd_file = findCmdLineFlag(argc, argv, "-exit");

  if (argc > 2
      || (argc > 1 && argv[1][0] == '-')) {
    showUsage(argv[0], init_filename);
    exit(1);
  }
  else {
    if (argc == 2) {
      char *cmd_file = argv[1];
      if (cmd_file) {
	int result = sourceTclFile(cmd_file, false, false, interp);
        if (exit_after_cmd_file) {
          int exit_code = (result == TCL_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
          exit(exit_code);
        }
      }
    }
  }
#if TCL_READLINE
  return Tcl_Eval(interp, "::tclreadline::Loop");
#else
  return TCL_OK;
#endif
}

static void
initStaApp(int &argc,
	   char *argv[],
	   Tcl_Interp *interp)
{
  sta::PrintArguments(argc, argv);
  initSta();
  Sta *sta = new Sta;
  Sta::setSta(sta);

  //----my args----
  sta::parseMyPowerArgs(argc, argv);
  //----end of my args----

  sta->makeComponents();
  sta->setTclInterp(interp);
  int thread_count = parseThreadsArg(argc, argv);
  sta->setThreadCount(thread_count);

  // Define swig TCL commands.
  Sta_Init(interp);
  // Eval encoded sta TCL sources.
  evalTclInit(interp, tcl_inits);
  Tcl_Eval(interp, "init_sta_cmds");
}

static void
showUsage(const char *prog,
	  const char *init_filename)
{
  printf("Usage: %s [options] [cmd_file]\n", prog);
  printf("\nGeneral options:\n");
  printf("  -help              show help and exit\n");
  printf("  -version           show version and exit\n");
  printf("  -no_init           do not read %s init file\n", init_filename);
  printf("  -threads count|max OpenSTA timing-analysis threads (default: 1)\n");
  printf("  -no_splash         do not show the license splash at startup\n");
  printf("  -exit              exit after reading cmd_file\n");
  printf("  cmd_file           source cmd_file\n");
  printf("\nPower analysis options (CUDA enabled by default):\n"
         "  -disable_cuda_power_analysis\n"
         "      run the multi-threaded CPU implementation\n"
         "  -multi_thread_number <N>\n"
         "      CPU worker threads for power analysis and FSDB reading (default: 16)\n"
         "  -result_dir <path>\n"
         "      result directory (default: ./res)\n"
         "  -max_event_num <N>\n"
         "      event budget per batch and CUDA event buffer capacity (default: 400000000)\n"
         "  -bsim_pin_threshold <N>\n"
         "      pin-count threshold for state-indexed power lookup (default: 16)\n");
  printf("\nCUDA options:\n"
         "  -cuda_device_id <id>\n"
         "      CUDA device for power analysis (default: last visible GPU)\n"
         "  -cuda_thread_partition_basis cycle|event\n"
         "      workload-partitioning strategy (default: cycle)\n"
         "  -n_cycle_per_thread <N>\n"
         "      static cycles per CUDA thread (default: 8); requires\n"
         "      -disable_n_cycle_auto_selection in CUDA mode\n"
         "  -n_event_per_thread_for_all_pins <N>\n"
         "      event-based thread-work configuration (default: 32)\n"
         "  -disable_n_cycle_auto_selection\n"
         "      disable event-density-aware cycle selection\n"
         "  -n_cycle_auto_selection_e_target <N>\n"
         "      target event count for automatic cycle selection (default: 8)\n"
         "  -n_cycle_auto_selection_parallelism_floor <N>\n"
         "      minimum parallelism target for sparse workloads (default: 512)\n"
         "  -disable_fusion\n"
         "      use separate dynamic and leakage CUDA kernels\n");
  printf("\nReporting options:\n"
         "  -report_circuit_stat\n"
         "      report circuit and gate statistics\n"
         "  -report_vcd_stat\n"
         "      report switching-activity statistics\n"
         "  -report_cuda_power_thread_alloc_stat\n"
         "      report CUDA thread-allocation statistics\n");
}
