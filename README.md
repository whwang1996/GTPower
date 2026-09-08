# GTPower

### GPU-Accelerated Gate-Level Time-Based Power Analysis

**GTPower** is an open-source GPU-accelerated gate-level time-based power analysis tool built on top of [OpenSTA](https://github.com/parallaxsw/OpenSTA).

This repository provides the source code for the following paper:

> **GPU-Accelerated Gate-Level Time-Based Power Analysis via Event-Density-Aware Partitioning and Kernel Fusion**  
> Weihao Wang, Yikang Ouyang, Hongyuan Liu, and Yuzhe Ma  
> *ACM Transactions on Architecture and Code Optimization (TACO)*

## Key Features

- GPU-accelerated gate-level time-based power analysis.
- Waveform-driven analysis using VCD and compressed VCD activity data, with optional FSDB support.
- Internal, switching, leakage, and glitch power calculation.
- Aggregate, per-cell, and per-cycle power results.
- Event-density-aware CUDA workload partitioning.
- Adaptive cycle-per-thread selection for irregular switching activity.
- Fused dynamic and leakage power kernels.
- Cycle-based and event-based CUDA workload partitioning.
- Memory-bounded processing of large switching-activity traces.
- Multi-threaded CPU reference implementation.
- Integration with the OpenSTA timing engine.

## Repository Structure

```text
GTPower/
├── app/                    # Executable entry point
├── power/                  # Power-analysis implementation
│   ├── TimeBasedPower.cc   # CPU time-based power analysis
│   ├── ReadVcdActivities.cc
│   ├── VcdReader.cc
│   ├── FsdbReader.cc
│   └── cuda/               # CUDA power-analysis kernels
├── util/cuda/              # Shared CUDA utilities
├── test/OpenSTA-sample/    # Small VCD-based example
├── examples/               # OpenSTA examples
└── CMakeLists.txt
```

The remaining directories contain the OpenSTA timing engine and its supporting infrastructure.

## Supported Inputs

A power-analysis run typically requires the following design files:

- Liberty timing and power library (`.lib`)
- Gate-level Verilog netlist
- Timing constraints (`.sdc`)
- Parasitic information (`.spef`)
- Switching-activity waveform:
  - VCD (`.vcd`)
  - Compressed VCD (`.vcd.gz`)
  - FSDB (`.fsdb`, requires FSDB support enabled at build time)

The activity-file parser is selected automatically from the filename extension.

## Requirements

The current build flow targets Linux and requires:

- A C++17-compatible compiler
- CMake 3.24 or later
- NVIDIA CUDA Toolkit
- OpenMP
- Tcl
- SWIG 3.0 or later
- Flex
- Bison
- Eigen3
- CUDD
- zlib

The experiments reported in the paper used GCC 11.4.0 to compile the C++ code and NVCC 12.4 to compile the CUDA code.

### Optional FSDB support

FSDB support is disabled by default. To enable it, set `FSDB_READER_DIR` during CMake configuration to the FSDB Reader directory provided by your local Verdi installation, then build GTPower. The directory must contain `ffrAPI.h`, with the `nffr` and `nsys` libraries in its `linux64/` subdirectory. Configuration fails if the supplied directory is missing the header or either library.

GTPower uses the reader libraries for multithreaded FSDB reading, with the thread count controlled by `-multi_thread_number`. Builds without FSDB support accept VCD and compressed VCD activity files and do not require Verdi or its reader SDK. Reading an FSDB file with such a build produces an error explaining how to enable support.

The FSDB reader SDK is not included in this repository and may be subject to separate licensing terms.

## Building

From the repository root, configure a new build directory without FSDB support:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDD_DIR=/path/to/cudd
```

Alternatively, enable FSDB support by supplying your Verdi FSDB Reader directory:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDD_DIR=/path/to/cudd \
  -DFSDB_READER_DIR=/path/to/verdi/share/FsdbReader
```

Replace the placeholder paths with your actual installation paths. `FSDB_READER_DIR` is cached by CMake: to disable FSDB in an existing build directory, reconfigure with `-DFSDB_READER_DIR=`. Omitting the argument preserves any previously cached path. Rebuild after changing this setting.

By default, new builds target the local GPU (`native`). To select an architecture explicitly, add `-DCMAKE_CUDA_ARCHITECTURES=86` to the configuration command, replacing `86` with your target architecture. Existing cached architecture settings are preserved.

After either configuration, build the project:

```bash
cmake --build build --parallel
```

The resulting executable is generated at:

```text
app/sta
```

Check the executable version with:

```bash
./app/sta -version
```

## Quick Start

A small VCD-based GCD example is provided under:

```text
test/OpenSTA-sample/
```

Run the example from the repository root:

```bash
mkdir -p test/OpenSTA-sample/res

cd test/OpenSTA-sample

../../app/sta \
  -exit \
  -max_event_num 1000000 \
  -result_dir ./res \
  -multi_thread_number 4 \
  -cuda_device_id 0 \
  -cuda_thread_partition_basis cycle \
  -n_cycle_per_thread 1 \
  -n_event_per_thread_for_all_pins 64 \
  -n_cycle_auto_selection_e_target 8 \
  -n_cycle_auto_selection_parallelism_floor 512 \
  -bsim_pin_threshold 16 \
  ./power_vcd.tcl \
  > ./res/run.log 2>&1
```

Replace `-cuda_device_id 0` if a different GPU should be used.

The example Tcl flow performs the following operations:

```tcl
read_liberty ./data/sky130hd_tt.lib
read_verilog ./data/gcd_sky130hd.v
link_design gcd

read_sdc ./data/gcd_sky130hd.sdc
read_spef ./data/gcd_sky130hd.spef

read_power_activities \
  -scope gcd_tb/gcd1 \
  -vcd ./data/gcd_sky130hd.vcd

report_power
```

Although the command-line flag is named `-vcd`, the implementation accepts `.vcd`, `.vcd.gz`, and, when FSDB support is enabled at build time, `.fsdb` files. The corresponding activity reader is selected from the filename extension.

## Output Files

For a CUDA run, the result directory contains:

```text
res/
├── run.log
├── cuda_cell_power.txt
└── cuda_per_cycle_waveform.txt
```

### Aggregate power results

`run.log` contains aggregate results such as:

- Total internal power
- Total switching power
- Leakage power
- Total glitch power
- Glitch internal power
- Glitch switching power
- Runtime and configuration information

### Per-cell power results

`cuda_cell_power.txt` contains a tab-separated per-cell power breakdown with the following columns:

```text
cell
Internal_Power
Glitch_Internal_Power
Switching_Power
Glitch_Switching_Power
Leakage_Power
Total_Power
Standard_Cell
```

### Per-cycle power waveforms

`cuda_per_cycle_waveform.txt` contains per-cycle waveforms for:

- Leakage power
- Internal power
- Glitch internal power
- Switching power
- Glitch switching power

When CUDA power analysis is disabled, the corresponding output files are:

```text
cpu_cell_power.txt
cpu_per_cycle_waveform.txt
```

## Running the CPU Implementation

The multi-threaded CPU reference implementation can be enabled with:

```text
-disable_cuda_power_analysis
```

For example, run the included GCD design from the repository root:

```bash
mkdir -p test/OpenSTA-sample/res-cpu

cd test/OpenSTA-sample

../../app/sta \
  -exit \
  -max_event_num 1000000 \
  -result_dir ./res-cpu \
  -disable_cuda_power_analysis \
  -multi_thread_number 4 \
  -cuda_device_id 0 \
  -cuda_thread_partition_basis cycle \
  -n_cycle_per_thread 1 \
  -n_event_per_thread_for_all_pins 64 \
  -n_cycle_auto_selection_e_target 8 \
  -n_cycle_auto_selection_parallelism_floor 512 \
  -bsim_pin_threshold 16 \
  ./power_vcd.tcl \
  > ./res-cpu/run.log 2>&1
```

`-multi_thread_number` controls the number of host threads used by the CPU implementation and activity-processing stages.

The current build configuration still requires CUDA when building the CPU implementation. The FSDB reader SDK is required only when FSDB support is enabled; this setting applies to both CPU and CUDA power-analysis runs.

## Important Command-Line Options

Among the options below, only `-max_event_num` is required for both CPU and CUDA power-analysis runs. All other options are optional and use their defaults when omitted. The examples above explicitly set some optional values; see the table below for the defaults.

| Option | Requirement | Description |
|---|---|---|
| `-max_event_num <N>` | Required | Event budget used for memory-bounded activity processing |
| `-result_dir <path>` | Optional | Directory used for generated result files (default: `./res`) |
| `-disable_cuda_power_analysis` | Optional | Run the multi-threaded CPU implementation |
| `-multi_thread_number <N>` | Optional | Number of CPU worker threads (default: `16`) |
| `-cuda_device_id <id>` | Optional | CUDA device selected for power analysis (default: last visible GPU) |
| `-cuda_thread_partition_basis cycle\|event` | Optional | CUDA workload-partitioning strategy (default: `cycle`) |
| `-n_cycle_per_thread <N>` | Optional | Static number of cycles assigned to each CUDA thread (default: `8`) |
| `-n_event_per_thread_for_all_pins <N>` | Optional | Event-based thread-work configuration (default: `32`) |
| `-disable_n_cycle_auto_selection` | Optional | Disable event-density-aware cycle selection |
| `-n_cycle_auto_selection_e_target <N>` | Optional | Target event count used by event-density-aware partitioning (default: `8`) |
| `-n_cycle_auto_selection_parallelism_floor <N>` | Optional | Minimum parallelism target for sparse workloads (default: `512`) |
| `-bsim_pin_threshold <N>` | Optional | Pin-count threshold for state-indexed power lookup (default: `16`) |
| `-disable_fusion` | Optional | Use separate dynamic and leakage CUDA kernels |
| `-report_circuit_stat` | Optional | Report circuit and gate statistics |
| `-report_vcd_stat` | Optional | Report switching-activity statistics |
| `-report_cuda_power_thread_alloc_stat` | Optional | Report CUDA thread-allocation statistics |

## Citation

If you use GTPower in your research, please cite:

```bibtex
@article{wangGTPower,
  author  = {Weihao Wang and Yikang Ouyang and Hongyuan Liu and Yuzhe Ma},
  title   = {{GPU-Accelerated Gate-Level Time-Based Power Analysis via Event-Density-Aware Partitioning and Kernel Fusion}},
  journal = {ACM Transactions on Architecture and Code Optimization},
  note    = {Accepted for publication}
}
```

The publication year, volume, issue, page numbers, and DOI can be added once the final publication metadata becomes available.

## Acknowledgments

GTPower is derived from [OpenSTA](https://github.com/parallaxsw/OpenSTA), developed by Parallax Software.

The original OpenSTA copyright and license notices are retained in the source tree. Issues related specifically to the GPU time-based power-analysis extensions should be reported in this repository. General OpenSTA questions should be directed to the upstream project.

## License

GTPower is distributed under the GNU General Public License version 3. See [LICENSE](LICENSE) for details.

Third-party components retain their respective licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for copyright notices and full license texts.

OpenSTA and third-party datasets, timing libraries, waveform files, FSDB reader libraries, and commercial tools may have their own copyright or redistribution conditions. Users are responsible for complying with all applicable licenses.
