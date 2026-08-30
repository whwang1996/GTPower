# GTPower

### GPU-Accelerated Gate-Level Time-Based Power Analysis

**GTPower** is an open-source GPU-accelerated gate-level time-based power analysis tool built on top of [OpenSTA](https://github.com/parallaxsw/OpenSTA).

This repository provides the source code for the following paper:

> **GPU-Accelerated Gate-Level Time-Based Power Analysis via Event-Density-Aware Partitioning and Kernel Fusion**  
> Weihao Wang, Yikang Ouyang, Hongyuan Liu, and Yuzhe Ma  
> *ACM Transactions on Architecture and Code Optimization (TACO)*

## Key Features

- GPU-accelerated gate-level time-based power analysis.
- Waveform-driven analysis using VCD, compressed VCD, or FSDB activity data.
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
├── postprocess/            # Result-processing utilities
├── run_experiment/         # Experiment automation scripts
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
  - FSDB (`.fsdb`)

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
- An FSDB reader SDK providing:
  - `ffrAPI.h`
  - `libnffr`
  - `libnsys`

The FSDB reader SDK is not included in this repository and may be subject to separate licensing terms.

The current CMake configuration links the FSDB reader libraries unconditionally. Therefore, the FSDB reader SDK is currently required at build time even when only VCD input is used.

## Building

From the repository root, configure the project with:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDD_DIR=/path/to/cudd \
  -DFSDB_READER_DIR=/path/to/fsdb-reader
```

`FSDB_READER_DIR` is expected to contain `ffrAPI.h`, with the corresponding libraries under:

```text
/path/to/fsdb-reader/linux64/
```

Build the project:

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
  -result_dir ./res \
  -cuda_thread_partition_basis cycle \
  -n_event_per_thread_for_all_pins 64 \
  -n_cycle_per_thread 1 \
  -n_cycle_auto_selection_e_target 8 \
  -n_cycle_auto_selection_parallelism_floor 512 \
  -bsim_pin_threshold 16 \
  -max_event_num 1000000 \
  -multi_thread_number 4 \
  -cuda_device_id 0 \
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

Although the command-line flag is named `-vcd`, the implementation accepts `.vcd`, `.vcd.gz`, and `.fsdb` files. The corresponding activity reader is selected from the filename extension.

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
  -disable_cuda_power_analysis \
  -result_dir ./res-cpu \
  -cuda_thread_partition_basis cycle \
  -n_event_per_thread_for_all_pins 64 \
  -n_cycle_per_thread 1 \
  -n_cycle_auto_selection_e_target 8 \
  -n_cycle_auto_selection_parallelism_floor 512 \
  -bsim_pin_threshold 16 \
  -max_event_num 1000000 \
  -multi_thread_number 4 \
  -cuda_device_id 0 \
  ./power_vcd.tcl \
  > ./res-cpu/run.log 2>&1
```

`-multi_thread_number` controls the number of host threads used by the CPU implementation and activity-processing stages.

The current build configuration still requires CUDA and the FSDB reader SDK when building the CPU implementation.

## Important Command-Line Options

| Option | Description |
|---|---|
| `-result_dir <path>` | Directory used for generated result files |
| `-cuda_device_id <id>` | CUDA device selected for power analysis |
| `-cuda_thread_partition_basis cycle\|event` | CUDA workload-partitioning strategy |
| `-n_cycle_per_thread <N>` | Static number of cycles assigned to each CUDA thread |
| `-n_event_per_thread_for_all_pins <N>` | Event-based thread-work configuration |
| `-n_cycle_auto_selection_e_target <N>` | Target event count used by event-density-aware partitioning |
| `-n_cycle_auto_selection_parallelism_floor <N>` | Minimum parallelism target for sparse workloads |
| `-bsim_pin_threshold <N>` | Pin-count threshold for state-indexed power lookup |
| `-max_event_num <N>` | Event budget used for memory-bounded activity processing |
| `-multi_thread_number <N>` | Number of CPU worker threads |
| `-disable_cuda_power_analysis` | Run the multi-threaded CPU implementation |
| `-disable_n_cycle_auto_selection` | Disable event-density-aware cycle selection |
| `-disable_fusion` | Use separate dynamic and leakage CUDA kernels |
| `-force_zero_slew` | Force zero slew for controlled evaluation |
| `-report_vcd_stat` | Report switching-activity statistics |
| `-report_circuit_stat` | Report circuit and gate statistics |
| `-report_cuda_power_thread_alloc_stat` | Report CUDA thread-allocation statistics |

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

OpenSTA and third-party datasets, timing libraries, waveform files, FSDB reader libraries, and commercial tools may have their own copyright or redistribution conditions. Users are responsible for complying with all applicable licenses.