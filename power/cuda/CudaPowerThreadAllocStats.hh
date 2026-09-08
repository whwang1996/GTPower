#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Types.hh"
#include "Types.cuh"

namespace sta {
namespace power {

// Forward declaration. The .cc will include the real Gate header.
struct Gate;

struct ThreadAllocSummary {
  // interval cycles count
  NPeriodVal n_cycle_interval = 0;

  // totals
  NThreadVal total_active_threads = 0;
  NThreadVal total_launched_threads = 0;
  NThreadVal total_padding_threads = 0;
  NBlockVal total_blocks = 0;

  // Binary-search proxy totals:
  // H_search = Σ_g N_g * (2 * Σ_i log2(k_i))
  double H_search_active = 0.0;
  double H_search_launched = 0.0;

  // per-gate distributions (computed within each round)
  double p50_avg_events_per_active_thread = 0.0;
  double p90_avg_events_per_active_thread = 0.0;
  double p99_avg_events_per_active_thread = 0.0;

  double p50_active_threads_per_gate = 0.0;
  double p90_active_threads_per_gate = 0.0;
  double p99_active_threads_per_gate = 0.0;

  // n_event per gate distribution (across gates)
  double p50_n_event_per_gate = 0.0;
  double p90_n_event_per_gate = 0.0;
  double p99_n_event_per_gate = 0.0;

  // sparse subset stats (defined by n_event threshold)
  NEeventVal sparse_event_threshold = 0;
  size_t n_sparse_gates = 0;
  NThreadVal sparse_total_active_threads = 0;
  double sparse_H_search_active = 0.0;
};

// Collect cycle-partition thread allocation statistics for one interval.
ThreadAllocSummary collect_cycle_partition_thread_alloc_stats(
  const std::vector<sta::power::Gate*>& gates,
  VcdEventTime interval_start_time,
  VcdEventTime interval_end_time,
  VcdEventTime vcd_time_unit_per_cycle,
  NThreadVal n_thread_per_block,
  bool enable_auto_select_n_cycle_per_thread,
  NPeriodVal global_n_cycle_per_thread,
  NThreadVal threads_target_sparse,
  NEeventVal E_target,
  NEeventVal sparse_event_threshold_override
);

// Print one round summary in a script-friendly single line.
// Includes interval_start_time / interval_end_time for aligning rounds in scripts.
void logThreadAllocSummaryRound(
  const ThreadAllocSummary& s,
  const std::string& tag,
  const VcdEventTime interval_start_time,
  const VcdEventTime interval_end_time
);

}  // namespace power
}  // namespace sta
