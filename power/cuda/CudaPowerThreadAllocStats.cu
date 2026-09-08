#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "Log.hh"
#include "CudaPowerThreadAllocStats.hh"
#include "Gate.cuh"

namespace sta {
namespace power {

// ----------------------------- numeric helpers -----------------------------

template <typename T>
static inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

static inline double safe_log2_k(size_t k) {
  return 1.0 + std::log2(static_cast<double>(std::max<size_t>(k, 1)));
}

static inline double padding_ratio_of(const ThreadAllocSummary& s) {
  return (s.total_launched_threads > 0)
    ? (static_cast<double>(s.total_padding_threads) / static_cast<double>(s.total_launched_threads))
    : 0.0;
}

static inline double percentile_inplace_sorted(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  if (p <= 0.0) return sorted.front();
  if (p >= 100.0) return sorted.back();
  const double pos = (p / 100.0) * (static_cast<double>(sorted.size() - 1));
  const size_t i = static_cast<size_t>(std::floor(pos));
  const size_t j = std::min(i + 1, sorted.size() - 1);
  const double frac = pos - static_cast<double>(i);
  return sorted[i] * (1.0 - frac) + sorted[j] * frac;
}

static inline double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return percentile_inplace_sorted(v, p);
}

// ----------------------------- auto n_cycle_per_thread (gate-specific) -----------------------------

static inline NPeriodVal compute_n_cycle_per_thread_by_event_count(
  const sta::power::Gate* gate,
  const NPeriodVal n_cycle,
  const NThreadVal threads_target,
  const NEeventVal E_target
) {
  const NEeventVal n_event = static_cast<NEeventVal>(
    std::accumulate(gate->pin_waveform_sizes,
                    gate->pin_waveform_sizes + gate->n_pin,
                    static_cast<size_t>(0))
  );

  if (n_cycle <= 0) return static_cast<NPeriodVal>(1);

  NPeriodVal n_cycle_target = (n_cycle + threads_target - 1) / threads_target;
  n_cycle_target = std::max(n_cycle_target, static_cast<NPeriodVal>(1));

  const double rho = static_cast<double>(n_event) / static_cast<double>(n_cycle);
  const double rho_floor = static_cast<double>(E_target) / static_cast<double>(n_cycle_target);
  const double rho_eff = rho + rho_floor;

  NPeriodVal n = static_cast<NPeriodVal>(llround(static_cast<double>(E_target) / rho_eff));
  n = std::min(n, n_cycle);
  n = std::max(n, static_cast<NPeriodVal>(1));
  return n;
}

// ----------------------------- collect stats -----------------------------

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
) {
  ThreadAllocSummary out;

  const NPeriodVal n_cycle_interval =
    (interval_end_time / vcd_time_unit_per_cycle) - (interval_start_time / vcd_time_unit_per_cycle);
  out.n_cycle_interval = n_cycle_interval;

  const NEeventVal sparse_thr =
    (sparse_event_threshold_override >= 0) ? sparse_event_threshold_override : E_target;
  out.sparse_event_threshold = sparse_thr;

  std::vector<double> dist_EperT;
  std::vector<double> dist_TperG;
  std::vector<double> dist_n_event;
  dist_EperT.reserve(gates.size());
  dist_TperG.reserve(gates.size());
  dist_n_event.reserve(gates.size());

  for (NGateVal gate_idx = 0; gate_idx < static_cast<NGateVal>(gates.size()); ++gate_idx) {
    const sta::power::Gate* g = gates[gate_idx];

    const size_t n_event_sz = std::accumulate(
      g->pin_waveform_sizes,
      g->pin_waveform_sizes + g->n_pin,
      static_cast<size_t>(0)
    );
    const NEeventVal n_event = static_cast<NEeventVal>(n_event_sz);

    double W_search_gate = 0.0;
    for (NPinVal i = 0; i < g->n_pin; ++i) {
      W_search_gate += safe_log2_k(g->pin_waveform_sizes[i]);
    }
    const double W_search_per_thread = 2.0 * W_search_gate;

    NPeriodVal n_cycle_per_thread = 1;
    if (enable_auto_select_n_cycle_per_thread) {
      n_cycle_per_thread = compute_n_cycle_per_thread_by_event_count(
        g, n_cycle_interval, threads_target_sparse, E_target
      );
    } else {
      n_cycle_per_thread = std::max(global_n_cycle_per_thread, static_cast<NPeriodVal>(1));
    }

    const NThreadVal n_active_threads =
      static_cast<NThreadVal>(ceil_div(n_cycle_interval, n_cycle_per_thread));
    const NBlockVal n_blocks =
      static_cast<NBlockVal>(ceil_div(n_active_threads, n_thread_per_block));
    const NThreadVal n_launched_threads =
      static_cast<NThreadVal>(n_blocks * n_thread_per_block);
    const NThreadVal n_padding_threads =
      static_cast<NThreadVal>(n_launched_threads - n_active_threads);

    const double avg_events_per_active_thread =
      (n_active_threads > 0) ? (static_cast<double>(n_event) / static_cast<double>(n_active_threads)) : 0.0;

    const double search_overhead_active =
      static_cast<double>(n_active_threads) * W_search_per_thread;
    const double search_overhead_launched =
      static_cast<double>(n_launched_threads) * W_search_per_thread;

    out.total_active_threads += n_active_threads;
    out.total_launched_threads += n_launched_threads;
    out.total_padding_threads += n_padding_threads;
    out.total_blocks += n_blocks;

    out.H_search_active += search_overhead_active;
    out.H_search_launched += search_overhead_launched;

    dist_EperT.push_back(avg_events_per_active_thread);
    dist_TperG.push_back(static_cast<double>(n_active_threads));
    dist_n_event.push_back(static_cast<double>(n_event));

    if (n_event <= sparse_thr) {
      out.n_sparse_gates += 1;
      out.sparse_total_active_threads += n_active_threads;
      out.sparse_H_search_active += search_overhead_active;
    }
  }

  out.p50_avg_events_per_active_thread = percentile(dist_EperT, 50.0);
  out.p90_avg_events_per_active_thread = percentile(dist_EperT, 90.0);
  out.p99_avg_events_per_active_thread = percentile(dist_EperT, 99.0);

  out.p50_active_threads_per_gate = percentile(dist_TperG, 50.0);
  out.p90_active_threads_per_gate = percentile(dist_TperG, 90.0);
  out.p99_active_threads_per_gate = percentile(dist_TperG, 99.0);

  out.p50_n_event_per_gate = percentile(dist_n_event, 50.0);
  out.p90_n_event_per_gate = percentile(dist_n_event, 90.0);
  out.p99_n_event_per_gate = percentile(dist_n_event, 99.0);

  return out;
}

// ----------------------------- logging -----------------------------

void logThreadAllocSummaryRound(
  const ThreadAllocSummary& s,
  const std::string& tag,
  const VcdEventTime interval_start_time,
  const VcdEventTime interval_end_time
) {
  LOG_INFO
    << "ThreadAllocRound"
    << " tag=" << tag
    << " interval_start_time=" << interval_start_time
    << " interval_end_time=" << interval_end_time
    << " n_cycle_interval=" << s.n_cycle_interval
    << " blocks=" << s.total_blocks
    << " active_threads=" << s.total_active_threads
    << " launched_threads=" << s.total_launched_threads
    << " padding_threads=" << s.total_padding_threads
    << " padding_ratio=" << padding_ratio_of(s)
    << " H_search_active=" << s.H_search_active
    << " H_search_launched=" << s.H_search_launched
    << " n_event_p50=" << s.p50_n_event_per_gate
    << " n_event_p90=" << s.p90_n_event_per_gate
    << " n_event_p99=" << s.p99_n_event_per_gate
    << " EperT_p50=" << s.p50_avg_events_per_active_thread
    << " EperT_p90=" << s.p90_avg_events_per_active_thread
    << " EperT_p99=" << s.p99_avg_events_per_active_thread
    << " TperG_p50=" << s.p50_active_threads_per_gate
    << " TperG_p90=" << s.p90_active_threads_per_gate
    << " TperG_p99=" << s.p99_active_threads_per_gate
    << " sparse_thr=" << s.sparse_event_threshold
    << " sparse_gates=" << s.n_sparse_gates
    << " sparse_active_threads=" << s.sparse_total_active_threads
    << " sparse_H_search_active=" << s.sparse_H_search_active;
}

}  // namespace power
}  // namespace sta
