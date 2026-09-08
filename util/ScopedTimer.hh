#pragma once

#include <iostream>
#include <chrono>
#include <queue>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "Timer.hh"
#include "Singleton.hh"

namespace utils {
class GlobalTimeStats : public utils::Singleton<utils::GlobalTimeStats> {
  class Node;

  private:
  thread_local static std::vector<std::pair<std::string, std::pair<double, double>>> task_stack_; 
  thread_local static Node* root_; // dummy root
  thread_local static std::string unit_;
  thread_local static size_t max_task_name_len_;
  static inline const std::string indentation_ = "  ";
  static inline const int task_name_padding_ = 1;

  void SetUnit();
  void LayerwiseSortAndResetTimeByUnit();
  void PreOrderPrint(Node* node, int depth);
  void AddRecord(double cpu_time, double wall_time);

  public:
  GlobalTimeStats(token);
  ~GlobalTimeStats();
  int Init();
  void Print();
  void StartTaskTiming(const std::string& task_name);
  void EndCurTaskTiming();
};

class ScopedTimer {
  using Time = decltype(std::chrono::steady_clock::now());

  public:
  ScopedTimer(double* wall = nullptr, double* cpu = nullptr)
      : walltime_(wall)
      , cputime_(cpu)
      , is_ended_(false)
      , is_naive_(true)
  {
    start_ = std::chrono::steady_clock::now();
    cpu_start_ = utils::CpuTime();
  }
  ScopedTimer(const std::string& task_name)
      : is_ended_(false)
      , is_naive_(false)
  {
    start_ = std::chrono::steady_clock::now();
    cpu_start_ = utils::CpuTime();
    GlobalTimeStats::instance().StartTaskTiming(task_name);
  }
  virtual ~ScopedTimer()
  {
    if (is_naive_) {
      auto [w, c] = GetMs();
      if (walltime_)
        *walltime_ = w;
      if (cputime_)
        *cputime_ = c;
    } else if (!is_ended_) {
      EndTiming();
    }
  }
  std::pair<double, double> GetMs()
  {
    auto cputime = (utils::CpuTime() - cpu_start_) * 1000;
    auto end = std::chrono::steady_clock::now();
    return { std::chrono::duration<double, std::milli>(end - start_).count(), cputime };
  }

  int EndTiming()
  {
    if (is_ended_) {
      std::cout << "timer has already ended" << std::endl;
      return 1;
    }
    GlobalTimeStats::instance().EndCurTaskTiming();
    is_ended_ = true;
    return 0;
  }

  private:
  double* walltime_ = nullptr;
  double* cputime_ = nullptr;
  Time start_;
  double cpu_start_ = 0;
  bool is_ended_;
  bool is_naive_;
};

#define SCOPED_TIMER(name) utils::ScopedTimer timer##__COUNTER__(name)
}