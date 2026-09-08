#pragma once

#include <cstddef>
#include <vector>

#include "Singleton.hh"

namespace utils::cuda {

class CudaMemStats : public utils::Singleton<CudaMemStats> {
public:
  using CategoryId = size_t;
  enum class Category : CategoryId {
    events,
    gate,
    gate_aux,
    delay,
    power_result,
    schedule,
    tile_result,
    lut_managed,
    lut_table,
    fallback_lut_index,
    bsim_lut_index,
    bsim_leakage_index,
    count
  };

  static constexpr CategoryId category_count = static_cast<CategoryId>(Category::count);

  CudaMemStats(token) : category_bytes_(category_count, 0) {}

  void reset();
  size_t total() const;
  void set(Category category, size_t bytes);
  void add(Category category, size_t bytes);

  void log(const char* title, const char* phase) const;

private:
  size_t& field(CategoryId category);
  static double bytesToGB(size_t bytes);
  static const char* categoryName(CategoryId category);

  std::vector<size_t> category_bytes_;
};

}  // namespace utils::cuda

#define CUDA_MEM_STATS utils::cuda::CudaMemStats::instance()
