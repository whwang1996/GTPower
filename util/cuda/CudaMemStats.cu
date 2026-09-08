#include "CudaMemStats.hh"

#include "Log.hh"

namespace utils::cuda {

void
CudaMemStats::reset()
{
  for (size_t& bytes : category_bytes_) {
    bytes = 0;
  }
}

size_t
CudaMemStats::total() const
{
  size_t bytes = 0;
  for (size_t category_bytes : category_bytes_) {
    bytes += category_bytes;
  }
  return bytes;
}

size_t&
CudaMemStats::field(CategoryId category)
{
  if (category >= category_bytes_.size()) {
    category_bytes_.resize(category + 1, 0);
  }
  return category_bytes_[category];
}

double
CudaMemStats::bytesToGB(size_t bytes)
{
  return bytes / (1024.0 * 1024.0 * 1024.0);
}

const char*
CudaMemStats::categoryName(CategoryId category)
{
  static constexpr const char* names[] = {
    "events_device",
    "gates_device",
    "gate_aux_device",
    "delay_device",
    "power_result_device",
    "schedule_device",
    "tile_result_device",
    "LUT managed",
    "LUT table_device",
    "fallback_lut_index_device",
    "BSIM LUT index_device",
    "BSIM leakage index_device"
  };

  return category < CudaMemStats::category_count ? names[category] : "category";
}

void
CudaMemStats::set(Category category, size_t bytes)
{
  field(static_cast<CategoryId>(category)) = bytes;
}

void
CudaMemStats::add(Category category, size_t bytes)
{
  field(static_cast<CategoryId>(category)) += bytes;
}

void
CudaMemStats::log(const char* title, const char* phase) const
{
  LOG_BEGIN(INFO, title);
  LOG_INFO << "phase: " << phase;
  for (size_t category = 0; category < category_bytes_.size(); ++category) {
    LOG_INFO << categoryName(category) << ": " << bytesToGB(category_bytes_[category]) << " GB";
  }
  LOG_INFO << "total tracked CUDA device memory: " << bytesToGB(total()) << " GB";
  LOG_END(INFO, title);
}

}  // namespace utils::cuda
