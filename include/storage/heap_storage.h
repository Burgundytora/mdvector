#ifndef __MDVECTOR_HEAP_STORAGE__
#define __MDVECTOR_HEAP_STORAGE__

#include "../concepts/storage_concept.h"
#include "../simd/allocator.h"

#include <vector>

namespace md {

// ============================================================================
// 堆存储（动态尺寸）- 纯数据
// ============================================================================
template <typename T, bool AligedSize = true>
class heap_storage {
 public:
  using value_type = T;
  using ownership = owns_data_tag;

 private:
  std::vector<T, auto_allocator<T>> data_;
  size_t size_ = 0;

 public:
  heap_storage() = default;
  explicit heap_storage(size_t n) : data_(get_aligned_size<T>(n)), size_(n) {}

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return data_.size(); }
  size_t used_size() const noexcept { return data_.size(); }

  void resize(size_t n) {
    size_ = n;
    data_.resize(get_aligned_size<T>(n));
  }
};

}  // namespace md

#endif  // __MDVECTOR_HEAP_STORAGE__