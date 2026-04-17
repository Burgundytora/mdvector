#ifndef __MDARRAY_VIEW_STORAGE__
#define __MDARRAY_VIEW_STORAGE__

#include "../concepts/storage_concept.h"

#include "../simd/simd.h"

namespace md {

// ============================================================================
// Span连续/View跨步 存储（外部指针）- 纯数据
// ============================================================================
template <typename T, bool AligedSize = true>
class view_storage {
 public:
  using value_type = T;
  using ownership = view_data_tag;

 private:
  T* data_ = nullptr;
  size_t raw_size_ = 0;
  size_t align_size_ = 0;
  size_t remaining_size_ = 0;

 public:
  view_storage() = default;
  view_storage(T* ptr, size_t n)
      : data_(ptr),
        raw_size_(n),
        align_size_(get_aligned_size<T>(raw_size_)),
        remaining_size_(raw_size_ > simd<T>::pack_size ? align_size_ - raw_size_ : raw_size_) {}

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  size_t size() const noexcept { return raw_size_; }
  size_t capacity() const noexcept { return align_size_; }
  size_t used_size() const noexcept { return align_size_; }
  size_t remaining_size() const noexcept { return remaining_size_; }
};

}  // namespace md

#endif  // __MDARRAY_VIEW_STORAGE__