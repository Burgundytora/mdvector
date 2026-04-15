#ifndef __MDVECTOR_STACK_STORAGE__
#define __MDVECTOR_STACK_STORAGE__

#include "../concepts/storage_concept.h"
#include "../simd/simd.h"

#include <array>

namespace md {

// ============================================================================
// 栈存储（固定尺寸）- 纯数据
// ============================================================================
template <typename T, size_t... Lengths>
class stack_storage {
 public:
  using value_type = T;
  using ownership = owns_data_tag;

 private:
  static constexpr size_t raw_size_ = (Lengths * ... * 1);
  static constexpr size_t aligned_size_ =
      (raw_size_ % simd<T>::pack_size == 0) ? raw_size_ : ((raw_size_ / simd<T>::pack_size) + 1) * simd<T>::pack_size;
  alignas(simd<T>::alignment) std::array<T, aligned_size_> data_;

 public:
  stack_storage() = default;

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  static constexpr size_t size() noexcept { return raw_size_; }
  static constexpr size_t capacity() noexcept { return aligned_size_; }
  size_t used_size() const noexcept { return aligned_size_; }
  size_t static_size() const noexcept { return aligned_size_; }
};

}  // namespace md

#endif  // __MDVECTOR_STACK_STORAGE__