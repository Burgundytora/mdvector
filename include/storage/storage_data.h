// core/storage_data.h
#ifndef __MDVECTOR_STORAGE_DATA__
#define __MDVECTOR_STORAGE_DATA__

#include "../common/detail.h"
#include "../simd/simd.h"
#include "../simd/allocator.h"
#include "storage_concept.h"

#include <array>
#include <vector>

namespace md {

// ============================================================================
// 1. 栈存储（固定尺寸）- 纯数据
// ============================================================================
template <typename T, size_t... Lengths>
class array_storage {
 public:
  using value_type = T;
  using ownership = owns_data_tag;

 private:
  static constexpr size_t raw_size_ = (Lengths * ... * 1);
  static constexpr size_t aligned_size_ =
      (raw_size_ % simd<T>::pack_size == 0) ? raw_size_ : ((raw_size_ / simd<T>::pack_size) + 1) * simd<T>::pack_size;
  alignas(simd<T>::alignment) std::array<T, aligned_size_> data_;

 public:
  array_storage() = default;

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  static constexpr size_t size() noexcept { return raw_size_; }
  static constexpr size_t capacity() noexcept { return aligned_size_; }
  size_t used_size() const noexcept { return aligned_size_; }
  size_t static_size() const noexcept { return aligned_size_; }
};

// ============================================================================
// 2. 堆存储（动态尺寸）- 纯数据
// ============================================================================
template <typename T>
class vector_storage {
 public:
  using value_type = T;
  using ownership = owns_data_tag;

 private:
  std::vector<T, auto_allocator<T>> data_;
  size_t size_ = 0;

 public:
  vector_storage() = default;
  explicit vector_storage(size_t n) : data_(get_aligned_size<T>(n)), size_(n) {}

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

// ============================================================================
// 3. Span 存储（外部指针，连续）- 纯数据
// ============================================================================
template <typename T>
class span_storage {
 public:
  using value_type = T;
  using ownership = view_data_tag;

 private:
  T* data_ = nullptr;
  size_t size_ = 0;

 public:
  span_storage() = default;
  span_storage(T* ptr, size_t n) : data_(ptr), size_(n) {}

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return get_aligned_size<T>(size_); }
  size_t used_size() const noexcept { return capacity(); }
};

// ============================================================================
// 4. View 存储（外部指针，跨步）- 纯数据
// ============================================================================
template <typename T>
class view_storage {
 public:
  using value_type = T;
  using ownership = view_data_tag;

 private:
  T* data_ = nullptr;
  size_t size_ = 0;

 public:
  view_storage() = default;

  void init(T* ptr, size_t n) {
    data_ = ptr;
    size_ = n;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return get_aligned_size<T>(size_); }
  size_t used_size() const noexcept { return capacity(); }
};

namespace detail {

// 编译期验证容器特性
template <typename T>
struct storage_checks {
  static_assert(BasicStorage<array_storage<T, 3, 3>>);
  static_assert(BasicStorage<vector_storage<T>>);
  static_assert(BasicStorage<span_storage<T>>);
  static_assert(BasicStorage<view_storage<T>>);

  static_assert(OwningStorage<array_storage<T, 3, 3>>);
  static_assert(OwningStorage<vector_storage<T>>);
  static_assert(!OwningStorage<span_storage<T>>);
  static_assert(!OwningStorage<view_storage<T>>);

  static_assert(!ViewStorage<array_storage<T, 3, 3>>);
  static_assert(!ViewStorage<vector_storage<T>>);
  static_assert(ViewStorage<span_storage<T>>);
  static_assert(ViewStorage<view_storage<T>>);
};

// 验证数值类型
template struct storage_checks<float>;
template struct storage_checks<double>;
template struct storage_checks<int>;

}  // namespace detail

}  // namespace md

#endif