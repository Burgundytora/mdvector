// core/storage.h
#ifndef __MDVECTOR_CORE_STORAGE__
#define __MDVECTOR_CORE_STORAGE__

#include "../common/detail.h"
#include "../simd/allocator.h"
#include <array>
#include <vector>
#include <random>
#include <algorithm>

namespace md {

// ============================================================================
// 存储策略标签（带维度信息）
// ============================================================================
struct vector_storage {};          // vector - 堆上动态尺寸，连续
struct span_storage {};            // span - 外部指针，连续内存视图
struct view_storage {};            // view - 外部指针，跨步视图
struct fixed_capacity_storage {};  // inplace_vector - 固定容量

// array 存储策略（带编译期维度）
template <size_t... Lengths>
struct array_storage {};

// ============================================================================
// 主模板
// ============================================================================
template <typename Derived, typename T, typename StoragePolicy>
class storage_impl;

// ============================================================================
// 特化1：array_storage（固定尺寸）
// ============================================================================
template <typename Derived, typename T, size_t... Lengths>
class storage_impl<Derived, T, array_storage<Lengths...>> {
 protected:
  static constexpr size_t raw_size_ = (Lengths * ... * 1);
  static constexpr size_t aligned_size_ =
      (raw_size_ % simd<T>::pack_size == 0) ? raw_size_ : ((raw_size_ / simd<T>::pack_size) + 1) * simd<T>::pack_size;

  alignas(simd<T>::alignment) std::array<T, aligned_size_> data_;

 public:
  storage_impl() = default;

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  size_t capacity() const noexcept { return aligned_size_; }
  size_t size() const noexcept { return raw_size_; }
  size_t used_size() const noexcept { return aligned_size_; }
  static constexpr size_t static_size() noexcept { return raw_size_; }

  void resize(size_t) {}

  void fill(T val) { std::fill_n(data_.data(), raw_size_, val); }
  void set_zeros()
    requires Numeric<T>
  {
    fill(static_cast<T>(0));
  }
  void set_ones()
    requires Numeric<T>
  {
    fill(static_cast<T>(1));
  }

  void set_arange(T start = 0, T step = 1)
    requires Numeric<T>
  {
    T current = start;
    for (size_t i = 0; i < raw_size_; ++i) {
      data_[i] = current;
      current += step;
    }
  }

  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Numeric<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < raw_size_; ++i) data_[i] = dis(gen);
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < raw_size_; ++i) data_[i] = dis(gen);
    }
  }

  void set_random_normal(T mean = 0, T stddev = 1)
    requires Numeric<T> && std::is_floating_point_v<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (size_t i = 0; i < raw_size_; ++i) data_[i] = dis(gen);
  }

  T* begin() noexcept { return data_.data(); }
  T* end() noexcept { return data_.data() + raw_size_; }
  const T* begin() const noexcept { return data_.data(); }
  const T* end() const noexcept { return data_.data() + raw_size_; }
};

// ============================================================================
// 特化2：vector_storage（动态尺寸）
// ============================================================================
template <typename Derived, typename T>
class storage_impl<Derived, T, vector_storage> {
 protected:
  std::vector<T, auto_allocator<T>> data_;
  size_t size_ = 0;

 public:
  storage_impl() = default;
  explicit storage_impl(size_t n) : data_(get_aligned_size<T>(n)), size_(n) {}

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  size_t capacity() const noexcept { return data_.size(); }
  size_t size() const noexcept { return size_; }
  size_t used_size() const noexcept { return data_.size(); }

  void resize(size_t n) {
    size_ = n;
    data_.resize(get_aligned_size<T>(n));
  }

  void fill(T val) { std::fill_n(data_.data(), size_, val); }
  void set_zeros()
    requires Numeric<T>
  {
    fill(static_cast<T>(0));
  }
  void set_ones()
    requires Numeric<T>
  {
    fill(static_cast<T>(1));
  }

  void set_arange(T start = 0, T step = 1)
    requires Numeric<T>
  {
    T current = start;
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = current;
      current += step;
    }
  }

  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Numeric<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) data_[i] = dis(gen);
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) data_[i] = dis(gen);
    }
  }

  void set_random_normal(T mean = 0, T stddev = 1)
    requires Numeric<T> && std::is_floating_point_v<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (size_t i = 0; i < size_; ++i) data_[i] = dis(gen);
  }

  T* begin() noexcept { return data_.data(); }
  T* end() noexcept { return data_.data() + size_; }
  const T* begin() const noexcept { return data_.data(); }
  const T* end() const noexcept { return data_.data() + size_; }
};

// ============================================================================
// 特化3：span_storage
// ============================================================================
template <typename Derived, typename T>
class storage_impl<Derived, T, span_storage> {
 protected:
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  size_t remaining_size_ = 0;

 public:
  storage_impl() = default;
  storage_impl(T* ptr, size_t n)
      : data_(ptr),
        size_(n),
        capacity_(get_aligned_size<T>(n)),
        remaining_size_(n > simd<T>::pack_size ? capacity_ - n : n) {}

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  size_t capacity() const noexcept { return capacity_; }
  size_t size() const noexcept { return size_; }
  size_t used_size() const noexcept { return capacity_; }
  size_t remaining_size() const noexcept { return remaining_size_; }

  void resize(size_t) {}

  void fill(T val) { std::fill_n(data_, size_, val); }
  void set_zeros()
    requires Numeric<T>
  {
    fill(static_cast<T>(0));
  }
  void set_ones()
    requires Numeric<T>
  {
    fill(static_cast<T>(1));
  }

  void set_arange(T start = 0, T step = 1)
    requires Numeric<T>
  {
    T current = start;
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = current;
      current += step;
    }
  }

  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Numeric<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) data_[i] = dis(gen);
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) data_[i] = dis(gen);
    }
  }

  void set_random_normal(T mean = 0, T stddev = 1)
    requires Numeric<T> && std::is_floating_point_v<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (size_t i = 0; i < size_; ++i) data_[i] = dis(gen);
  }

  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }
};

// ============================================================================
// 特化4：view_storage
// ============================================================================
template <typename Derived, typename T>
class storage_impl<Derived, T, view_storage> {
 protected:
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t align_size_ = 0;
  size_t remaining_size_ = 0;

 public:
  storage_impl() = default;

  void init_storage(T* ptr, size_t n) {
    data_ = ptr;
    size_ = n;
    align_size_ = get_aligned_size<T>(n);
    remaining_size_ = n > simd<T>::pack_size ? align_size_ - n : n;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  size_t capacity() const noexcept { return align_size_; }
  size_t size() const noexcept { return size_; }
  size_t used_size() const noexcept { return align_size_; }
  size_t remaining_size() const noexcept { return remaining_size_; }

  void resize(size_t) {}

  void fill(T val) {
    auto& derived = static_cast<Derived&>(*this);
    std::fill(derived.begin(), derived.end(), val);
  }

  void set_zeros()
    requires Numeric<T>
  {
    fill(static_cast<T>(0));
  }
  void set_ones()
    requires Numeric<T>
  {
    fill(static_cast<T>(1));
  }

  void set_arange(T start = 0, T step = 1)
    requires Numeric<T>
  {
    auto& derived = static_cast<Derived&>(*this);
    T current = start;
    for (auto& val : derived) {
      val = current;
      current += step;
    }
  }

  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Numeric<T>
  {
    auto& derived = static_cast<Derived&>(*this);
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (auto& val : derived) val = dis(gen);
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      for (auto& val : derived) val = dis(gen);
    }
  }

  void set_random_normal(T mean = 0, T stddev = 1)
    requires Numeric<T> && std::is_floating_point_v<T>
  {
    auto& derived = static_cast<Derived&>(*this);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (auto& val : derived) val = dis(gen);
  }
};

}  // namespace md

#endif