#pragma once

#include <limits>
#include <memory>

#include "simd_base.h"

namespace md {
template <typename T>
class simd_allocator {
 public:
  using value_type = T;

  simd_allocator() noexcept = default;

  // 添加复制构造函数
  simd_allocator(const simd_allocator&) noexcept = default;

  // 添加移动构造函数
  simd_allocator(simd_allocator&&) noexcept = default;

  // 添加模板复制构造函数（允许从其他类型的 simd_allocator 转换）
  template <typename U>
  simd_allocator(const simd_allocator<U>&) noexcept {}

  // 赋值运算符
  simd_allocator& operator=(const simd_allocator&) noexcept = default;

  // 比较运算符（分配器应该总是相等的）
  template <typename U>
  bool operator==(const simd_allocator<U>&) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const simd_allocator<U>&) const noexcept {
    return false;
  }

  static constexpr size_t alignment_for() {
    if constexpr (std::is_arithmetic_v<T>) {
      return simd<T>::alignment;
    } else {
      return alignof(T);
    }
  }

  T* allocate(size_t n) {
    if (n > max_size()) {
      throw std::bad_alloc();
    }
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(n * sizeof(T), alignment_for());
#else
    // aligned_alloc(alignment_for(), n * sizeof(T));

    // macos下奇数个会bad_alloc
    // 在 macOS 和其他 Unix 系统上，使用 posix_memalign 替代 aligned_alloc
    // posix_memalign 没有大小必须是 alignment 整数倍的限制
    if (posix_memalign(&ptr, alignment_for(), n * sizeof(T)) != 0) {
      ptr = nullptr;
    }
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t _ = 0) noexcept {
    if (p) {
#ifdef _WIN32
      _aligned_free(p);
#else
      free(p);
#endif
    }
  }

  size_t max_size() const noexcept { return std::numeric_limits<size_t>::max() / sizeof(T); }
};

template <typename T>
using auto_allocator = std::conditional_t<std::is_floating_point_v<T>, simd_allocator<T>, std::allocator<T>>;

}  // namespace md
