#ifndef __MDVECTOR_ALLOCATOR_H__
#define __MDVECTOR_ALLOCATOR_H__

#include <memory>

#include "../simd/simd.h"

template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;

  // 必须的别名
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using is_always_equal = std::true_type;

  // 允许分配器类型转换的构造函数
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U>;
  };

  // 默认构造函数
  AlignedAllocator() noexcept = default;

  // 拷贝构造函数
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  static constexpr size_t alignment_for() {
    // 对数值类型使用SIMD对齐，其他类型使用默认对齐
    if constexpr (std::is_arithmetic_v<T>) {
      return simd<T>::alignment;
    } else {
      return alignof(T);  // 使用类型的自然对齐
    }
  }

  T* allocate(size_t n) {
    if (n > max_size()) {
      throw std::bad_alloc();
    }
    void* ptr =
#ifdef _WIN32
        _aligned_malloc(n * sizeof(T), alignment_for());
#else
        aligned_alloc(alignment_for(), n * sizeof(T));
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  // 释放函数
  void deallocate(T* p, size_t _ = 0) noexcept {
    if (p) {
#ifdef _WIN32
      _aligned_free(p);
#else
      free(p);
#endif
    }
  }

  // 最大可分配大小
  size_t max_size() const noexcept { return std::numeric_limits<size_t>::max() / sizeof(T); }

  // 构造对象
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  // 销毁对象
  template <typename U>
  void destroy(U* p) {
    p->~U();
  }

  // 比较操作符
  template <typename U>
  bool operator==(const AlignedAllocator<U>&) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocator<U>&) const noexcept {
    return false;
  }
};

#endif  // __MDVECTOR_ALLOCATOR_H__
