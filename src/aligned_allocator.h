#ifndef __ALIGNED_ALLOCATOR_H__
#define __ALIGNED_ALLOCATOR_H__

#include <cstdlib>
#include <memory>

// 内存对齐 默认AVX2 32bit
// TODO: 根据不同指令集 条件宏设置不同对齐大小
constexpr size_t alignment_ = 32;

// 默认AVX2 32对齐
template <typename T, size_t Alignment = alignment_>
struct AlignedAllocator {
  using value_type = T;

  using pointer = T*;
  using const_pointer = const T*;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  // 对齐分配函数
  pointer allocate(size_type n) {
    if (n > std::numeric_limits<size_type>::max() / sizeof(T)) throw std::bad_alloc();

    size_t size = n * sizeof(T);
    void* ptr;

// 平台兼容的分配方式
#ifdef _WIN32
    ptr = _aligned_malloc(size, Alignment);
#else
    if (posix_memalign(&ptr, Alignment, size) != 0) ptr = nullptr;
#endif

    if (!ptr) throw std::bad_alloc();
    return static_cast<pointer>(ptr);
  }

  // 对齐释放函数
  void deallocate(pointer p, size_type) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
  }

  // 备用
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
  bool operator==(const AlignedAllocator&) const { return true; }
  bool operator!=(const AlignedAllocator&) const { return false; }
};

#endif  // __ALIGNED_ALLOCATOR_H__