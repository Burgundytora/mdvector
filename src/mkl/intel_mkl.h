#ifndef HEADER_INTEL_MKL_H_
#define HEADER_INTEL_MKL_H_

#include "mkl.h"

// amd平台性能不如avx与for循环
void set_mkl_avx2_sequential_mode() {
  // 添加指令集和线程层双重设置
  MKL_Set_Threading_Layer(MKL_THREADING_SEQUENTIAL);
  MKL_Enable_Instructions(MKL_ENABLE_AVX2);
}

void set_mkl_avx512_sequential_mode() {
  // 添加指令集和线程层双重设置
  MKL_Set_Threading_Layer(MKL_THREADING_SEQUENTIAL);
  MKL_Enable_Instructions(MKL_ENABLE_AVX512);
}

template <typename T>
class MklAlignedAllocator {
 public:
  using value_type = T;
  T* allocate(size_t n) {
    // size_t aligned_size = ((n + 8 - 1) / 8) * 8;
    // aligned_size = std::max(aligned_size, size_t{8});
    size_t byte_size = n * sizeof(T);
    size_t aligned_size = (byte_size + 63) / 64 * 64;  // 确保是64的倍数??
    // void* ptr = mkl_malloc(aligned_size * sizeof(T), 64);
    void* ptr = mkl_malloc(aligned_size, 64);
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p) noexcept {
    if (p) {
      mkl_free(p);
    }
  }
};

template <class T>
void mkl_add(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    vsAdd(static_cast<int>(n), a, b, c);
  } else {
    vdAdd(static_cast<int>(n), a, b, c);
  }
}

template <class T>
void mkl_sub(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    vsSub(static_cast<int>(n), a, b, c);
  } else {
    vdSub(static_cast<int>(n), a, b, c);
  }
}

template <class T>
void mkl_mul(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    vsMul(static_cast<int>(n), a, b, c);
  } else {
    vdMul(static_cast<int>(n), a, b, c);
  }
}

template <class T>
void mkl_div(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    vsDiv(static_cast<int>(n), a, b, c);
  } else {
    vdDiv(static_cast<int>(n), a, b, c);
  }
}

#endif  // HEADER_INTEL_MKL_H_