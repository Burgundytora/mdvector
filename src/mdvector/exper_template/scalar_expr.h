#ifndef __MDVECTOR_SCALAR_EXPR_H__
#define __MDVECTOR_SCALAR_EXPR_H__

#include "base_expr.h"

#if defined(__GNUC__) || defined(__clang__)
#include <cstring>
#endif

// ======================== 标量包装类 ========================
template <typename T>
class ScalarWrapper : public Expr<ScalarWrapper<T>> {
  T value_;
  typename simd<T>::type simd_value_;

  // 防止编译器过度优化
  static void force_simd_store(typename simd<T>::type& dest, typename simd<T>::type src) {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang：内存屏障+强制内存写入
    asm volatile("" ::: "memory");
    alignas(simd<T>::alignment) T buffer[simd<T>::pack_size];
    simd<T>::store(buffer, src);
    std::memcpy(&dest, buffer, sizeof(dest));
#else
    // MSVC：volatile写入
    volatile char* dummy = reinterpret_cast<volatile char*>(&dest);
    simd<T>::store(reinterpret_cast<T*>(const_cast<char*>(dummy)), src);
#endif
  }

 public:
#if defined(__clang__)
  __attribute__((noinline, used))
#endif
  explicit ScalarWrapper(T val)
      : value_(val) {
    // 确保value_已初始化
    std::atomic_signal_fence(std::memory_order_seq_cst);

    // 分步操作防止优化
    auto tmp = simd<T>::set1(value_);
    force_simd_store(simd_value_, tmp);

    // 最终验证（调试用）
#ifndef NDEBUG
    T verify[simd<T>::pack_size];
    simd<T>::store(verify, simd_value_);
    assert(verify[0] == value_ && "SIMD init failed!");
#endif
  }

  // 允许拷贝
  ScalarWrapper(const ScalarWrapper&) = default;

  template <typename U>
  typename simd<U>::type eval_simd(size_t) const {
    return simd_value_;
  }

  template <typename U>
  typename simd<U>::type eval_simd_mask(size_t) const {
    return simd_value_;
  }

  size_t size() const { return 1; }

  std::array<size_t, 1> shape() const { return std::array<size_t, 1>{1}; }
};

#endif  // __MDVECTOR_SCALAR_EXPR_H__