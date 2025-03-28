#ifndef __OPERATOR_H__
#define __OPERATOR_H__

#include "expr.h"

// ======================== 标量包装类 ========================
template <typename T>
class ScalarWrapper : public Expr<ScalarWrapper<T>> {
  T value_;
  typename simd<T>::type simd_value_;

  // 防止编译器过度优化
  static void force_simd_store(typename simd<T>::type& dest, typename simd<T>::type src) {
    volatile T* dummy = reinterpret_cast<volatile T*>(&dest);
    simd<T>::store(const_cast<T*>(dummy), src);
  }

 public:
  explicit ScalarWrapper(T val) : value_(val) { force_simd_store(simd_value_, simd<T>::set1(value_)); }

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

// ======================== 运算符重载 ========================
// 向量 + 向量
template <typename L, typename R>
AddExpr<L, R> operator+(const Expr<L>& lhs, const Expr<R>& rhs) {
  return AddExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const Expr<L>& lhs, T rhs) {
  return AddExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 + 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const Expr<R>& rhs) {
  return AddExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

// 向量 - 向量
template <typename L, typename R>
SubExpr<L, R> operator-(const Expr<L>& lhs, const Expr<R>& rhs) {
  return SubExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const Expr<L>& lhs, T rhs) {
  return SubExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 - 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const Expr<R>& rhs) {
  return SubExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

// 向量 * 向量
template <typename L, typename R>
MulExpr<L, R> operator*(const Expr<L>& lhs, const Expr<R>& rhs) {
  return MulExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const Expr<L>& lhs, T rhs) {
  return MulExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 * 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const Expr<R>& rhs) {
  return MulExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

// 向量 / 向量
template <typename L, typename R>
DivExpr<L, R> operator/(const Expr<L>& lhs, const Expr<R>& rhs) {
  return DivExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const Expr<L>& lhs, T rhs) {
  return DivExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 / 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const Expr<R>& rhs) {
  return DivExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

#endif  // __OPERATOR_H__