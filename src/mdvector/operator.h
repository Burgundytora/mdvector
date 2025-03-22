#include "expr.h"

// ======================== 运算符重载 ========================
template <typename L, typename R>
AddExpr<L, R> operator+(const Expr<L>& lhs, const Expr<R>& rhs) {
  return AddExpr<L, R>(lhs.derived(), rhs.derived());
}

template <typename L, typename R>
SubExpr<L, R> operator-(const Expr<L>& lhs, const Expr<R>& rhs) {
  return SubExpr<L, R>(lhs.derived(), rhs.derived());
}

template <typename L, typename R>
MulExpr<L, R> operator*(const Expr<L>& lhs, const Expr<R>& rhs) {
  return MulExpr<L, R>(lhs.derived(), rhs.derived());
}

template <typename L, typename R>
DivExpr<L, R> operator/(const Expr<L>& lhs, const Expr<R>& rhs) {
  return DivExpr<L, R>(lhs.derived(), rhs.derived());
}
