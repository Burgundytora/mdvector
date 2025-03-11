#ifndef HEADER_EXPR_H_
#define HEADER_EXPR_H_

// 表达式模板
template <typename Derived>
class Expr {
 public:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  // 关键方法：直接操作目标内存
  template <typename T>
  void eval_to(T* __restrict dest) const {
    derived().eval_to_impl(dest);
  }
};

#endif  // HEADER_EXPR_H_
