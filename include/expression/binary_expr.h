#ifndef __MDARRAY_BINARY_EXPR__
#define __MDARRAY_BINARY_EXPR__

#include "extract_layout.h"
#include "../simd/simd_op_binary.h"

namespace md {

template <typename T, typename L, typename R, typename Cal>
class binary_expr : public base_expr<binary_expr<T, L, R, Cal>, T> {
 public:
  using value_type = T;
  static constexpr size_t rank_ = derived_rank<L, R>();
  using layout_type = layout_type_t<L, R>;

 private:
  AutoType<L> lhs;
  AutoType<R> rhs;

 public:
  binary_expr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t used_size() const {
    if constexpr (std::is_arithmetic_v<R>) {
      return lhs.used_size();
    } else {
      return rhs.used_size();
    }
  }

  auto extents() const {
    if constexpr (std::is_arithmetic_v<R>) {
      return lhs.extents();
    } else {
      return rhs.extents();
    }
  }

  template <typename U>
  typename simd<U>::type load_simd(size_t i) const {
    auto l = lhs.template load_simd<U>(i);
    auto r = rhs.template load_simd<U>(i);
    return simd_op_binary<U, Cal>(l, r);
  }

  template <typename U>
  typename simd<U>::type load_simd_mask(size_t i) const {
    auto l = lhs.template load_simd_mask<U>(i);
    auto r = rhs.template load_simd_mask<U>(i);
    return simd_op_binary<U, Cal>(l, r);
  }

  // 取负
  auto operator-() const noexcept
    requires Numeric<T>
  {
    return *this * static_cast<T>(-1);
  }

  // 取正
  auto operator+() const noexcept
    requires Numeric<T>
  {
    return *this;
  }

  // // 打印 注释掉防止循环依赖
  // void print() {
  //   md::vector<T, rank_, layout_type> result = *this;
  //   result.print();
  // }
};

}  // namespace md

#endif  // __MDARRAY_BINARY_EXPR__