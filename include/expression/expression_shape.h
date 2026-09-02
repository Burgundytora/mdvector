#pragma once

#include <stdexcept>
#include <type_traits>

namespace md::detail {

template <typename L, typename R>
inline void debug_check_same_shape(const L& lhs, const R& rhs) {
#ifndef NDEBUG
  if constexpr (L::rank_ != R::rank_) {
    throw std::invalid_argument("expression rank mismatch");
  } else if (lhs.extents() != rhs.extents()) {
    throw std::invalid_argument("expression shape mismatch");
  }
#else
  (void)lhs;
  (void)rhs;
#endif
}

template <typename Expr, typename Dest>
inline void debug_check_destination_shape(const Expr& expr, const Dest& dest) {
#ifndef NDEBUG
  if constexpr (Expr::rank_ != Dest::rank_) {
    throw std::invalid_argument("expression destination rank mismatch");
  } else if (expr.extents() != dest.extents()) {
    throw std::invalid_argument("expression destination shape mismatch");
  }
#else
  (void)expr;
  (void)dest;
#endif
}

}  // namespace md::detail
