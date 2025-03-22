#ifndef __MDVECTOR_BASEEXPR_H__
#define __MDVECTOR_BASEEXPR_H__

#include "mask.h"
#include "simd_config.h"

// ======================== 表达式模板基类 ========================
template <typename Derived>
class Expr {
 public:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  size_t size() const { return derived().size(); }

  auto eval_simd(size_t i) const { return static_cast<const Derived&>(*this).eval_simd(i); }

  template <typename Dest>
  void eval_to(Dest* dest) const {
    const size_t n = size();
    constexpr size_t pack_size = SimdConfig<Dest>::pack_size;
    size_t i = 0;

    for (; i <= n - pack_size; i += pack_size) {
      // 虚函数开销在这？？
      auto simd_val = derived().template eval_simd<Dest>(i);
      // __m256d simd_val = _mm256_load_pd(ttt);
      if constexpr (std::is_same_v<std::remove_const_t<Dest>, float>) {
        _mm256_store_ps(dest + i, simd_val);
      } else {
        auto simd_val1 = simd_val;
        _mm256_store_pd(dest + i, simd_val);
      }
    }

    // 使用预生成掩码处理尾部元素
    if (i < n) {
      const size_t remaining = n - i;
      auto simd_val = derived().template eval_simd_mask<std::remove_const_t<Dest>>(i);

      if constexpr (std::is_same_v<std::remove_const_t<Dest>, float>) {
        // float类型使用mask_table_8
        const __m256i mask = mask_table_8[remaining];
        _mm256_maskstore_ps(dest + i, mask, simd_val);
      } else {
        // double类型使用mask_table_4
        const __m256i mask = mask_table_4[remaining];
        _mm256_maskstore_pd(dest + i, mask, simd_val);
      }
    }
  }
};

// ======================== 表达式类 ========================
template <typename L, typename R>
class AddExpr : public Expr<AddExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  AddExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_add_ps(l, r);
    } else {
      return _mm256_add_pd(l, r);
    }
  }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_add_ps(l, r);
    } else {
      return _mm256_add_pd(l, r);
    }
  }
};

template <typename L, typename R>
class SubExpr : public Expr<SubExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  SubExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_sub_ps(l, r);
    } else {
      return _mm256_sub_pd(l, r);
    }
  }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_sub_ps(l, r);
    } else {
      return _mm256_sub_pd(l, r);
    }
  }
};

template <typename L, typename R>
class MulExpr : public Expr<MulExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  MulExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_mul_ps(l, r);
    } else {
      return _mm256_mul_pd(l, r);
    }
  }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_mul_ps(l, r);
    } else {
      return _mm256_mul_pd(l, r);
    }
  }
};

template <typename L, typename R>
class DivExpr : public Expr<DivExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  DivExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_div_ps(l, r);
    } else {
      return _mm256_div_pd(l, r);
    }
  }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_maskdiv_ps(l, mask_table_8[l.size() - i], r);
    } else {
      return _mm256_div_pd(l, r);
    }
  }
};

#endif  // __MDVECTOR_BASEEXPR_H__
