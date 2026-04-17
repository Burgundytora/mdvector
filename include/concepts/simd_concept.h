#ifndef __MDARRAY_SIMD_CONCEPT__
#define __MDARRAY_SIMD_CONCEPT__

#include "base_concept.h"

namespace md {

// ============================================================================
// SIMD 类型成员检查
// ============================================================================
template <typename S>
concept HasSimdTypes = requires {
  typename S::type;
  typename S::ref_type;
  typename S::const_type;
  typename S::const_ref_type;

  { S::alignment } -> std::convertible_to<size_t>;
  { S::pack_size } -> std::convertible_to<size_t>;
};

// ============================================================================
// SIMD 广播操作
// ============================================================================
template <typename S, typename T>
concept HasSimdBroadcast = requires(T val) {
  { S::set1(val) } -> std::same_as<typename S::type>;
};

// ============================================================================
// SIMD 读写操作
// ============================================================================
template <typename S, typename T>
concept HasSimdLoadStore = requires(const T* p, T* mp, typename S::const_ref_type v) {
  // 对齐读写
  { S::load(p) } -> std::same_as<typename S::type>;
  { S::store(mp, v) } -> std::same_as<void>;

  // 非对齐读写
  { S::loadu(p) } -> std::same_as<typename S::type>;
  { S::storeu(mp, v) } -> std::same_as<void>;
};

// ============================================================================
// SIMD 掩码读写操作
// ============================================================================
template <typename S, typename T>
concept HasSimdMaskLoadStore = requires(const T* p, T* mp, typename S::const_ref_type v, size_t rem) {
  { S::mask_load(p, rem) } -> std::same_as<typename S::type>;
  { S::mask_store(mp, rem, v) } -> std::same_as<void>;
  { S::mask_loadu(p, rem) } -> std::same_as<typename S::type>;
  { S::mask_storeu(mp, rem, v) } -> std::same_as<void>;
};

// ============================================================================
// SIMD 四则运算
// ============================================================================
template <typename S>
concept HasSimdArithmetic = requires(typename S::const_ref_type a, typename S::const_ref_type b) {
  { S::add(a, b) } -> std::same_as<typename S::type>;
  { S::sub(a, b) } -> std::same_as<typename S::type>;
  { S::mul(a, b) } -> std::same_as<typename S::type>;
  { S::div(a, b) } -> std::same_as<typename S::type>;
};

// ============================================================================
// SIMD 比较操作
// ============================================================================
template <typename S>
concept HasSimdCompare = requires(typename S::const_ref_type a, typename S::const_ref_type b) {
  { S::eq(a, b) } -> std::same_as<typename S::type>;
  { S::ne(a, b) } -> std::same_as<typename S::type>;
  { S::lt(a, b) } -> std::same_as<typename S::type>;
  { S::le(a, b) } -> std::same_as<typename S::type>;
  { S::gt(a, b) } -> std::same_as<typename S::type>;
  { S::ge(a, b) } -> std::same_as<typename S::type>;
};

// ============================================================================
// SIMD 位运算（整数专用）
// ============================================================================
template <typename S>
concept HasSimdBitwise = requires(typename S::const_ref_type a, typename S::const_ref_type b) {
  { S::bit_and(a, b) } -> std::same_as<typename S::type>;
  { S::bit_or(a, b) } -> std::same_as<typename S::type>;
  { S::bit_xor(a, b) } -> std::same_as<typename S::type>;
  { S::bit_not(a) } -> std::same_as<typename S::type>;
};

// ============================================================================
// SIMD 数学函数
// ============================================================================
template <typename S>
concept HasSimdMath = requires(typename S::const_ref_type v) {
  { S::abs(v) } -> std::same_as<typename S::type>;
  { S::sqrt(v) } -> std::same_as<typename S::type>;
  { S::max(v, v) } -> std::same_as<typename S::type>;
  { S::min(v, v) } -> std::same_as<typename S::type>;
};

// ============================================================================
// SIMD FMA 操作
// ============================================================================
template <typename S>
concept HasSimdFma =
    requires(typename S::const_ref_type a, typename S::const_ref_type b, typename S::const_ref_type c) {
      { S::fma(a, b, c) } -> std::same_as<typename S::type>;  // a * b + c
      { S::fms(a, b, c) } -> std::same_as<typename S::type>;  // a * b - c
    };

}  // namespace md

#endif  // __MDARRAY_SIMD_CONCEPT__