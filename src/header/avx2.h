﻿#ifndef HEADER_AVX2_H_
#define HEADER_AVX2_H_

#include <immintrin.h>

#include <array>
#include <mdspan>
#include <vector>


// ========================================================
// SIMD内存对齐配置模板
template <typename T>
struct SimdConfig;
template <>
struct SimdConfig<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using simd_type = __m256;
};
template <>
struct SimdConfig<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using simd_type = __m256d;
};
// ========================================================

// ========================================================
// 尾部元素使用AVX2掩码
// 预生成所有可能的掩码（针对 pack_size=4 double）
alignas(32) static const __m256i mask_table_4[4] = {
    _mm256_set_epi64x(0, 0, 0, 0),    // 0元素
    _mm256_set_epi64x(0, 0, 0, -1),   // 1元素
    _mm256_set_epi64x(0, 0, -1, -1),  // 2元素
    _mm256_set_epi64x(0, -1, -1, -1)  // 3元素
};
// 预生成所有可能的掩码（针对 pack_size=8 float）
alignas(32) static const __m256i mask_table_8[8] = {
    _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),         // 0元素
    _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),        // 1元素
    _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),       // 2元素
    _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),     // 3元素
    _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),    // 4元素
    _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),   // 5元素
    _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),   // 6元素
    _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),  // 7元素
};

// ========================================================
// 双精度除法
// 近似倒数（精度：~单精度）
// _forceinline __m256d _mm256_rcp_pd_approx(__m256d v) {
//   __m128 v_low = _mm256_cvtpd_ps(v);
//   __m128 recip_ps = _mm_rcp_ps(v_low);
//   return _mm256_cvtps_pd(recip_ps);
// }

// // 高精度倒数（通过牛顿迭代）
// _forceinline __m256d _mm256_rcp_pd(__m256d v, int iterations = 2) {
//   __m256d recip = _mm256_rcp_pd_approx(v);
//   for (int i = 0; i < iterations; ++i) {
//     recip = _mm256_mul_pd(recip, _mm256_sub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(v, recip)));
//   }
//   return recip;
// }
// ========================================================

// ========================================================
// 基础运算赋值 c = a ? b
// c = a + b
template <class T>
void avx2_add(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  if constexpr (std::is_same_v<T, float>) {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      _mm256_store_ps(c + i, _mm256_add_ps(va, vb));
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_8[remaining];
      __m256 va = _mm256_maskload_ps(a + i, mask);
      __m256 vb = _mm256_maskload_ps(b + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_add_ps(va, vb));
    }
  } else {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      _mm256_store_pd(c + i, _mm256_add_pd(va, vb));
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_4[remaining];
      __m256d va = _mm256_maskload_pd(a + i, mask);
      __m256d vb = _mm256_maskload_pd(b + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_add_pd(va, vb));
    }
  }
}

// c = a - b
template <class T>
void avx2_sub(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  if constexpr (std::is_same_v<T, float>) {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_sub_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_8[remaining];
      __m256 va = _mm256_maskload_ps(a + i, mask);
      __m256 vb = _mm256_maskload_ps(b + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_sub_ps(va, vb));
    }
  } else {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_sub_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_4[remaining];
      __m256d va = _mm256_maskload_pd(a + i, mask);
      __m256d vb = _mm256_maskload_pd(b + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_sub_pd(va, vb));
    }
  }
}

// c = a * b
template <class T>
void avx2_mul(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  if constexpr (std::is_same_v<T, float>) {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      _mm256_store_ps(c + i, _mm256_mul_ps(va, vb));
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_8[remaining];
      __m256 va = _mm256_maskload_ps(a + i, mask);
      __m256 vb = _mm256_maskload_ps(b + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_mul_ps(va, vb));
    }
  } else {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      _mm256_store_pd(c + i, _mm256_mul_pd(va, vb));
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_4[remaining];
      __m256d va = _mm256_maskload_pd(a + i, mask);
      __m256d vb = _mm256_maskload_pd(b + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_mul_pd(va, vb));
    }
  }
}

// c = a / b
template <class T>
void avx2_div(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  if constexpr (std::is_same_v<T, float>) {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_div_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_8[remaining];
      __m256 va = _mm256_maskload_ps(a + i, mask);
      __m256 vb = _mm256_maskload_ps(b + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_div_ps(va, vb));
    }
  } else {
    size_t i = 0;
    for (; i <= n - pack_size; i += pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);

      // 将双精度数视为整数处理，调整指数部分快速近似倒数
      const __m256d magic = _mm256_set1_pd(1.9278640450003146e-284);  // 魔法常数
      const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000);

      __m256d x = _mm256_or_pd(vb, magic);  // 防止除零
      __m256i xi = _mm256_castpd_si256(x);
      xi = _mm256_sub_epi64(_mm256_set1_epi64x(0x7FE0000000000000), xi);
      xi = _mm256_and_si256(xi, exp_mask);  // 保留指数部分
      __m256d recip = _mm256_castsi256_pd(xi);

      // __m256d recip = _mm256_rcp_pd_approx(vb);
      const __m256d two = _mm256_set1_pd(2.0);
      recip = _mm256_mul_pd(recip, _mm256_fnmadd_pd(vb, recip, two));  // FMA计算 (2 - v*recip)
      recip = _mm256_mul_pd(recip, _mm256_fnmadd_pd(vb, recip, two));  // FMA计算 (2 - v*recip)

      __m256d vc = _mm256_mul_pd(va, recip);
      _mm256_store_pd(c + i, vc);
    }
    size_t remaining = n - i;
    if (remaining > 0) {
      __m256i mask = mask_table_4[remaining];
      __m256d va = _mm256_maskload_pd(a + i, mask);
      __m256d vb = _mm256_maskload_pd(b + i, mask);

      // 将双精度数视为整数处理，调整指数部分快速近似倒数
      const __m256d magic = _mm256_set1_pd(1.9278640450003146e-284);  // 魔法常数
      const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000);

      __m256d x = _mm256_or_pd(vb, magic);  // 防止除零
      __m256i xi = _mm256_castpd_si256(x);
      xi = _mm256_sub_epi64(_mm256_set1_epi64x(0x7FE0000000000000), xi);
      xi = _mm256_and_si256(xi, exp_mask);  // 保留指数部分
      __m256d recip = _mm256_castsi256_pd(xi);

      // __m256d recip = _mm256_rcp_pd_approx(vb);
      const __m256d two = _mm256_set1_pd(2.0);
      recip = _mm256_mul_pd(recip, _mm256_fnmadd_pd(vb, recip, two));  // FMA计算 (2 - v*recip)
      recip = _mm256_mul_pd(recip, _mm256_fnmadd_pd(vb, recip, two));  // FMA计算 (2 - v*recip)

      _mm256_maskstore_pd(c + i, mask, _mm256_mul_pd(va, recip));
    }
  }
}

// ========================================================
// 基础累积运算 c ?= a
// c += a
template <class T>
void avx2_add_inplace(const T* a, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  for (size_t i = 0; i < n; i += pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_add_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_add_pd(vc, va));
    }
  }
}

// c -= a
template <class T>
void avx2_sub_inplace(const T* a, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  for (size_t i = 0; i < n; i += pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_sub_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_sub_pd(vc, va));
    }
  }
}

// c *= a
template <class T>
void avx2_mul_inplace(const T* a, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  for (size_t i = 0; i < n; i += pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_mul_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_mul_pd(vc, va));
    }
  }
}

// c /= a
template <class T>
void avx2_div_inplace(const T* a, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;

  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_div_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_div_pd(vc, va));
    }
  }
}
// ========================================================

// ========================================================
// FMA操作
// d[i] = a[i] * b[i] + c[i]
template <class T>
void avx2_fma_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmadd_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmadd_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = a[i] * b[i] + c[i]
template <class T>
void avx2_fma_c_abc(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmadd_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmadd_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// d[i] = a[i] * b[i] - c[i]
template <class T>
void avx2_fms_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmsub_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmsub_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = a[i] * b[i] - c[i]
template <class T>
void avx2_fms_c_abc(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmsub_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmsub_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// d[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnma_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmadd_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmadd_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnma_c_abc(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmadd_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmadd_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// d[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnms_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmsub_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmsub_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnms_c_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmsub_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmsub_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}
// ========================================================

// ========================================================
// 辅助AVX2复制函数
template <class T>
void avx2_copy(const T* src, T* dest, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      _mm256_store_ps(dest + i, _mm256_load_ps(src + i));
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      _mm256_store_pd(dest + i, _mm256_load_pd(src + i));
    }
  }
}
// ... 其他融合指令
// ========================================================

// ========================================================
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

// ========================================================
// 表达式模板惰性计算  张量-张量
template <typename L, typename R>
class AddExpr : public Expr<AddExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  AddExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  template <class T>
  void eval_to_impl(T* __restrict dest) const {
    avx2_add(lhs.data(), rhs.data(), dest, lhs.size());
  }
};

template <typename L, typename R>
class SubExpr : public Expr<SubExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  SubExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  template <class T>
  void eval_to_impl(T* __restrict dest) const {
    avx2_sub(lhs.data(), rhs.data(), dest, lhs.size());
  }
};

template <typename L, typename R>
class MulExpr : public Expr<MulExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  MulExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  template <class T>
  void eval_to_impl(T* __restrict dest) const {
    avx2_mul(lhs.data(), rhs.data(), dest, lhs.size());
  }
};

template <typename L, typename R>
class DivExpr : public Expr<DivExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  DivExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  template <class T>
  void eval_to_impl(T* __restrict dest) const {
    avx2_div(lhs.data(), rhs.data(), dest, lhs.size());
  }
};
// ========================================================

// ========================================================
// 运算符重载返回表达式
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
// ========================================================

// ========================================================
// 标量运算
// 加法：c[i] = a[i] + scalar
template <typename T>
void avx2_add_scalar(const T* a, T scalar, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  // 生成标量向量（提前生成避免循环内重复创建）
  typename SimdConfig<T>::simd_type scalar_vec;
  if constexpr (std::is_same_v<T, float>) {
    scalar_vec = _mm256_set1_ps(scalar);
  } else {
    scalar_vec = _mm256_set1_pd(scalar);
  }

  // 主循环处理完整SIMD块
  for (; i <= n - pack_size; i += pack_size) {
    typename SimdConfig<T>::simd_type a_vec;
    if constexpr (std::is_same_v<T, float>) {
      a_vec = _mm256_load_ps(a + i);
      _mm256_store_ps(c + i, _mm256_add_ps(a_vec, scalar_vec));
    } else {
      a_vec = _mm256_load_pd(a + i);
      _mm256_store_pd(c + i, _mm256_add_pd(a_vec, scalar_vec));
    }
  }

  // 掩码处理尾部元素
  size_t remaining = n - i;
  if (remaining > 0) {
    typename SimdConfig<T>::simd_type a_vec;
    __m256i mask;

    if constexpr (std::is_same_v<T, float>) {
      // 生成float掩码（8元素）
      mask = mask_table_8[remaining];
      a_vec = _mm256_maskload_ps(a + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_add_ps(a_vec, scalar_vec));
    } else {
      // 生成double掩码（4元素）
      mask = mask_table_4[remaining];
      a_vec = _mm256_maskload_pd(a + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_add_pd(a_vec, scalar_vec));
    }
  }
}

// 减法：c[i] = a[i] - scalar
template <typename T>
void avx2_sub_scalar(const T* a, T scalar, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  // 生成标量向量（提前生成避免循环内重复创建）
  typename SimdConfig<T>::simd_type scalar_vec;
  if constexpr (std::is_same_v<T, float>) {
    scalar_vec = _mm256_set1_ps(scalar);
  } else {
    scalar_vec = _mm256_set1_pd(scalar);
  }

  // 主循环处理完整SIMD块
  for (; i <= n - pack_size; i += pack_size) {
    typename SimdConfig<T>::simd_type a_vec;
    if constexpr (std::is_same_v<T, float>) {
      a_vec = _mm256_load_ps(a + i);
      _mm256_store_ps(c + i, _mm256_sub_ps(a_vec, scalar_vec));
    } else {
      a_vec = _mm256_load_pd(a + i);
      _mm256_store_pd(c + i, _mm256_sub_pd(a_vec, scalar_vec));
    }
  }

  // 掩码处理尾部元素
  size_t remaining = n - i;
  if (remaining > 0) {
    typename SimdConfig<T>::simd_type a_vec;
    __m256i mask;

    if constexpr (std::is_same_v<T, float>) {
      // 生成float掩码（8元素）
      mask = mask_table_8[remaining];
      a_vec = _mm256_maskload_ps(a + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_sub_ps(a_vec, scalar_vec));
    } else {
      // 生成double掩码（4元素）
      mask = mask_table_4[remaining];
      a_vec = _mm256_maskload_pd(a + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_sub_pd(a_vec, scalar_vec));
    }
  }
}

// 乘法：c[i] = a[i] * scalar
template <typename T>
void avx2_mul_scalar(const T* a, T scalar, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  // 生成标量向量（提前生成避免循环内重复创建）
  typename SimdConfig<T>::simd_type scalar_vec;
  if constexpr (std::is_same_v<T, float>) {
    scalar_vec = _mm256_set1_ps(scalar);
  } else {
    scalar_vec = _mm256_set1_pd(scalar);
  }

  // 主循环处理完整SIMD块
  for (; i <= n - pack_size; i += pack_size) {
    typename SimdConfig<T>::simd_type a_vec;
    if constexpr (std::is_same_v<T, float>) {
      a_vec = _mm256_load_ps(a + i);
      _mm256_store_ps(c + i, _mm256_mul_ps(a_vec, scalar_vec));
    } else {
      a_vec = _mm256_load_pd(a + i);
      _mm256_store_pd(c + i, _mm256_mul_pd(a_vec, scalar_vec));
    }
  }

  // 掩码处理尾部元素
  size_t remaining = n - i;
  if (remaining > 0) {
    typename SimdConfig<T>::simd_type a_vec;
    __m256i mask;

    if constexpr (std::is_same_v<T, float>) {
      // 生成float掩码（8元素）
      mask = mask_table_8[remaining];
      a_vec = _mm256_maskload_ps(a + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_mul_ps(a_vec, scalar_vec));
    } else {
      // 生成double掩码（4元素）
      mask = mask_table_4[remaining];
      a_vec = _mm256_maskload_pd(a + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_mul_pd(a_vec, scalar_vec));
    }
  }
}

// 除法：c[i] = a[i] / scalar
template <typename T>
void avx2_div_scalar(const T* a, T scalar, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  // 生成标量向量（提前生成避免循环内重复创建）
  typename SimdConfig<T>::simd_type scalar_vec;
  if constexpr (std::is_same_v<T, float>) {
    scalar_vec = _mm256_set1_ps(scalar);
  } else {
    scalar_vec = _mm256_set1_pd(scalar);

    // 除法优化
    // 将双精度数视为整数处理，调整指数部分快速近似倒数
    const __m256d magic = _mm256_set1_pd(1.9278640450003146e-284);  // 魔法常数
    const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000);

    __m256d x = _mm256_or_pd(scalar_vec, magic);  // 防止除零
    __m256i xi = _mm256_castpd_si256(x);
    xi = _mm256_sub_epi64(_mm256_set1_epi64x(0x7FE0000000000000), xi);
    xi = _mm256_and_si256(xi, exp_mask);  // 保留指数部分
    __m256d recip = _mm256_castsi256_pd(xi);

    const __m256d two = _mm256_set1_pd(2.0);
    recip = _mm256_mul_pd(recip, _mm256_fnmadd_pd(scalar_vec, recip, two));  // FMA计算 (2 - v*recip)
    recip = _mm256_mul_pd(recip, _mm256_fnmadd_pd(scalar_vec, recip, two));  // FMA计算 (2 - v*recip)
    scalar_vec = recip;
  }

  // 主循环处理完整SIMD块
  for (; i <= n - pack_size; i += pack_size) {
    typename SimdConfig<T>::simd_type a_vec;
    if constexpr (std::is_same_v<T, float>) {
      a_vec = _mm256_load_ps(a + i);
      _mm256_store_ps(c + i, _mm256_div_ps(a_vec, scalar_vec));
    } else {
      a_vec = _mm256_load_pd(a + i);
      _mm256_store_pd(c + i, _mm256_mul_pd(a_vec, scalar_vec));
    }
  }

  // 掩码处理尾部元素
  size_t remaining = n - i;
  if (remaining > 0) {
    typename SimdConfig<T>::simd_type a_vec;
    __m256i mask;

    if constexpr (std::is_same_v<T, float>) {
      // 生成float掩码（8元素）
      mask = mask_table_8[remaining];
      a_vec = _mm256_maskload_ps(a + i, mask);
      _mm256_maskstore_ps(c + i, mask, _mm256_div_ps(a_vec, scalar_vec));
    } else {
      // 生成double掩码（4元素）
      mask = mask_table_4[remaining];
      a_vec = _mm256_maskload_pd(a + i, mask);
      _mm256_maskstore_pd(c + i, mask, _mm256_mul_pd(a_vec, scalar_vec));
    }
  }
}
// ========================================================

// ========================================================
// 标量运算

// template <typename L, typename R>
// AddExpr<L, R> operator+(const Expr<L>& lhs, const Expr<R>& rhs) {
//   return AddExpr<L, R>(lhs.derived(), rhs.derived());
// }

// template <typename L, typename R>
// SubExpr<L, R> operator-(const Expr<L>& lhs, const Expr<R>& rhs) {
//   return SubExpr<L, R>(lhs.derived(), rhs.derived());
// }

// template <typename L, typename R>
// MulExpr<L, R> operator*(const Expr<L>& lhs, const Expr<R>& rhs) {
//   return MulExpr<L, R>(lhs.derived(), rhs.derived());
// }

// template <typename L, typename R>
// DivExpr<L, R> operator/(const Expr<L>& lhs, const Expr<R>& rhs) {
//   return DivExpr<L, R>(lhs.derived(), rhs.derived());
// }
// ========================================================

#endif  // HEADER_AVX2_H_