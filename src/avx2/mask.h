#ifndef __MASK_H__
#define __MASK_H__

#include <immintrin.h>

// ==================== 尾部元素使用AVX2掩码 =======================

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

// 除法加速
const __m256d magic = _mm256_set1_pd(1.9278640450003146e-284);  // 魔法常数
const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000);
const __m256i exp_mask2 = _mm256_set1_epi64x(0x7FE0000000000000);

#endif  // __MASK_H__