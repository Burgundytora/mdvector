#ifndef __MDVECTOR_SIMD_H__
#define __MDVECTOR_SIMD_H__

#if defined(USE_AVX2)
#include "x86_avx2.h"

#elif defined(USE_AVX512)
#include "x86_avx512.h"

#elif defined(USE_SSE)
#include "x86_sse.h"

#elif defined(USE_NEON)
#include "arm_neon.h"

#elif defined(USE_RVV)
#include "risc_v.h"

#else
#include "x86_avx2.h"  // 默认avx2

#endif

#endif  // __SIMD_H__