#ifndef __MDVECTOR_SIMD_SELECT__
#define __MDVECTOR_SIMD_SELECT__

#include <iostream>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
#include "arch_x86/avx512.h"
#elif defined(__AVX2__)
#include "arch_x86/avx2.h"
#elif defined(__SSE4_1__)
#include "arch_x86/sse.h"
#else
#include "arch_common/none.h"
#endif
#elif defined(__arm__) || defined(__aarch64__)
#include "arch_arm/arm_neon.h"
#elif defined(__riscv)
#include "arch_risc/risc_v.h"
#else
#include "arch_common/none.h"
#endif

namespace md {

void print_simd_type() {
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
  std::cout << "x86 avx512...\n";
#elif defined(__AVX2__)
  std::cout << "x86 avx2...\n";
#elif defined(__SSE4_1__)
  std::cout << "x86 sse...\n";
#else
  std::cout << "x86 none...\n";
#endif
#elif defined(__arm__) || defined(__aarch64__)
  std::cout << "arm neon...\n";
#elif defined(__riscv)
  std::cout << "riscv...\n";
#else
  std::cout << "none...\n";
#endif
}

}  // namespace md

#endif  // __MDVECTOR_SIMD_SELECT__