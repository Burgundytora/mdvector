#pragma once

#include <iostream>
#include <type_traits>

#include "simd_op_tags.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
#include "arch_x86/avx512.h"
#include "arch_x86/avx512_binary.h"
#include "arch_x86/avx512_select.h"
#include "arch_x86/avx512_unary.h"
#elif defined(__AVX2__)
#include "arch_x86/avx2.h"
#include "arch_x86/avx2_binary.h"
#include "arch_x86/avx2_select.h"
#include "arch_x86/avx2_unary.h"
#elif defined(__SSE4_1__)
#include "arch_x86/sse.h"
#include "arch_x86/sse_binary.h"
#include "arch_x86/sse_select.h"
#include "arch_x86/sse_unary.h"
#else
#include "arch_common/none.h"
#include "arch_common/none_binary.h"
#include "arch_common/none_select.h"
#include "arch_common/none_unary.h"
#endif
#elif defined(__arm__) || defined(__aarch64__)
#include "arch_arm/arm_neon.h"
#include "arch_arm/neon_binary.h"
#include "arch_arm/neon_select.h"
#include "arch_arm/neon_unary.h"
#elif defined(__riscv)
#include "arch_risc/risc_v.h"
#include "arch_risc/risc_v_binary.h"
#include "arch_risc/risc_v_select.h"
#include "arch_risc/risc_v_unary.h"
#else
#include "arch_common/none.h"
#include "arch_common/none_binary.h"
#include "arch_common/none_select.h"
#include "arch_common/none_unary.h"
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
