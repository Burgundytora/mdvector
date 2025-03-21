#ifndef __BLAS_FUNCTION_H__
#define __BLAS_FUNCTION_H__

#include <cmath>

// #include "cblas.h"

void blas_add(const double* a, const double* b, double* result, int n) {
  // #pragma simd
  for (int i = 0; i < n; ++i) {
    result[i] = a[i] + b[i];
  }
}

void blas_sub(const double* a, const double* b, double* result, int n) {
  // #pragma simd
  for (int i = 0; i < n; ++i) {
    result[i] = a[i] - b[i];
  }
}

void blas_mul(const double* a, const double* b, double* result, int n) {
  // #pragma simd
  for (int i = 0; i < n; ++i) {
    result[i] = a[i] * b[i];
  }
}

void blas_div(const double* a, const double* b, double* result, int n) {
  // #pragma simd
  for (int i = 0; i < n; ++i) {
    result[i] = a[i] / b[i];
  }
}

#endif  // __BLAS_FUNCTION_H__