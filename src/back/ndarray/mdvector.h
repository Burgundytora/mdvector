#include <array>
#include <immintrin.h>
// #include <mkl.h>
#include <memory>
#include <type_traits>

template <typename T, size_t N>
class MDVector {
 public:
  // 构造函数与维度校验...

  // 核心运算方法
  void add(const MDVector& rhs) {
#ifdef MDVECTOR_USE_MKL
    mkl_add_impl(data_, rhs.data_, total_elements_);
#elif defined(MDVECTOR_USE_AVX2)
    avx2_add_impl(data_, rhs.data_, total_elements_);
#else
    scalar_add_impl(data_, rhs.data_, total_elements_);
#endif
  }

  void multiply(const MDVector& rhs, MDVector& result) const {
    static_assert(N == 2, "Matrix multiply requires 2D");
#ifdef MDVECTOR_USE_MKL
    mkl_gemm_impl(data_, rhs.data_, result.data_);
#elif defined(MDVECTOR_USE_AVX2)
    avx2_gemm_impl(data_, rhs.data_, result.data_);
#else
    scalar_gemm_impl(data_, rhs.data_, result.data_);
#endif
  }

 private:
  // MKL实现
  void mkl_add_impl(T* a, const T* b, size_t n) {
    if constexpr (std::is_same_v<T, float>) {
      vsAdd(n, a, b, a);
    } else {
      vdAdd(n, a, b, a);
    }
  }

  void mkl_gemm_impl(const T* a, const T* b, T* c) const {
    constexpr T alpha = 1.0, beta = 0.0;
    cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dims_[0], dims_[1], rhs.dims_[1], alpha, a, dims_[1], b,
               rhs.dims_[1], beta, c, dims_[1]);
  }

  // AVX2实现
  void avx2_add_impl(T* a, const T* b, size_t n) {
    if constexpr (std::is_same_v<T, float>) {
      constexpr size_t SIMD_WIDTH = 8;
      size_t aligned_n = (n / SIMD_WIDTH) * SIMD_WIDTH;
      for (size_t i = 0; i < aligned_n; i += SIMD_WIDTH) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        _mm256_store_ps(a + i, _mm256_add_ps(va, vb));
      }
      // 尾部处理...
    }
    // 双精度实现类似...
  }

  void avx2_gemm_impl(const T* a, const T* b, T* c) const {
    if constexpr (std::is_same_v<T, float>) {
      // AVX2优化的GEMM核心循环
      for (int i = 0; i < dims_[0]; ++i) {
        for (int k = 0; k < dims_[1]; ++k) {
          __m256 a_vec = _mm256_set1_ps(a[i * dims_[1] + k]);
          for (int j = 0; j < dims_[1]; j += 8) {
            __m256 b_vec = _mm256_load_ps(b + k * dims_[1] + j);
            __m256 c_vec = _mm256_load_ps(c + i * dims_[1] + j);
            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_store_ps(c + i * dims_[1] + j, c_vec);
          }
        }
      }
    }
  }
};