#include "include_md_all.h"
#include <iostream>
#include <array>
#include <algorithm>

static int check(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
  }
  std::cout << "PASS: " << msg << "\n";
  return 0;
}

int test_simd_float_op() {
  int errors = 0;
  // 1) set1/load/store
  alignas(64) float src[16] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  alignas(64) float dst[16] = {0.0f};

  auto a = md::simd<float>::set1(5.0f);
  md::simd<float>::store(dst, a);
  errors += check(dst[0] == 5.0f && dst[1] == 5.0f, "simd<float>::set1/store");

  auto b = md::simd<float>::load(src);
  md::simd<float>::store(dst, b);
  errors +=
      check(dst[0] == 1.0f && dst[1] == 2.0f && dst[2] == 3.0f && dst[3] == 4.0f, "simd<float>::load/store round-trip");

  // 2) add/sub/mul/div
  auto sum = md::simd<float>::add(a, b);
  auto diff = md::simd<float>::sub(a, b);
  auto mul = md::simd<float>::mul(a, b);
  auto div = md::simd<float>::div(md::simd<float>::set1(8.0f), md::simd<float>::set1(2.0f));

  alignas(64) float tmp[4];
  md::simd<float>::store(tmp, sum);
  errors += check(tmp[0] == 6.0f && tmp[1] == 7.0f, "simd<float>::add");
  md::simd<float>::store(tmp, diff);
  errors += check(tmp[0] == 4.0f && tmp[1] == 3.0f, "simd<float>::sub");
  md::simd<float>::store(tmp, mul);
  errors += check(tmp[0] == 5.0f && tmp[1] == 10.0f, "simd<float>::mul");
  md::simd<float>::store(tmp, div);
  errors += check(tmp[0] == 4.0f, "simd<float>::div");

  // 3) mask load/store semantics (remaining < pack_size)
  alignas(64) float src2[16] = {10.0f, 20.0f, 30.0f, 40.0f};
  alignas(64) float dst2[16] = {0.0f};
  auto m = md::simd<float>::mask_load(src2, 2);
  md::simd<float>::mask_store(dst2, 2, m);
  errors += check(dst2[0] == 10.0f && dst2[1] == 20.0f && dst2[2] == 0.0f, "simd<float>::mask_load/mask_store");

  // 4) loadu/storeu
  alignas(64) float src3[8] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
  alignas(64) float dst3[8] = {0.0f};
  auto nu = md::simd<float>::loadu(src3);
  md::simd<float>::storeu(dst3, nu);
  errors += check(dst3[0] == 7.0f && dst3[7] == 14.0f, "simd<float>::loadu/storeu");

  return errors;
}

int test_vector_float_op() {
  int errors = 0;
  std::cout << "=== md::vector<float> operations test ===\n";
  md::vector<float, 2> v(2, 2);
  v.fill(3.0f);

  std::cout << "  compute operations...\n";
  auto r = v + 2.0f;
  auto s = r * 3.0f;
  auto t = s - v;
  auto q = t / 2.0f;
  std::cout << "  operations ready, prepare divisor...\n";

  md::vector<float, 2> s_eval = s;  // force evaluate delayed expression
  std::cout << "  s_eval done\n";
  md::vector<float, 2> t_eval = t;
  std::cout << "  t_eval done\n";
  md::vector<float, 2> q_eval = q;
  std::cout << "  q_eval done\n";

  errors += check(s_eval(0, 0) == 15.0f && t_eval(0, 0) == 12.0f && q_eval(0, 0) == 6.0f,
                  "md::vector<float> arithmetic via expression templates");

  md::vector<float, 2> expected(2, 2);
  expected.fill(12.0f);
  bool equal = true;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      if (t_eval(i, j) != expected(i, j)) {
        equal = false;
        break;
      }
    }
    if (!equal) break;
  }
  errors += check(equal, "md::vector<float> result content validation");

  md::vector<float, 2> expected_div(2, 2);
  expected_div.fill(6.0f);
  equal = true;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      if (q_eval(i, j) != expected_div(i, j)) {
        equal = false;
        break;
      }
    }
    if (!equal) break;
  }
  errors += check(equal, "md::vector<float> division result content validation");

  return errors;
}

int main() {
  int errors = 0;
  std::cout << "=== simd float unit test ===\n";
  errors += test_simd_float_op();
  errors += test_vector_float_op();
  if (errors == 0) {
    std::cout << "ALL PASS\n";
  } else {
    std::cerr << "FAIL COUNT: " << errors << "\n";
  }
  return errors;
}