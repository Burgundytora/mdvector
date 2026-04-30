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

int test_simd_int_op() {
  int errors = 0;
  // 1) set1/load/store
  alignas(64) int src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  alignas(64) int dst[16] = {0};

  auto a = md::simd<int>::set1(5);
  md::simd<int>::store(dst, a);
  errors += check(dst[0] == 5 && dst[1] == 5, "simd<int>::set1/store");

  auto b = md::simd<int>::load(src);
  md::simd<int>::store(dst, b);
  errors += check(dst[0] == 1 && dst[1] == 2 && dst[2] == 3 && dst[3] == 4, "simd<int>::load/store round-trip");

  // 2) add/sub/mul/div
  auto sum = md::simd<int>::add(a, b);
  auto diff = md::simd<int>::sub(a, b);
  auto mul = md::simd<int>::mul(a, b);
  auto div = md::simd<int>::div(md::simd<int>::set1(8), md::simd<int>::set1(2));

  alignas(64) int tmp[4];
  md::simd<int>::store(tmp, sum);
  errors += check(tmp[0] == 6 && tmp[1] == 7, "simd<int>::add");
  md::simd<int>::store(tmp, diff);
  errors += check(tmp[0] == 4 && tmp[1] == 3, "simd<int>::sub");
  md::simd<int>::store(tmp, mul);
  errors += check(tmp[0] == 5 && tmp[1] == 10, "simd<int>::mul");
  md::simd<int>::store(tmp, div);
  errors += check(tmp[0] == 4, "simd<int>::div");

  // 3) mask load/store semantics (remaining < pack_size)
  alignas(64) int src2[16] = {10, 20, 30, 40};
  alignas(64) int dst2[16] = {0};
  auto m = md::simd<int>::mask_load(src2, 2);
  md::simd<int>::mask_store(dst2, 2, m);
  errors += check(dst2[0] == 10 && dst2[1] == 20 && dst2[2] == 0, "simd<int>::mask_load/mask_store");

  // 4) loadu/storeu
  alignas(64) int src3[8] = {7, 8, 9, 10, 11, 12, 13, 14};
  alignas(64) int dst3[8] = {0};
  auto nu = md::simd<int>::loadu(src3);
  md::simd<int>::storeu(dst3, nu);
  errors += check(dst3[0] == 7 && dst3[7] == 14, "simd<int>::loadu/storeu");

  return errors;
}

int test_vector_int_op() {
  int errors = 0;
  std::cout << "=== md::vector<int> operations test ===\n";
  md::vector<int, 2> v(2, 2);
  v.fill(3);

  std::cout << "  compute operations...\n";
  auto r = v + 2;
  auto s = r * 3;
  auto t = s - v;
  auto q = t / 2;

  md::vector<int, 2> s_eval = s;  // force evaluate delayed expression
  std::cout << "  s_eval done\n";
  md::vector<int, 2> t_eval = t;
  std::cout << "  t_eval done\n";
  md::vector<int, 2> q_eval = q;
  std::cout << "  q_eval done\n";

  errors += check(s_eval(0, 0) == 15 && t_eval(0, 0) == 12 && q_eval(0, 0) == 6,
                  "md::vector<int> arithmetic via expression templates");

  md::vector<int, 2> expected(2, 2);
  expected.fill(12);
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
  errors += check(equal, "md::vector<int> result content validation");

  md::vector<int, 2> expected_div(2, 2);
  expected_div.fill(6);
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
  errors += check(equal, "md::vector<int> division result content validation");

  return errors;
}

int main() {
  int errors = 0;
  std::cout << "=== simd int unit test ===\n";
  errors += test_simd_int_op();
  errors += test_vector_int_op();
  if (errors == 0) {
    std::cout << "ALL PASS\n";
  } else {
    std::cerr << "FAIL COUNT: " << errors << "\n";
  }
  return errors;
}
