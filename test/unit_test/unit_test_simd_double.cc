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

int test_simd_double_op() {
  int errors = 0;
  // 1) set1/load/store
  alignas(64) double src[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  alignas(64) double dst[16] = {0.0};

  auto a = md::simd<double>::set1(5.0);
  md::simd<double>::store(dst, a);
  errors += check(dst[0] == 5.0 && dst[1] == 5.0, "simd<double>::set1/store");

  auto b = md::simd<double>::load(src);
  md::simd<double>::store(dst, b);
  errors +=
      check(dst[0] == 1.0 && dst[1] == 2.0 && dst[2] == 3.0 && dst[3] == 4.0, "simd<double>::load/store round-trip");

  // 2) add/sub/mul/div
  auto sum = md::simd<double>::add(a, b);
  auto diff = md::simd<double>::sub(a, b);
  auto mul = md::simd<double>::mul(a, b);
  auto div = md::simd<double>::div(md::simd<double>::set1(8.0), md::simd<double>::set1(2.0));

  alignas(64) double tmp[4];
  md::simd<double>::store(tmp, sum);
  errors += check(tmp[0] == 6.0 && tmp[1] == 7.0, "simd<double>::add");
  md::simd<double>::store(tmp, diff);
  errors += check(tmp[0] == 4.0 && tmp[1] == 3.0, "simd<double>::sub");
  md::simd<double>::store(tmp, mul);
  errors += check(tmp[0] == 5.0 && tmp[1] == 10.0, "simd<double>::mul");
  md::simd<double>::store(tmp, div);
  errors += check(tmp[0] == 4.0, "simd<double>::div");

  // 3) mask load/store semantics (remaining < pack_size)
  alignas(64) double src2[16] = {10.0, 20.0, 30.0, 40.0};
  alignas(64) double dst2[16] = {0.0};
  auto m = md::simd<double>::mask_load(src2, 2);
  md::simd<double>::mask_store(dst2, 2, m);
  errors += check(dst2[0] == 10.0 && dst2[1] == 20.0 && dst2[2] == 0.0, "simd<double>::mask_load/mask_store");

  // 4) loadu/storeu
  alignas(64) double src3[4] = {7.0, 8.0, 9.0, 10.0};
  alignas(64) double dst3[4] = {0.0};
  auto nu = md::simd<double>::loadu(src3);
  md::simd<double>::storeu(dst3, nu);
  errors += check(dst3[0] == 7.0 && dst3[3] == 10.0, "simd<double>::loadu/storeu");

  std::println("{} {}", dst3[0], dst3[3]);

  return errors;
}

int test_vector_double_op() {
  int errors = 0;
  std::cout << "=== md::vector<double> operations test ===\n";
  md::vector<double, 2> v(2, 2);
  v.fill(3.0);

  std::cout << "  compute operations...\n";
  auto r = v + 2.0;
  auto s = r * 3.0;
  auto t = s - v;
  std::cout << "  operations ready, prepare divisor...\n";
  md::vector<double, 2> divisor(2, 2);
  divisor.fill(2.0);
  std::cout << "  divisor filled, test vector division...\n";
  auto q = t / divisor;
  std::cout << "  division done\n";

  md::vector<double, 2> s_eval = s;  // force evaluate delayed expression
  std::cout << "  s_eval done\n";
  md::vector<double, 2> t_eval = t;
  std::cout << "  t_eval done\n";
  md::vector<double, 2> q_eval = q;
  std::cout << "  q_eval done\n";

  errors += check(s_eval(0, 0) == 15.0 && t_eval(0, 0) == 12.0 && q_eval(0, 0) == 6.0,
                  "md::vector<double> arithmetic via expression templates");

  md::vector<double, 2> expected(2, 2);
  expected.fill(12.0);
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
  errors += check(equal, "md::vector<double> result content validation");

  md::vector<double, 2> expected_div(2, 2);
  expected_div.fill(6.0);
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
  errors += check(equal, "md::vector<double> division result content validation");

  return errors;
}

int main() {
  int errors = 0;
  std::cout << "=== simd double unit test ===\n";
  errors += test_simd_double_op();
  errors += test_vector_double_op();
  if (errors == 0) {
    std::cout << "ALL PASS\n";
  } else {
    std::cerr << "FAIL COUNT: " << errors << "\n";
  }
  return errors;
}