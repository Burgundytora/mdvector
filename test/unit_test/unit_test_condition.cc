#include "include_md_all.h"

#include <iostream>

template <typename T, typename Expected>
bool check_values(const md::vector<T, 1>& actual, Expected expected, const char* name) {
  for (size_t i = 0; i < actual.size(); ++i) {
    const T value = static_cast<T>(expected(i));
    if (actual[i] != value) {
      std::cerr << name << " failed at " << i << ": expected " << value << ", got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

template <typename T>
bool check_span_and_view_tail() {
  constexpr size_t pack_size = md::simd<T>::pack_size;
  const size_t count = pack_size + 1;

  md::vector<T, 1> contiguous(count + 2);
  for (size_t i = 0; i < contiguous.size(); ++i) contiguous[i] = static_cast<T>(i + 1);
  const T left_guard = contiguous[0];
  const T right_guard = contiguous[count + 1];

  md::span<T, 1> span(contiguous.data() + 1, std::array<size_t, 1>{count});
  if (span.remaining_size() != 1) return false;
  md::vector<T, 1> span_result = span + static_cast<T>(10);
  if (!check_values(span_result, [](size_t i) { return static_cast<T>(i + 12); }, "span tail load")) return false;
  span = span + static_cast<T>(20);
  if (contiguous[0] != left_guard || contiguous[count + 1] != right_guard) return false;
  for (size_t i = 0; i < count; ++i) {
    if (contiguous[i + 1] != static_cast<T>(i + 22)) return false;
  }

  md::vector<T, 1> strided(count * 2 + 3);
  strided.fill(static_cast<T>(-7));
  for (size_t i = 0; i < count; ++i) strided[1 + i * 2] = static_cast<T>(i + 1);
  md::view<T, 1> view(strided.data() + 1, std::array<size_t, 1>{count}, std::array<size_t, 1>{2});
  if (view.remaining_size() != 1) return false;
  md::vector<T, 1> view_result = view * static_cast<T>(3);
  if (!check_values(view_result, [](size_t i) { return static_cast<T>((i + 1) * 3); }, "view tail load")) return false;
  view = view + static_cast<T>(30);
  for (size_t i = 0; i < count; ++i) {
    if (strided[1 + i * 2] != static_cast<T>(i + 31)) return false;
    if (strided[2 + i * 2] != static_cast<T>(-7)) return false;
  }
  return true;
}

int main() {
  const size_t count = md::simd<double>::pack_size * 2 + 1;
  md::vector<double, 1> a(count);
  md::vector<double, 1> b(count);
  for (size_t i = 0; i < count; ++i) {
    a[i] = static_cast<double>(i) - 2.0;
    b[i] = static_cast<double>(count - i) - 3.0;
  }

  auto delayed = md::where((a > b) || (a == 0.0), a + 100.0, b - 100.0);
  md::vector<double, 1> selected = delayed;
  if (!check_values(
          selected, [&](size_t i) { return (a[i] > b[i] || a[i] == 0.0) ? a[i] + 100.0 : b[i] - 100.0; }, "where"))
    return 1;

  md::vector<double, 1> result = a == b;
  if (!check_values(result, [&](size_t i) { return a[i] == b[i]; }, "equal")) return 1;
  result = a != b;
  if (!check_values(result, [&](size_t i) { return a[i] != b[i]; }, "not equal")) return 1;
  result = a < b;
  if (!check_values(result, [&](size_t i) { return a[i] < b[i]; }, "less")) return 1;
  result = a <= b;
  if (!check_values(result, [&](size_t i) { return a[i] <= b[i]; }, "less equal")) return 1;
  result = a > b;
  if (!check_values(result, [&](size_t i) { return a[i] > b[i]; }, "greater")) return 1;
  result = a >= b;
  if (!check_values(result, [&](size_t i) { return a[i] >= b[i]; }, "greater equal")) return 1;
  result = (a > 0.0) && (b > 0.0);
  if (!check_values(result, [&](size_t i) { return a[i] > 0.0 && b[i] > 0.0; }, "logical and")) return 1;
  result = (a > 0.0) || (b > 0.0);
  if (!check_values(result, [&](size_t i) { return a[i] > 0.0 || b[i] > 0.0; }, "logical or")) return 1;
  result = !(a > 0.0);
  if (!check_values(result, [&](size_t i) { return !(a[i] > 0.0); }, "logical not")) return 1;
  result = md::logical_xor(a > 0.0, b > 0.0);
  if (!check_values(result, [&](size_t i) { return (a[i] > 0.0) != (b[i] > 0.0); }, "logical xor")) return 1;

  md::vector<double, 1> scalar_select = md::select(a >= 0.0, 1.0, -1.0);
  if (!check_values(scalar_select, [&](size_t i) { return a[i] >= 0.0 ? 1.0 : -1.0; }, "select")) return 1;

  if (!check_span_and_view_tail<float>() || !check_span_and_view_tail<double>() || !check_span_and_view_tail<int>()) {
    std::cerr << "span/view SIMD tail check failed\n";
    return 1;
  }

  return 0;
}
