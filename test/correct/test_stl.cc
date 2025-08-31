#include <string>

#include "mdvector.h"

using md::all;
using md::slice;

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  vector_2d<double> a({2, 3});
  a.set_value(0.1);

  double sum;

  sum = std::reduce(a.begin(), a.end());
  std::cout << "mdvector: sum of 6 * 0.1 = " << sum << " (expected 0.6)\n";

  double val = 1.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      a(i, j) = val;
      val += 1.0;
    }
  }

  sum = std::reduce(a.begin(), a.end());
  std::cout << "mdvector: sum of 1~6 = " << sum << " (expected 21)\n";

  std::cout << "mdvector: print 1~6 : ";
  for (const auto &it : a) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  auto slice = a.span(1, all());

  sum = std::reduce(slice.begin(), slice.end());
  std::cout << "span: sum of span 4~6 = " << sum << " (expected 15)\n";

  std::cout << "span: print 4~6 : ";
  for (const auto &it : slice) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  return 0;
}