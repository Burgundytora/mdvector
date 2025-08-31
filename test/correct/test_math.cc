#include <string>

#include "mdvector.h"

using md::all;
using md::slice;
using md::span;

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  vector_2d<double> a({2, 3});
  a.set_value(0.1);

  double val = 0.1;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      a(i, j) = val;
      val += 0.1;
    }
  }

  std::cout << "mdvector: print 0.1~0.6 : ";
  for (const auto &it : a) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  vector_2d<double> a_cos_plus_10 = a.cos() + 10.0;
  std::cout << "mdvector: print cos(0.1~0.6)+10 : ";
  for (const auto &it : a_cos_plus_10) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  vector_2d<double> a_exp_2_plus_1 = a.exp(2) + 1.0;
  std::cout << "mdvector: print exp 2^(0.1~0.6)+1 : ";
  for (const auto &it : a_exp_2_plus_1) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  std::cout << "span: print 0.1~0.3 : ";
  for (const auto &it : a.span(0, all())) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  std::cout << "span: print 0.4~0.6 : ";
  for (const auto &it : a.span(1, all())) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  std::cout << "span: print (0.1~0.3).sin() : ";
  for (const auto &it : a.span(0, all()).sin()) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  std::cout << "span: print sin(0.1~0.3)+10 : ";
  vector_1d<double> temp = sin(a.span(0, all())) + 10.0;
  for (const auto &it : temp) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  std::cout << "span: print (0.4~0.6)^3 : ";
  for (const auto &it : a.span(1, all()).pow(3)) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  return 0;
}