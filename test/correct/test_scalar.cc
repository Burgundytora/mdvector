#include <string>

#include "mdvector.h"

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  vector_2d<double> a({2, 3});
  a.set_value(0.1);
  std::cout << " a  = " << a(0, 0) << " (expected 0.1)" << std::endl;

  vector_2d<double> temp = a;
  vector_2d<double> temp2 = a;
  std::cout << " new temp = a =" << temp(0, 0) << " (expected 0.1)" << std::endl;
  std::cout << "\n";

  int i = 1;

  ///////////////////////////////////////
  // 基础测试1~4
  {
    temp = a + 0.1;
    std::cout << i++ << ": temp = a(0.1) + 0.1 = " << temp(0, 0) << " (expected 0.2)" << std::endl;

    temp = a - 0.1;
    std::cout << i++ << ": temp = a(0.1) - 0.1 = " << temp(0, 0) << " (expected 0.0)" << std::endl;

    temp = a * 0.1;
    std::cout << i++ << ": temp = a(0.1) * 0.1 = " << temp(0, 0) << " (expected 0.01)" << std::endl;

    temp = a / 0.1;
    std::cout << i++ << ": temp = a(0.1) / 0.1 = " << temp(1, 2) << " (expected 1)" << std::endl;

    std::cout << "\n";
  }

  ///////////////////////////////////////
  // 基础测试5~8
  {
    temp = 0.1 + a;
    std::cout << i++ << ": temp = 0.1 + a(0.1) = " << temp(0, 0) << " (expected 0.2)" << std::endl;

    temp = 0.1 - a;
    std::cout << i++ << ": temp = 0.1 - a(0.1) = " << temp(0, 0) << " (expected 0.0)" << std::endl;

    temp = 0.1 * a;
    std::cout << i++ << ": temp = 0.1 * a(0.1) = " << temp(1, 2) << " (expected 0.01)" << std::endl;

    temp = 0.1 / a;
    std::cout << i++ << ": temp = 0.1 / a(0.1) = " << temp(0, 0) << " (expected 1)" << std::endl;
    std::cout << "\n";
  }

  ///////////////////////////////////////
  // 混合运算9~11
  {
    temp = 0.2 + a + 0.1 + a;
    std::cout << i++ << ": temp = 0.1 + a(0.1) + 0.2 + a = " << temp(0, 0) << " (expected 0.5)" << std::endl;

    temp = 0.1 + a * 0.2 - 0.3 * 10.0;
    std::cout << i++ << ": temp = 0.1 + a(0.1) * 0.2 - 0.3 * 10.0 = " << temp(0, 0) << " (expected -2.88)" << std::endl;

    temp = (a + 0.1) * 2.0 / (a - 0.2);
    std::cout << i++ << ": temp = (a(0.1) + 0.1) * 2.0 / (a(0.1) - 0.2) = " << temp(0, 0) << " (expected -4)"
              << std::endl;
    std::cout << "\n";
  }

  ///////////////////////////////////////
  // ?=运算12~15
  {
    temp = a;
    temp += (0.1 + a * 0.2 - 0.3 * 10.0);
    std::cout << i++ << ": temp(0.1) += 0.1 + a(0.1) * 0.2 - 0.3 * 10.0 = " << temp(0, 0) << " (expected -2.78)"
              << std::endl;

    temp = a;
    temp -= (0.1 + a * 0.2 - 0.3 * 10.0);
    std::cout << i++ << ": temp(0.1) -= 0.1 + a(0.1) * 0.2 - 0.3 * 10.0 = " << temp(0, 0) << " (expected 2.98)"
              << std::endl;
    temp = a;

    temp *= (0.1 + a * 0.2 - 0.3 * 10.0);
    std::cout << i++ << ": temp(0.1) *= 0.1 + a(0.1) * 0.2 - 0.3 * 10.0 = " << temp(0, 0) << " (expected -0.288)"
              << std::endl;

    temp = a;
    temp /= (0.1 + a * 0.2 - 0.3 * 10.0);
    std::cout << i++ << ": temp(0.1) /= 0.1 + a(0.1) * 0.2 - 0.3 * 10.0 = " << temp(0, 0) << " (expected -0.0347222)"
              << std::endl;
    std::cout << "\n";
  }

  return 0;
}