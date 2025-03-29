#include <string>

#include "src/mdvector/mdvector.h"

int main(int args, char *argv[]) {
  mdvector_2d<double> a({2, 3});
  a.set_value(0.1);

  std::cout << "\n--- Testing multiplication ---" << std::endl;
  mdvector_2d<double> temp = a * 2.0;  // 应该得到0.2
  mdvector_2d<double> temp_vec = temp;

  std::cout << "\n--- Testing mixed ops ---" << std::endl;
  mdvector_2d<double> result = temp + 0.1;  // 应该得到0.3

  std::cout << "\nVerification:" << std::endl;
  std::cout << "a * 2.0 = " << temp_vec(0, 0) << " (expected 0.2)" << std::endl;
  std::cout << "a*2 + 0.1 = " << result(0, 0) << " (expected 0.3)" << std::endl;

  result = temp + 0.1;  // 应该得到0.3
  std::cout << "a*2 + 0.1 = " << result(0, 0) << " (expected 0.3)" << std::endl;

  result = 0.1 + temp;  // 应该得到0.3
  std::cout << " 0.1 +  a*2 = " << result(0, 0) << " (expected 0.3)" << std::endl;

  result = temp - 0.1;  // 应该得到0.1
  std::cout << "a*2 - 0.1 = " << result(0, 0) << " (expected 0.1)" << std::endl;

  result = temp * 0.1;  // 应该得到0.02
  std::cout << "a*2 * 0.1 = " << result(0, 0) << " (expected 0.02)" << std::endl;

  result = temp / 0.1;  // 应该得到2
  std::cout << "a*2 / 0.1 = " << result(0, 0) << " (expected 2)" << std::endl;

  return 0;
}