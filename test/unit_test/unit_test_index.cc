
#include "include_md.h"

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  md::vector<double, 3> a({2, 3, 4});

  std::println("layout_right shape: {}", a.extents());
  std::println("1d index of [1,1,4]: [1*3*4+1*4+3] = {}", a.get_1d_index(1, 1, 3));
  std::println("md index of [19]: [1,1,3] = {}", a.get_md_index(19));

  md::vector<double, 2, std::layout_left> b({3, 4});

  std::println("layout_right shape: {}", b.extents());
  std::println("1d index of [2,1]: [2+1*3] = {}", b.get_1d_index(2, 1));
  std::println("md index of [5]: [2,1] = {}", b.get_md_index(5));

  std::cout << "index test down. \n";

  return 0;
}