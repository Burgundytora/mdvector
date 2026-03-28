
#include "include_md_all.h"

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  md::vector<double, 2> a(2, 3);
  a.fill(0.1);

  double sum;

  sum = std::reduce(a.begin(), a.end());
  std::cout << "mdvector: sum of 6 * 0.1 = " << sum << " (expected 0.6)\n";

  a.arange(1.0);

  sum = std::reduce(a.begin(), a.end());
  std::cout << "mdvector: sum of 1~6 = " << sum << " (expected 21)\n";

  std::cout << "mdvector: print 1~6 : ";
  a.print();

  auto slice = a.span(1, all());

  sum = std::reduce(slice.begin(), slice.end());
  std::cout << "span: sum of span 4~6 = " << sum << " (expected 15)\n";

  std::cout << "span: print 4~6 : ";
  slice.print();

  std::cout << "stl test down. \n";

  return 0;
}