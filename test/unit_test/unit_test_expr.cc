
#include "include_md_all.h"

int main(int args, char *argv[]) {
  md::vector<double, 2> a(1, 3);
  md::vector<double, 2> b(1, 3);
  md::vector<double, 2> c(1, 3);
  md::vector<double, 2> res(1, 3);
  a.fill(1.0);
  b.fill(2.0);
  c.fill(3.0);

  auto tmp_plus = a + b;
  auto tmp_multi = a * b;

  res = c + tmp_plus;

  res.print();

  return 0;
}