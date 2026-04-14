#include "core/dev_vector.h"

int main() {
  md::vector<double, 2> a({2, 2});
  md::vector<double, 2> b({2, 2});
  a.set_ones();
  a.print();

  a += 2.0;
  a.print();

  b.fill(10.0);
  a = a + b * a / b;
  a.print();
  return 0;
}