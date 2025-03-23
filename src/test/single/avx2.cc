#include "src/avx2/function.h"
#include "src/test/test_set.h"

template <class T>
void test_avx2() {
  AlignedAllocator<T> allocator_;

  T* data1_ = allocator_.allocate(total_element);
  T* data2_ = allocator_.allocate(total_element);
  T* data3_ = allocator_.allocate(total_element);
  T* data4_ = allocator_.allocate(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
    data4_[i] = 4;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": avx2");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      avx2_add<T>(data1_, data2_, data3_, total_element);
      // avx2_add<T>(data4_, data3_, data3_, total_element);
    }

    if constexpr (do_sub) {
      avx2_sub<T>(data1_, data2_, data3_, total_element);
      avx2_sub<T>(data4_, data3_, data3_, total_element);
    }

    if constexpr (do_mul) {
      avx2_mul<T>(data1_, data2_, data3_, total_element);
      avx2_mul<T>(data4_, data3_, data3_, total_element);
    }

    if constexpr (do_div) {
      avx2_div<T>(data1_, data2_, data3_, total_element);
      avx2_div<T>(data4_, data3_, data3_, total_element);
    }
  }

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

int main(int args, char* argv[]) {
  // double
  test_avx2<double>();

  return 0;
}