
#include <string>

#include "ndarray.h"
#include "test_type.h"

int main(int args, char *argv[]) {
  // 检查类型满足SIMD
  // CheckTypeSIMD<double>();
  // CheckTypeSIMD<float>();
  // CheckTypeSIMD<bool>();
  // CheckTypeSIMD<int>();
  // CheckTypeSIMD<quat>();
  // CheckTypeSIMD<pos>();      // 异常, 长度为24, 不满足2的幂次 需手动对齐 或创建幽灵成员
  // CheckTypeSIMD<posSIMD>();  // 正确  通过alignal指令对齐32
  // // CheckTypeSIMD<array<int, 3>>(); // 正确  array是POD
  // CheckTypeSIMD<vector<bool>>();  // 错误  vector不是POD

  // 打印类型信息
  // std::cout << typeid(double).name() << ", " << sizeof(double) << "\n";
  // std::cout << typeid(float).name() << ", " << sizeof(float) << "\n";
  // std::cout << typeid(bool).name() << ", " << sizeof(bool) << "\n";
  // std::cout << typeid(int).name() << ", " << sizeof(int) << "\n";
  // std::cout << typeid(quat).name() << ", " << sizeof(quat) << "\n";
  // std::cout << typeid(pos).name() << ", " << sizeof(pos) << "\n";
  // std::cout << typeid(posSIMD).name() << ", " << sizeof(posSIMD) << "\n";
  // std::cout << typeid(vector<bool>).name() << ", " << sizeof(vector<bool>) << "\n";

  // 各维度长度
  size_t a = 2;
  size_t b = 3;
  size_t c = 1;

  // 创建ndarray的shape
  _2d_shape ss = {a, b};

  // 创建ndarray
  _2d_array_double dat1(ss);                      // 使用定义好的shape构造
  _2d_array_double dat2(_2d_shape{2, 3});         // 临时创建shape构造
  _2d_array_double dat2_error1(_2d_shape{2, 4});  // 第二维长度不同 与data1进行运算会出错 +会unsafe Plus会抛出异常
  _2d_array_double dat2_direct({a, b});           // 调用array初始化列表
  _2d_array_double dat3(_2d_shape{2, 3});         // 同data1
  _3d_array_double dat4(_3d_shape{3, 3, 3});      // 同data1
  _3d_array_double dat5;                          // 先声明，后设置维度
  dat5.SetShape(_3d_shape{3, 3, 3});

  // 输入参数
  std::cout << "data1[1,1]:" << dat1[1, 1] << "\n";
  dat1.SetValue(0.5);
  std::cout << "data1[1,1]:" << dat1[1, 1] << "\n";
  dat1.SetValue(0.3);
  std::cout << "data1[1,1]:" << dat1[1, 1] << "\n";
  dat1.SetValue(0.1);
  std::cout << "data1[1,1]:" << dat1[1, 1] << "\n";
  dat2.SetValue(0.2);
  std::cout << "data2[1,1]:" << dat2[1, 1] << "\n";

  // 显示数据内容
  std::cout << "\ndata1:\n";
  dat1.ShowDataMatrixStyle();  // 矩阵形式
  std::cout << "\ndata2:\n";
  dat2.ShowDataMatrixStyle();  // 矩阵形式
  std::cout << "\ndata3:\n";
  dat3.ShowDataMatrixStyle();  // 向量形式

  // 执行向量化运算
  // 函数加法 safe
  dat3 = dat1.Plus(dat2);
  std::cout << "\nplus data3: 0.3\n";
  dat3.ShowDataMatrixStyle();
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";

  // 符号加法 unsafe
  dat3 = dat1 + dat2;
  std::cout << "\n+ data3: 0.3\n";
  dat3.ShowDataMatrixStyle();
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";

  dat3.PlusEqual(dat2);
  std::cout << "\nplusequal data3: 0.5\n";
  dat3.ShowDataMatrixStyle();
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";

  dat3 += dat2;
  std::cout << "\n+= data3: 0.7\n";
  dat3.ShowDataMatrixStyle();
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";

  dat3.MinusEqual(dat2);
  std::cout << "\nminusequal data3: 0.5\n";
  dat3.ShowDataMatrixStyle();
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";

  dat3 -= dat2;
  std::cout << "\n-= data3: 0.3\n";
  dat3.ShowDataMatrixStyle();
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";

  _2d_array_double dat_temp = dat1 * dat2;
  std::cout << "\n* dat_temp: 0.02\n";
  dat_temp.ShowDataMatrixStyle();
  std::cout << "dat_temp[1,1]:" << dat_temp[1, 1] << "\n";

  // 执行向量化运算 与基础类型
  dat_temp = dat1 + 0.0001;
  std::cout << "\n+ dat_temp: 0.1001\n";
  dat_temp.ShowDataMatrixStyle();
  std::cout << "dat_temp[1,1]:" << dat_temp[1, 1] << "\n";

  dat_temp += double(0.001);
  std::cout << "\n+= dat_temp: 0.1011\n";
  dat_temp.ShowDataMatrixStyle();
  std::cout << "dat_temp[1,1]:" << dat_temp[1, 1] << "\n";

  dat_temp *= 1000;
  std::cout << "\n*= dat_temp: 101.1\n";
  dat_temp.ShowDataMatrixStyle();
  std::cout << "dat_temp[1,1]:" << dat_temp[1, 1] << "\n";

  // dat1.MinusEqual(dat2_error);  // 维度不同 会抛出异常

  // 显示数据内容
  std::cout << "\ndata1:\n";
  dat1.ShowDataMatrixStyle();  // 矩阵形式
  std::cout << "\ndata2:\n";
  dat2.ShowDataMatrixStyle();  // 矩阵形式
  std::cout << "\ndata3:\n";
  dat3.ShowDataMatrixStyle();  // 向量形式

  // 多维索引方法
  // [i,j,k] C-style unsafe-fast
  std::cout << "data1[1,1]:" << dat1[1, 1] << "\n";
  std::cout << "data2[1,1]:" << dat2[1, 1] << "\n";
  std::cout << "data3[1,1]:" << dat3[1, 1] << "\n";
  // std::cout << "data3[9,9]:" << dat3[9, 9] << "\n";  // UB未定义行为
  // (i,j,k) eigen/xtensor-style unsafe-fast
  std::cout << "data1(1,1):" << dat1(1, 1) << "\n";
  std::cout << "data2(1,1):" << dat2(1, 1) << "\n";
  std::cout << "data3(1,1):" << dat3(1, 1) << "\n";
  // std::cout << "data3(9,9):" << dat3(9, 9) << "\n";  // UB未定义行为
  // .at(i,j,k) cxx-style safe-slow
  std::cout << "data1.at(1,1):" << dat1.at(1, 1) << "\n";
  std::cout << "data2.at(1,1):" << dat2.at(1, 1) << "\n";
  std::cout << "data3.at(1,1):" << dat3.at(1, 1) << "\n";
  // std::cout << "data3.at(9,9):" << dat3.at(9, 9) << "\n";  // 错误 索引越界

  // 正常完成
  std::cout << "down!" << std::endl;

  return 0;
}