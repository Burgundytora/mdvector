#include <string>

#include "src/mdvector/mdvector.h"

int main(int args, char *argv[]) {
  // 各维度长度
  size_t a = 2;
  size_t b = 3;
  size_t c = 1;

  // 创建ndarray的shape
  mdshape_2d ss = {a, b};

  // 创建ndarray
  mdvector_2d<double> dat1(ss);                       // 使用定义好的shape构造
  mdvector_2d<double> dat2(mdshape_2d{2, 3});         // 临时创建shape构造
  mdvector_2d<double> dat2_error1(mdshape_2d{2, 4});  // 第二维长度不同 与data1进行运算会出错 +会unsafe Plus会抛出异常
  mdvector_2d<double> dat2_direct({a, b});            // 调用array初始化列表
  mdvector_2d<double> dat3(mdshape_2d{2, 3});         // 同data1
  mdvector_3d<double> dat4(mdshape_3d{3, 3, 3});      // 同data1
  mdvector_3d<double> dat5;                           // 先声明，后设置维度
  dat5.Reset(mdshape_3d{3, 3, 3});

  // 输入参数
  std::cout << "data1(1,1):" << dat1(1, 1) << "\n";
  dat1.set_value(0.5);
  std::cout << "data1(1,1):" << dat1(1, 1) << "\n";
  dat1.set_value(0.3);
  std::cout << "data1(1,1):" << dat1(1, 1) << "\n";
  dat1.set_value(0.1);
  std::cout << "data1(1,1):" << dat1(1, 1) << "\n";
  dat2.set_value(0.2);
  std::cout << "data2(1,1):" << dat2(1, 1) << "\n";

  // 显示数据内容
  std::cout << "\ndata1:\n";
  dat1.show_data_matrix_style();
  std::cout << "\ndata2:\n";
  dat2.show_data_matrix_style();
  std::cout << "\ndata3:\n";
  dat3.show_data_matrix_style();

  // 符号加法 unsafe
  dat3 = dat1 + dat2;
  std::cout << "\n+ data3: 0.3\n";
  dat3.show_data_matrix_style();
  std::cout << "data3(1,1):" << dat3(1, 1) << "\n";

  dat3 += dat2;
  std::cout << "\n+= data3: 0.5\n";
  dat3.show_data_matrix_style();
  std::cout << "data3(1,1):" << dat3(1, 1) << "\n";

  dat3 -= dat2;
  std::cout << "\n-= data3: 0.3\n";
  dat3.show_data_matrix_style();
  std::cout << "data3(1,1):" << dat3(1, 1) << "\n";

  mdvector_2d<double> dat_temp = dat1 * dat2;
  std::cout << "\n* dat_temp: 0.02\n";
  dat_temp.show_data_matrix_style();
  std::cout << "dat_temp(1,1):" << dat_temp(1, 1) << "\n";

  mdvector_2d<double> dat_temp2 = dat1 / dat2;
  std::cout << "\n* dat_temp2: 0.5\n";
  dat_temp2.show_data_matrix_style();
  std::cout << "dat_temp2(1,1):" << dat_temp2(1, 1) << "\n";

  // 执行向量化运算 与基础类型
  dat_temp = dat1 + 0.01;
  std::cout << "\n+ dat_temp: 0.11\n";
  dat_temp.show_data_matrix_style();
  std::cout << "dat_temp(1,1):" << dat_temp(1, 1) << "\n";

  dat_temp += double(0.001);
  std::cout << "\n+= dat_temp: 0.111\n";
  dat_temp.show_data_matrix_style();
  std::cout << "dat_temp(1,1):" << dat_temp(1, 1) << "\n";

  dat_temp *= static_cast<double>(1000);
  std::cout << "\n*= dat_temp: 111\n";
  dat_temp.show_data_matrix_style();
  std::cout << "dat_temp(1,1):" << dat_temp(1, 1) << "\n";

  // dat1.MinusEqual(dat2_error);  // 维度不同 会抛出异常

  // 显示数据内容
  std::cout << "\ndata1:\n";
  dat1.show_data_matrix_style();  // 矩阵形式
  std::cout << "\ndata2:\n";
  dat2.show_data_matrix_style();  // 矩阵形式
  std::cout << "\ndata3:\n";
  dat3.show_data_matrix_style();  // 向量形式

  // 多维索引方法
  // (i,j,k) eigen/xtensor-style unsafe-fast
  std::cout << "data1(1,1):" << dat1(1, 1) << "\n";
  std::cout << "data2(1,1):" << dat2(1, 1) << "\n";
  std::cout << "data3(1,1):" << dat3(1, 1) << "\n";
  //   std::cout << "data3(9,9):" << dat3(9, 9) << "\n";  // UB未定义行为
  // .at(i,j,k) cxx-style safe-slow
  std::cout << "data1.at(1,1):" << dat1.at(1, 1) << "\n";
  std::cout << "data2.at(1,1):" << dat2.at(1, 1) << "\n";
  std::cout << "data3.at(1,1):" << dat3.at(1, 1) << "\n";
  //   std::cout << "data3.at(9,9):" << dat3.at(9, 9) << "\n";  // 错误 索引越界

  // 正常完成
  std::cout << "down!" << std::endl;

  return 0;
}