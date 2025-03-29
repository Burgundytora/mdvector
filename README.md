# mdvector - 多维高性能SIMD向量库


**mdvector** 是一个**轻量级头文件**形式基于现代C++的多维数组计算库，通过 **SIMD指令集优化** 和 **表达式模板技术**，在元素级运算（Element-wise）场景下达到**接近手写汇编极限性能**，同时支持**python风格切片操作**与切片的高性能计算。

**mdvector** is a **lightweight header-only** multidimensional array computing library based on modern C++. Through **SIMD instruction set optimization** and **expression template techniques**, it achieves **near hand-written assembly performance** in element-wise operations while supporting **Python-style slicing operations** and high-performance computation on slices.


## 🚀 核心特性

### 1. 极致性能优化
- **SIMD 全指令集支持**：SSE/AVX2/AVX512（x86）、NEON（ARM）、RISC-V自动适配，内存对齐与尾部掩码处理，相比手写指令集基本无性能损失仅
- **表达式模板**：复杂运算（如 `res = a + b - c * d / e`）零临时变量开销

### 2. 多维灵活操作
- **任意维度支持**：通过 `std::mdspan`（C++23）或自定义实现（C++17）
- **安全索引**：`vec.at(d1,d2,d3)`（边界检查）与 **快速索引** `vec(d1,d2,d3)`
- **惰性视图**：支持自定义指针偏移实现切片（`subspan`） 切片同样支持高性能表达式操作

### 3. 内存安全设计
- **Rust风格安全证明**：所有 `unsafe` 操作可以绑定维度类型来静态验证
- **编译期形状检查**：通过类型系统确保维度一致性

### 4. 跨平台兼容
- **宏 自动适配**：x86/ARM/RISC-V 架构无缝切换
- **编译器友好**：GCC/Clang/MSVC 全支持
- **轻量级**：只需包含单个头文件
- **兼容性**：需要C++17标准即可，无需C++23的标准库mdspan等特性


## 📊 性能对比
- **md expr为此项目**   
- **hwy为google-highway**
- **expr为基于for循环的表达式模板**
<div align="center">
  <img src="docs/images/win-2d.png" width="90%">
  <p><em>性能对比(越高越好)</em></p>
</div>
<div align="center">
  <img src="docs/images/linux-2d.png" width="90%">
  <p><em>性能对比(越高越好)</em></p>
</div>
<div align="center">
  <img src="docs/images/win-3d.png" width="90%">
  <p><em>性能对比(越高越好)</em></p>
</div>
<div align="center">
  <img src="docs/images/linux-3d.png" width="90%">
  <p><em>性能对比(越高越好)</em></p>
</div>

## 📦 快速开始

### 使用
```bash
git clone https://github.com/Burgundytora/mdvector.git
代码中include mdvector.h头文件即可 cmake指令集与编译选项参考附带cmake文件夹
