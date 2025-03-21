**MDVector设计思想**


1.多维支持：支持任意维度数组，通过std::mdspan(C++23标准库)，或者自定义mdspan(C++17可变参数模板结合预计算strides)，提供unsafe索引方式vec(d1, d2, d3)  vec[d1, d2, d3]，以及safe索引方式vec.at(d1, d2, d3)


2.高性能element-wise计算：基于SIMD指令集流水线计算，首地址内存对齐，尾部元素高性能掩码处理，小数据高频计算下，与avx2指令性能仅损失15%，显著高于绝大部分第三方库。相比eigen在windows下小元素计算优势较明显，其他情况下性能基本持平。相比STL或其他库，性能高度稳定，可在各个平台、编译器、指令集环境下实现性能最优计算。


3.复杂表达式支持：基于CRTP静态多态思想，实现表达式模板嵌套SIMD指令集技术，支持复杂表达式（例如res=a+b-c*d/e），可对+-*/外操作符进行重载


4.内存安全性：借鉴rust内存安全编程思想，进行unsafe操作前，需先给编译器"证明"这个操作是安全的，每个MDVector需要绑定一个shape类型，创建n个不同尺寸的MDVector前先定义n个shape类型，进行elementwise计算进行编译期shape静态类型检查


5.跨平台：支持x86架构avx2，x86-avx512以及arm-neon需集成，各平台开发完后通过宏与constexpr可自动适配simd


6.扩展性：针对性能热点/复杂数学运算，可拓展专用SIMD函数特例，或者表达式模板特例/偏特例，可达到手写汇编级别极限性能


7.切片操作/部分视图：通过STL标准库submdspan(C++26 编译器未)实现多维数组原生切片视图，或自定义指针操作实现，另外可借助C++最新STL的ranges与view实现轻量级视图


8.标量操作：表达式模板进行标量scalar特例，去除不必要性能开销


9.除法速度优化：除法的cpu周期比其他运算高一个量级，使用魔法常数+两次牛顿迭代进行优化（理论误差1e-11），需核实


10.基本算法：如reduce，transform，find等，可直接使用vector STL，对于部分切片的算法，如果切片内存连续，使用基于指针的迭代器结合STL，非内存连续切片的操作，需要单独适配


11.三角函数：成员整体/内存连续切片使用SIMD指令实现（查表法、魔法公式、常用级数公式，可参考vectorclass），非连续切片遍历实现


12.惰性计算：带实现，可通过C++20新特性view视图实现，需使用C++17以上，或自定义实现，通过指针+偏移。


13.GEMM：不在考虑范围内，对于2d-shape，可通模板特例，结合eigen的map实现无开销映射，然后调用eigen进行GEMM
