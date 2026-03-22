add_executable(benchmark_2d 2d/benchmark_2d.cc)
add_executable(benchmark_3d 3d/benchmark_3d.cc)

add_mingw_link(benchmark_2d)
add_mingw_link(benchmark_3d)
