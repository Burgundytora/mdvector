add_executable(benchmark_2d 2d/benchmark_2d.cc)
add_executable(benchmark_3d 3d/benchmark_3d.cc)
add_executable(benchmark_reduction benchmark_reduction.cc)


if(UNIX)
    target_link_libraries(benchmark_2d PRIVATE "pthread")
    target_link_libraries(benchmark_3d PRIVATE "pthread")
    target_link_libraries(benchmark_reduction PRIVATE "pthread")
endif()

add_mingw_link(benchmark_2d)
add_mingw_link(benchmark_3d)
add_mingw_link(benchmark_reduction)
