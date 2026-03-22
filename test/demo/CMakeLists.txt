include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(demo_mdspan demo_mdspan.cc)

add_mingw_link(demo_mdspan)
