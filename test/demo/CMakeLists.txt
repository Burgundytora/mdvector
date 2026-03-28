include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(demo_mdvector demo_mdvector.cc)

add_mingw_link(demo_mdvector)
