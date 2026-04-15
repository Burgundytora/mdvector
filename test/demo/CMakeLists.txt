include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(demo_mdvector demo_mdvector.cc)

if(UNIX)
    target_link_libraries(demo_mdvector PRIVATE "pthread")
endif()

add_mingw_link(demo_mdvector)