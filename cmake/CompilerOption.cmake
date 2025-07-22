# 编译器选项
if(MSVC)
  add_compile_options(/w)
  add_compile_options(/bigobj)
  add_definitions(
    -D_CRT_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_DEPRECATE
    -DNOMINMAX
  )
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /MANIFEST:NO")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO")
  add_compile_options(/utf-8) # MSVC专用UTF-8选项

else()
  add_compile_options("-Werror")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # # Clang需要同时设置编译和链接标志
    # add_compile_options(-flto=thin)  # 或者使用 -flto
    # add_link_options(-flto=thin)     # 必须同时添加链接选项
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # add_definitions("-flto")
  endif()
endif()
