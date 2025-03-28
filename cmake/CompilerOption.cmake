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
else()
	add_compile_options("-Werror")
	if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    	# 尝试使用lld链接器
    	find_program(LLD_PATH "lld")
		if(LLD_PATH)
			set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
		else()
			message(WARNING "lld not found, falling back to default linker")
		endif()
		# Clang需要同时设置编译和链接标志
		add_compile_options(-flto=thin)  # 或者使用 -flto
		add_link_options(-flto=thin)     # 必须同时添加链接选项
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		add_definitions("-flto")
	endif()
endif()