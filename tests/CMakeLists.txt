CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# test_executable
ADD_EXECUTABLE( test_executable 
        ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/cl_helper.h
)

TARGET_INCLUDE_DIRECTORIES( test_executable 
	PRIVATE "." )

TARGET_LINK_LIBRARIES( test_executable
	PUBLIC flexcore ${OpenCL_LIBRARIES})

TARGET_COMPILE_OPTIONS( test_executable
        PUBLIC "-std=c++14"
        PUBLIC "-DKERNEL_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
)

TARGET_INCLUDE_DIRECTORIES( test_executable PUBLIC
	${OpenCL_INCLUDE_DIRS}
)

