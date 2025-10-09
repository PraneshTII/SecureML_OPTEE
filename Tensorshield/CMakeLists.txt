cmake_minimum_required(VERSION 3.13)
project (tensorshield LANGUAGES C CXX)
option(DARKNET_GPU "Enable OpenCL support" OFF)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(PROJECT_VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 11)
# REMOVE THESE TWO LINES:
# set(CMAKE_C_COMPILER "gcc")
# set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
include(GNUInstallDirs)

set (DARKNET_BINARY darknetp)
set (DARKNET_LIBRARY darknetp_l)
set (DARKNET_LIBRARY_STATIC darknetp_s)

find_package(Threads REQUIRED)

if (DARKNET_GPU)
	find_package(OpenCL REQUIRED)
	add_library(opencl INTERFACE IMPORTED)
	set_property(TARGET opencl PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})
	set_property(TARGET opencl PROPERTY
		INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARY})
	find_package(clBLAS REQUIRED PATHS ${clBLAS_DIR})
	add_library(clblas INTERFACE IMPORTED)
	set_property(TARGET clblas PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${CLBLAS_INCLUDE_DIRS})
	set_property(TARGET clblas PROPERTY
		INTERFACE_LINK_LIBRARIES ${CLBLAS_LIBRARIES})
endif()

list (APPEND DARKNET_CFLAGS "-Os")
list (APPEND DARKNET_CFLAGS "-Wfatal-errors")
list (APPEND DARKNET_CFLAGS "-Wno-unused-variable")
list (APPEND DARKNET_CFLAGS "-Wno-unused-result")
list (APPEND DARKNET_CFLAGS "-Wno-unknown-pragmas")
list (APPEND DARKNET_CFLAGS "-Wno-deprecated-declarations")

FILE(GLOB SRC host/src/*.c)
FILE(GLOB EXAM host/examples/*.c)
set(MAIN host/main.c)

include_directories(${CMAKE_SOURCE_DIR}/export/usr/include)
link_directories(${CMAKE_SOURCE_DIR}/export/usr/lib)

add_executable (${PROJECT_NAME} ${SRC} ${EXAM} ${MAIN})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${DARKNET_DEFINITIONS})
target_compile_options(${PROJECT_NAME} PRIVATE ${DARKNET_CFLAGS})
target_include_directories(${PROJECT_NAME}
				 PRIVATE host/include
				 PRIVATE host/src
				 PRIVATE include)

if (DARKNET_GPU)
target_link_libraries (${PROJECT_NAME} PRIVATE opencl clblas teec m Threads::Threads)
else()
target_link_libraries (${PROJECT_NAME} PRIVATE teec m Threads::Threads)
endif()

install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
