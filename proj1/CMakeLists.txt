cmake_minimum_required(VERSION 2.8.12)
project(parallel-course-projects)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
#message(${CMAKE_MODULE_PATH})
find_package(Threads REQUIRED)
#find_package(TBB REQUIRED)
set(TBB_INCLUDE_DIR "/home/oblivion/ws/var/tbb-4.3/src/include/")
set(TBB_LIBRARY
  "/home/oblivion/ws/var/tbb-4.3/src/build/linux_intel64_gcc_cc5.2.0_libc2.19_kernel3.13.0_release/libtbb.so.2") 
set(TBB_MALLOC_LIBRARY 
  "/home/oblivion/ws/var/tbb-4.3/src/build/linux_intel64_gcc_cc5.2.0_libc2.19_kernel3.13.0_release/libtbbmalloc.so.2") 
include_directories(
  include
  #${TBB_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIR}
)
#message(${TBB_INCLUDE_DIRS})
#message(${TBB_LIBRARY_DIRS})
# link_directories(${TBB_LIBRARY_DIRS})
add_compile_options(-std=c++14 -Wall -O0)

#######################   Project1   ###############################
#add_executable(sample-bitonic sample_bitonic.c) 

add_library(thread_pool src/thread_pool.cpp)
add_executable(pbitonic-stdthread src/pbitonic-stdthread.cpp)
target_link_libraries(pbitonic-stdthread
  ${CMAKE_THREAD_LIBS_INIT}
  thread_pool
  ${TBB_LIBRARY}
  ${TBB_MALLOC_LIBRARY}
)
