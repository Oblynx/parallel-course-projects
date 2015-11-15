cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_COMPILER g++)
project(parallel-course-projects)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
set(TBB_LIBRARY
  "/home/oblivion/ws/var/tbb-4.4/lib/intel64/gcc4.4/libtbb.so.2"
) 
set(TBB_MALLOC_LIBRARY 
  "/home/oblivion/ws/var/tbb-4.4/lib/intel64/gcc4.4/libtbbmalloc.so.2"
)
include_directories(include)
#add_compile_options(-std=c++14 -Wall -g -O1 -fno-omit-frame-pointer -march=native -fPIC)
add_compile_options(-std=c++11 -Wall -O2 -march=native)
#add_definitions(-D__DEBUG__)

#######################   Project1   ###############################
add_executable(sample-bitonic sample_bitonic.c) 

add_library(thread_pool src/thread_pool.cpp)
target_link_libraries(thread_pool
  PUBLIC ${TBB_LIBRARY}
  PUBLIC ${TBB_MALLOC_LIBRARY}
  PUBLIC ${CMAKE_THREAD_LIBS_INIT}
)
add_executable(pbitonic-stdthread src/pbitonic-stdthread.cpp)
target_link_libraries(pbitonic-stdthread
  thread_pool
  #-fsanitize=thread
)
add_executable(pbitonic-omp src/pbitonic-omp.cpp)
target_compile_options(pbitonic-omp PUBLIC -fopenmp)
target_link_libraries(pbitonic-omp -fopenmp)
