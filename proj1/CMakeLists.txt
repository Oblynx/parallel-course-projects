cmake_minimum_required(VERSION 2.8.12)
project(parallel-course-projects)

find_package(Threads REQUIRED)
#include_directories(
#  include
#)
add_compile_options(-std=c++11 -Wall -O2 -pthread)

#######################   Project1   ###############################
add_executable(sample-bitonic sample_bitonic.c) 
add_executable(pbitonic-stdthread src/pbitonic-stdthread.cpp)
target_link_libraries(pbitonic-stdthread ${CMAKE_THREAD_LIBS_INIT})
