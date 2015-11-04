cmake_minimum_required(VERSION 2.8.12)
project(parallel-course-projects)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
message(${CMAKE_MODULE_PATH})
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
include_directories(
  include
  ${TBB_INCLUDE_DIRS}
)
message(${TBB_INCLUDE_DIRS})
message(${TBB_LIBRARY_DIRS})
# link_directories(${TBB_LIBRARY_DIRS})
add_compile_options(-std=c++11 -Wall -O2 -pthread)

#######################   Project1   ###############################
add_executable(sample-bitonic sample_bitonic.c) 
add_executable(pbitonic-stdthread src/pbitonic-stdthread_noxtraelts.cpp)
target_link_libraries(pbitonic-stdthread
  ${CMAKE_THREAD_LIBS_INIT}
  ${TBB_LIBRARIES}
)
