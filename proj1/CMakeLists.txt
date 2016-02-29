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
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native")
set(CMAKE_C_FLAGS "-Wall -O2 -march=native")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -g -O1 -fno-omit-frame-pointer -march=native -fPIC")
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#add_definitions(-DBATCH_EXPERIMENTS)

#######################   Project1   ###############################
add_executable(sample-bitonic sample_bitonic.c) 

add_library(thread_pool src/thread_pool.cpp)
target_link_libraries(thread_pool
  ${TBB_LIBRARY}
  ${TBB_MALLOC_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
)
add_executable(pbitonic-stdthread src/pbitonic-stdthread.cpp)
target_link_libraries(pbitonic-stdthread
  thread_pool
  #-fsanitize=thread
)
add_executable(pbitonic-omp src/pbitonic-omp.cpp)
set_target_properties(pbitonic-omp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
target_link_libraries(pbitonic-omp ${OpenMP_CXX_FLAGS})
