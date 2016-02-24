cmake_minimum_required(VERSION 2.6.4)
project(cuda-project)

find_package(CUDA)

include_directories(
  include
)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g -Og -fno-omit-frame-pointer -march=native")
  add_definitions(-DDEBUG)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

#######################   Project3   ###############################
add_executable(makeGraph src/makeGraph.cpp)
cuda_add_executable(fw src/main.cu)
