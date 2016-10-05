cmake_minimum_required(VERSION 2.6.4)
project(cuda-project)

find_package(CUDA REQUIRED)

include_directories(
  include
)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native")
list(APPEND CUDA_NVCC_FLAGS " -O2 --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g -Og -fno-omit-frame-pointer -march=native")
  add_definitions(-DDEBUG)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

#add_definitions(-DLOG)
add_definitions(-DNO_TEST)

#######################   Project3   ###############################
add_library(run_cpu src/run_cpu.cpp)
cuda_add_library(run_gpu src/run_gpu.cu)
cuda_add_library(utils src/utils.cu)

add_executable(makeGraph src/makeGraph.cpp)
add_executable(fw src/main.cpp)
target_link_libraries(fw
  utils
  run_cpu
  run_gpu
)
