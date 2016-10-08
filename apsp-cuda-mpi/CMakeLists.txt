cmake_minimum_required(VERSION 2.6.4)
project(project4)

# MPI with cmake:
# https://cmake.org/pipermail/cmake/2011-June/045037.html
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
  include
  ${MPI_INCLUDE_PATH}
)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native")
list(APPEND CUDA_NVCC_FLAGS "-O2 --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg
-Wno-deprecated-gpu-targets --default-stream per-thread")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g -Og -fno-omit-frame-pointer -march=native")
  add_definitions(-DDEBUG)
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

#add_definitions(-DLOG)
add_definitions(-DNO_TEST)
#add_definitions(-D__GXX_EXPERIMENTAL_CXX0X__)

#######################   Project4   ###############################
### Util libs ###
add_library(mpi_handler src/mpi_handler.cpp)
target_link_libraries(mpi_handler ${MPI_LIBRARIES})
cuda_add_library(utils src/utils.cu)
cuda_add_library(kernels src/kernels.cu)
target_link_libraries(kernels utils)

### Task libs ###
add_library(run_cpu src/run_cpu.cpp)
target_link_libraries(run_cpu utils)
add_library(run_gpu src/run_gpu.cpp)
target_link_libraries(run_gpu
  mpi_handler
  utils
  kernels
)
add_library(slave src/slave.cpp)
target_link_libraries(slave
  mpi_handler
  utils
  kernels
)

### Execs ###
add_executable(makeGraph src/makeGraph.cpp)
add_executable(fw src/main.cpp)
target_link_libraries(fw
  utils
  mpi_handler
  run_cpu
  run_gpu
  slave
)

if(MPI_COMPILE_FLAGS)
  set_target_properties(mpi_handler PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(fw PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(mpi_handler PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(fw PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
