cmake_minimum_required(VERSION 2.6.4)
project(apsp-gpu-mpi)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

# MPI with cmake:
# https://cmake.org/pipermail/cmake/2011-June/045037.html
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
  include
  ${MPI_INCLUDE_PATH}
  ${MPI_INCLUDE_PATH}/mpich
)

message(STATUS MPI_LIBRARIES: ${MPI_LIBRARIES})
message(STATUS MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH})
message(STATUS MPIEXEC: ${MPIEXEC})
message(STATUS MPIEXEC_NUMPROC_FLAGS: ${MPIEXEC_NUMPROC_FLAGS})
message(STATUS MPIEXEC_PREFLAGS: ${MPIEXEC_PREFLAGS})
message(STATUS MPIEXEC_POSTFLAGS: ${MPIEXEC_POSTFLAGS})

#set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_CXX_FLAGS "-Wall -O2 -march=native")
list(APPEND CUDA_NVCC_FLAGS "-O2 --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg -Wno-deprecated-gpu-targets --default-stream per-thread")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-Wall -g -Og -fno-omit-frame-pointer -fno-optimize-sibling-calls -march=native")
  add_definitions(-DDEBUG)
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

#add_definitions(-DLOG)
#add_definitions(-DNO_TEST)

#######################   Project4   ###############################
### Util libs ###
add_library(mpi_handler src/mpi_handler.cpp)
target_link_libraries(mpi_handler ${MPI_LIBRARIES})
cuda_add_library(kernels src/kernels.cu)
cuda_add_library(kernels_test src/kernels_test.cu)

### Task libs ###
add_library(tests src/tests.cpp)
target_link_libraries(tests
  kernels_test
)
add_library(algorithm src/algo.cpp)
target_link_libraries(algorithm
  mpi_handler
  kernels
)

### Execs ###
add_executable(makeGraph src/makeGraph.cpp)
add_executable(apsp src/main.cpp)
target_link_libraries(apsp
  mpi_handler
  tests
  algorithm
)

if(MPI_COMPILE_FLAGS)
  set_target_properties(mpi_handler PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(algorithm PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  set_target_properties(apsp PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  set_target_properties(mpi_handler PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(algorithm PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
  set_target_properties(apsp PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
