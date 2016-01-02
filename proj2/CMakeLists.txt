cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_COMPILER mpic++)
project(parallel-course-projects)

include_directories(
  include
  #"/usr/lib/openmpi/include"
)
#set(MPI_libs "/usr/lib/openmpi/lib")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O2 -march=native")
#set(CMAKE_C_FLAGS "-Wall -O2 -march=native")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -g -O1 -fno-omit-frame-pointer -march=native -fPIC")
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

#######################   Project2   ###############################
add_library(kNNAlgo src/kNNAlgo.cpp)
add_library(utils src/utils.cpp)
add_library(mpi_handler src/mpi_handler.cpp)

add_executable(kNNsingle src/kNN_single_main.cpp)
target_link_libraries(kNNsingle
  kNNAlgo
  utils
  mpi_handler
)
