cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_COMPILER mpic++.mpich)
# Run execs with mpirun.mpich?
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
  set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -g -Og -fno-omit-frame-pointer -march=native -fPIC")
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)
add_definitions(-DBATCH)

#######################   Project2   ###############################
add_library(kNNAlgo src/kNNAlgo.cpp)
add_library(utils src/utils.cpp)
add_library(mpi_handler src/mpi_handler.cpp)

add_executable(kNN src/kNN_main.cpp)
target_link_libraries(kNN
  kNNAlgo
  utils
  mpi_handler
)

#######################   TESTS   ###############################
add_executable(test_kNNsingle src/test_kNNsingle.cpp)
target_link_libraries(test_kNNsingle
  kNNAlgo
  utils
  mpi_handler
)
add_executable(test_mpiTransfers src/test_mpi_transfers.cpp)
target_link_libraries(test_mpiTransfers
  kNNAlgo
  utils
  mpi_handler
)

