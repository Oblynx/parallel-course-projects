cmake_minimum_required(VERSION 2.6.4)
# Run execs with mpirun.mpich?
project(parallel-course-projects)

find_package(MPI REQUIRED)
include_directories(
  include
  ${MPI_INCLUDE_PATH}
)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g -Og -fno-omit-frame-pointer -march=native -fPIC")
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)
add_definitions(-DBATCH)
add_definitions(-DNDEBUG)
add_definitions(-D__GXX_EXPERIMENTAL_CXX0X__)

#######################   Project2   ###############################
add_library(kNNAlgo src/kNNAlgo.cpp)
add_library(utils src/utils.cpp)
add_library(mpi_handler src/mpi_handler.cpp)
target_link_libraries(mpi_handler ${MPI_LIBRARIES})

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

