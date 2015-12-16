cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_COMPILER g++)
project(parallel-course-projects)

include_directories(include)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native")
set(CMAKE_C_FLAGS "-Wall -O2 -march=native")
### Debugging options
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -g -O1 -fno-omit-frame-pointer -march=native -fPIC")
  add_definitions(-D__DEBUG__)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#add_definitions(-DBATCH_EXPERIMENTS)

#######################   Project2   ###############################
add_library(kNNAlgo src/kNNAlgo.cpp)
add_library(utils src/utils.cpp)

add_executable(kNNsingle src/kNN_single_main.cpp)
target_link_libraries(kNNsingle
  kNNAlgo
  utils
)
