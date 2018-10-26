project(ArrowCheck)
cmake_minimum_required(VERSION 3.11)

find_package(xtensor)
find_package(arrow)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(${xtensor_INCLUDE_DIRS})
include_directories(${arrow_INCLUDE_DIRS})

add_executable(xarrow main.cpp)

target_link_libraries(xarrow /usr/local/lib64/libarrow.so.11)