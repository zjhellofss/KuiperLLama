cmake_minimum_required(VERSION 3.22)
project(lc)

set(CMAKE_CXX_STANDARD 20)
find_package(glog REQUIRED)
add_executable(lc main.cpp)

target_link_libraries(lc glog::glog)
target_include_directories(lc PUBLIC ${glog_INCLUDE_DIR})
