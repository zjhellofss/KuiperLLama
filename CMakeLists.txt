cmake_minimum_required(VERSION 3.22)
project(lc)

set(CMAKE_CXX_STANDARD 20)
find_package(glog REQUIRED)
aux_source_directory(./source/tensor/ DIR_TENSOR)
aux_source_directory(./source/alloc/ DIR_ALLOC)
aux_source_directory(./source/buffer/ DIR_BUFFER)
add_executable(lc main.cpp ${DIR_TENSOR} ${DIR_ALLOC} ${DIR_BUFFER})

target_link_libraries(lc glog::glog)
target_include_directories(lc PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(lc PUBLIC ./include)
