cmake_minimum_required(VERSION 3.22)
project(lc)

set(CMAKE_CXX_STANDARD 20)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(./source/tensor/ DIR_TENSOR)
aux_source_directory(./source/base/ DIR_BASE)
aux_source_directory(./source/op/ DIR_OP)
aux_source_directory(./source/model/ DIR_MODEL)

add_executable(lc main.cpp ${DIR_TENSOR} ${DIR_BASE}  ${DIR_OP} ${DIR_MODEL})
target_link_libraries(lc glog::glog)
target_include_directories(lc PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(lc PUBLIC ./include)
target_include_directories(lc PUBLIC ${Armadillo_INCLUDE_DIR})

target_link_libraries(lc sentencepiece)
target_link_libraries(lc ${ARMADILLO_LIBRARIES})
