cmake_minimum_required(VERSION 3.16)
project(llama_infer)

set(CMAKE_CXX_STANDARD 20)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(./source/tensor/ DIR_TENSOR)
aux_source_directory(./source/base/ DIR_BASE)
aux_source_directory(./source/op/ DIR_OP)
aux_source_directory(./source/model/ DIR_MODEL)
aux_source_directory(./source/op/kernels DIR_KERNEL)

add_executable(llama_infer main.cpp ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_MODEL} ${DIR_KERNEL})
target_link_libraries(llama_infer glog::glog)
target_include_directories(llama_infer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama_infer PUBLIC ./include)
target_include_directories(llama_infer PUBLIC ${Armadillo_INCLUDE_DIR})

target_link_libraries(llama_infer sentencepiece)
target_link_libraries(llama_infer ${ARMADILLO_LIBRARIES})
