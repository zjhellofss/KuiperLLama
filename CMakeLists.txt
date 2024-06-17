cmake_minimum_required(VERSION 3.16)
project(llama_infer CXX CUDA)
include(cmake/cuda.cmake)

find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(kuiper/source/tensor/ DIR_TENSOR)
aux_source_directory(kuiper/source/base/ DIR_BASE)
aux_source_directory(kuiper/source/op/ DIR_OP)
aux_source_directory(kuiper/source/model/ DIR_MODEL)
aux_source_directory(kuiper/source/op/kernels DIR_KERNEL)
aux_source_directory(kuiper/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_MODEL} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llama sentencepiece glog::glog)
target_link_libraries(llama ${ARMADILLO_LIBRARIES})
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})

add_executable(llama_infer main.cu)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer llama)
set_target_properties(llama_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)