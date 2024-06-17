project(llama_infer CXX CUDA)
add_executable(llama_infer main.cpp)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer llama)