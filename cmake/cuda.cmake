if (MSVC)
    # Setting this to true brakes Visual Studio builds.
    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
endif ()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.11)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
            find_package(CUDAToolkit QUIET)
            set(CUDA_TOOLKIT_INCLUDE ${CUDAToolkit_INCLUDE_DIRS})
        else ()
            set(CUDA_FIND_QUIETLY TRUE)
            find_package(CUDA 9.0)
        endif ()

        set(CUDA_FOUND TRUE)
        set(CUDA_VERSION_STRING ${CMAKE_CUDA_COMPILER_VERSION})
    else ()
        message(STATUS "No CUDA compiler found")
    endif ()
else ()
    set(CUDA_FIND_QUIETLY TRUE)
    find_package(CUDA 9.0)
endif ()

if (CUDA_FOUND)
    message(STATUS "Found CUDA Toolkit v${CUDA_VERSION_STRING}")

    set(HAVE_CUDA TRUE)


    set(CMAKE_CUDA_ARCHITECTURES "86")

    set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT /usr/local/cuda)
else ()
    message(STATUS "CUDA was not found.")
endif ()