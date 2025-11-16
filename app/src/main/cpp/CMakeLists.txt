cmake_minimum_required(VERSION 3.22)
project(llama_android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Build option to toggle real llama.cpp vs stub
option(USE_LLAMA "Build with third_party/llama.cpp" OFF)

# Auto-enable if the directory exists
if (EXISTS "${CMAKE_SOURCE_DIR}/../../../../third_party/llama.cpp/CMakeLists.txt")
    set(USE_LLAMA ON)
endif()

if (DEFINED CMAKE_ANDROID_ARCH_ABI)
    message(STATUS "BanyaChat build ABI: ${CMAKE_ANDROID_ARCH_ABI}")
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        add_compile_options(-march=armv8.2-a+dotprod+i8mm -mtune=cortex-a78)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+i8mm -mtune=cortex-a78")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+i8mm -mtune=cortex-a78")
    endif()
endif()

if (USE_LLAMA)
    set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../../../../third_party/llama.cpp")
    if (NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "llama.cpp not found. Expected at: ${LLAMA_CPP_DIR}\nHint: git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp && git submodule update --init --recursive")
    endif()

    add_compile_definitions(GGML_USE_K_QUANTS=1)

    add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama_build)
endif()

# OpenCL Shim library: Uses android_dlopen_ext to bypass Android namespace restrictions
add_library(opencl_shim STATIC
    opencl_shim.cpp
)
target_link_libraries(opencl_shim PRIVATE dl log android)
# Make opencl_shim globally available to all subdirectories
set(OPENCL_SHIM_TARGET opencl_shim CACHE INTERNAL "OpenCL shim library target")

add_library(llama_jni SHARED
    native/jni_bridge.cpp
)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_SOURCE_DIR}
)

if (USE_LLAMA)
    # OpenCL 헤더 경로 추가
    set(OPENCL_HEADERS_DIR "${CMAKE_SOURCE_DIR}/../../../../third_party/OpenCL-Headers")
    if (EXISTS "${OPENCL_HEADERS_DIR}/CL/cl.h")
        target_include_directories(llama_jni PRIVATE
            ${OPENCL_HEADERS_DIR}
        )
        message(STATUS "OpenCL headers found at: ${OPENCL_HEADERS_DIR}")
    endif()
    
    target_include_directories(llama_jni PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../../third_party/llama.cpp
        ${CMAKE_SOURCE_DIR}/../../../../third_party/llama.cpp/vendor
    )
    
    # OpenCL 라이브러리 찾기 및 연결 (NDK가 제공하는 스텁 라이브러리)
    find_library(OPENCL_LIBRARY OpenCL)
    if (OPENCL_LIBRARY)
        message(STATUS "Found OpenCL library: ${OPENCL_LIBRARY}")
        target_link_libraries(llama_jni PRIVATE ${OPENCL_LIBRARY})
    else()
        message(WARNING "OpenCL library not found! OpenCL backend may not work.")
    endif()
    
    # Link llama.cpp libraries and OpenCL shim
    target_link_libraries(llama_jni
        PRIVATE
        llama
        opencl_shim  # OpenCL shim provides runtime-loaded OpenCL functions via android_dlopen_ext
        log
        android
    )
else()
    target_link_libraries(llama_jni
        PRIVATE
        log
        android
    )
    target_compile_definitions(llama_jni PRIVATE LLAMA_STUB_MODE=1)
endif()


