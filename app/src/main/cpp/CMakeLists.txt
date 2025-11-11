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

add_library(llama_jni SHARED
    native/jni_bridge.cpp
)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_SOURCE_DIR}
)

if (USE_LLAMA)
    target_include_directories(llama_jni PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../../third_party/llama.cpp
        ${CMAKE_SOURCE_DIR}/../../../../third_party/llama.cpp/vendor
        ${CMAKE_SOURCE_DIR}/../../../../third_party/Vulkan-Headers/include
    )
    target_link_libraries(llama_jni
        PRIVATE
        llama
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


