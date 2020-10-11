cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(oneDnnBenchmark LANGUAGES CXX)

set(NeoOnnx_BUILD OFF CACHE BOOL "Build NeoOnnx")
set(NeoMathEngine_ENABLE_VULKAN OFF CACHE BOOL "Build Vulkan backend for MathEngine")
set(DNNL_BUILD_TESTS OFF CACHE BOOL "Build test for oneDNN")
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Build examples for oneDNN")

add_subdirectory(third_party/NeoML/NeoML)
add_subdirectory(third_party/oneDNN)

add_subdirectory(oneDnnBenchmark)
