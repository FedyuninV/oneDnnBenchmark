cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(oneDnnBenchmark LANGUAGES CXX)

if(OPENCLROOT)
	message(NOTICE "Using OCL runtime for oneDNN")
	set(DNNL_GPU_RUNTIME "OCL" CACHE STRING "Use GPU runtime in oneDNN")
else()
	message(NOTICE "No OCL SDK provided for oneDNN")
	set(NeoMathEngine_ENABLE_VULKAN OFF CACHE BOOL "Don't use GPU in NeoML")
endif()

set(NeoOnnx_BUILD OFF CACHE BOOL "Build NeoOnnx")
add_subdirectory(third_party/NeoML/NeoML)

set(DNNL_BUILD_TESTS OFF CACHE BOOL "Build test for oneDNN")
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Build examples for oneDNN")
add_subdirectory(third_party/oneDNN)

add_subdirectory(oneDnnBenchmark)
