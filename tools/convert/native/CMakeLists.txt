# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
#
# The code search of the converter as a small shared library (yskn_trellis).
#   cmake -S tools/convert/native -B build_trellis -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_trellis
# Output: build_trellis/bin/yskn_trellis.dll (Windows) or build_trellis/bin/libyskn_trellis.so (Linux).
cmake_minimum_required(VERSION 3.24)
project(yskn_trellis LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# The released models were converted on an RTX 5090 (compute capability 12.0).  Another architecture:
#   -DYSKN_TRELLIS_CUDA_ARCHS=<number>   (the kernel needs 75,152 bytes of shared memory per block; not tested elsewhere)
set(YSKN_TRELLIS_CUDA_ARCHS "120" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES ${YSKN_TRELLIS_CUDA_ARCHS})
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
find_package(CUDAToolkit REQUIRED)

add_library(yskn_trellis SHARED trellis_search.cu)
target_link_libraries(yskn_trellis PRIVATE CUDA::cudart)
# --fmad=false: every multiply and add is rounded on its own, so the result does not depend on how the compiler fuses them.
target_compile_options(yskn_trellis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
