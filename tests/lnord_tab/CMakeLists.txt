# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
#
# Standalone build for the host-only LNORD table tests (src/lnord_tab.h).
# No CUDA, no GPU, no third-party deps.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(yskn_lnord_tab_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(lnord_tab_test
  lnord_tab_test.cpp
)

target_include_directories(lnord_tab_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

if(MSVC)
  target_compile_options(lnord_tab_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(lnord_tab_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME lnord_tab COMMAND lnord_tab_test)
