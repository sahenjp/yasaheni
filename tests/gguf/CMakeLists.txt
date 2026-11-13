# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
#
# Standalone build for the GGUF reader test. No CUDA, no third-party deps:
# just gguf.cpp + gguf_test.cpp and the STL.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/gguf_test
cmake_minimum_required(VERSION 3.16)
project(yskn_gguf_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gguf_test
  gguf_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf.cpp
)

target_include_directories(gguf_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

if(MSVC)
  target_compile_options(gguf_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(gguf_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- fixture generation -----------------------------------------------------
# make_fixture.py writes <out>.gguf plus a <out>.json sidecar of expected
# values. It runs at build time so the binary always tests a fresh fixture.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR "python3 is required to generate the GGUF fixture")
endif()

set(GGUF_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixture)
set(GGUF_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf)
set(GGUF_FIXTURE_JSON ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fixture
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py
  COMMENT "Generating GGUF fixture"
  VERBATIM
)

add_custom_target(gguf_fixture ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf)
add_dependencies(gguf_test gguf_fixture)

# --- ctest ------------------------------------------------------------------
enable_testing()
add_test(NAME gguf_reader
  COMMAND gguf_test
          --fixture ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf
          --sidecar ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json
          --out-dir ${CMAKE_CURRENT_BINARY_DIR}
)
