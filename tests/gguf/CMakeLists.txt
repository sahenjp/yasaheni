# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
#
# Standalone build for the GGUF reader and the GGUF -> engine adapter tests.
# No CUDA, no third-party deps: just gguf.cpp + gguf_model.cpp and the STL.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/gguf_test
#   ./build/gguf_model_test
cmake_minimum_required(VERSION 3.16)
project(yskn_gguf_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(YSKN_GGUF_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf.cpp
)
set(YSKN_GGUF_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

add_executable(gguf_test
  gguf_test.cpp
  ${YSKN_GGUF_SRC}
)

target_include_directories(gguf_test PRIVATE ${YSKN_GGUF_INCLUDE})

if(MSVC)
  target_compile_options(gguf_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(gguf_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- adapter test -----------------------------------------------------------
# gguf_model_test links the reader plus the adapter (gguf_model.cpp for the
# hyperparameters / validation / describe, gguf_map.cpp for the name mapper).
add_executable(gguf_model_test
  gguf_model_test.cpp
  ${YSKN_GGUF_SRC}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_map.cpp
)

target_include_directories(gguf_model_test PRIVATE ${YSKN_GGUF_INCLUDE})

if(MSVC)
  target_compile_options(gguf_model_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(gguf_model_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- fixture generation -----------------------------------------------------
# make_fixture.py writes <out>.gguf plus a <out>.json sidecar of expected
# values. It runs at build time so the binary always tests a fresh fixture.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR "python3 is required to generate the GGUF fixtures")
endif()

set(GGUF_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixture)
set(GGUF_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf)
set(GGUF_FIXTURE_JSON ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json)
set(GGUF_FIXTURE_V2 ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture_v2.gguf)
set(GGUF_VOCAB_ONLY ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_vocab_only.gguf)

# Three fixtures: the main v3 file, a v2 file with identical content (the reader
# must accept both header versions), and a metadata-only file shaped like
# llama.cpp's ggml-vocab-*.gguf, which carries no tensors and no data section.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf
    ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json
    ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture_v2.gguf
    ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture_v2.gguf.json
    ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_vocab_only.gguf
    ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_vocab_only.gguf.json
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fixture
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf --version 3
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture_v2.gguf --version 2
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_vocab_only.gguf --no-tensors
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py
  COMMENT "Generating GGUF fixtures"
  VERBATIM
)

add_custom_target(gguf_fixture ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf
          ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture_v2.gguf
          ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_vocab_only.gguf)
add_dependencies(gguf_test gguf_fixture)

# The adapter fixtures: one valid Qwen3-shaped model, one with a wrong shape,
# one carrying a tensor the mapper does not understand.
set(GGUF_MODEL_FIXTURES
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model.gguf
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_broken.gguf
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_unknown.gguf
)

add_custom_command(
  OUTPUT ${GGUF_MODEL_FIXTURES}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fixture
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture_model.py --variant good --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model.gguf
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture_model.py --variant broken --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_broken.gguf
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture_model.py --variant unknown --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_unknown.gguf
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture_model.py ${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py
  COMMENT "Generating GGUF model fixtures"
  VERBATIM
)

add_custom_target(gguf_model_fixture ALL DEPENDS ${GGUF_MODEL_FIXTURES})
add_dependencies(gguf_model_test gguf_model_fixture)

# --- ctest ------------------------------------------------------------------
enable_testing()
add_test(NAME gguf_reader
  COMMAND gguf_test
          --fixture ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf
          --sidecar ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json
          --out-dir ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME gguf_model_adapter
  COMMAND gguf_model_test
          --good ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model.gguf
          --broken ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_broken.gguf
          --unknown ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_unknown.gguf
          --out-dir ${CMAKE_CURRENT_BINARY_DIR}
)
