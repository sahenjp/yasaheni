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

# --- YSKN conformance ------------------------------------------------------
# Checks this repository's reader against upstream's own format fixture:
# tools/testdata/tiny.yskn was written by upstream's tools/yskn_fixture.py, and
# tiny_expected.json by the same script. format.hpp and signs.hpp are the two
# pieces under test, so this needs no GGUF sources.
add_executable(yskn_conformance_test
  yskn_conformance_test.cpp
)

target_include_directories(yskn_conformance_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${YSKN_GGUF_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
)

if(MSVC)
  target_compile_options(yskn_conformance_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(yskn_conformance_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME yskn_conformance
         COMMAND yskn_conformance_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/testdata/tiny.yskn)

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

# --- engine wiring test -----------------------------------------------------
# gguf_engine_test links the reader, the adapter and the bridge. It also exercises
# the C ABI (yskn_gguf_probe), so it needs the repo's include/ directory for
# yskn_abi.h and third_party/ for nlohmann/json.hpp.
add_executable(gguf_engine_test
  gguf_engine_test.cpp
  ${YSKN_GGUF_SRC}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_engine.cpp
)

target_include_directories(gguf_engine_test PRIVATE
  ${YSKN_GGUF_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
)

if(MSVC)
  target_compile_options(gguf_engine_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(gguf_engine_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- quantiser / encoder test -----------------------------------------------
# The GGUF -> engine tensor bridge: rotation, codebook training, Viterbi and the
# layout:"warp" interleave. Host only -- no CUDA, no GGUF fixture, no GPU.
add_executable(gguf_quant_test
  gguf_quant_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_quant.cpp
)

target_include_directories(gguf_quant_test PRIVATE ${YSKN_GGUF_INCLUDE})

if(MSVC)
  target_compile_options(gguf_quant_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(gguf_quant_test PRIVATE -Wall -Wextra -Wpedantic)
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
add_dependencies(gguf_engine_test gguf_model_fixture)

# --- ctest ------------------------------------------------------------------
enable_testing()
add_test(NAME gguf_reader
  COMMAND gguf_test
          --fixture ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf
          --sidecar ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_fixture.gguf.json
          --out-dir ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME gguf_quant
  COMMAND gguf_quant_test
)

add_test(NAME gguf_model_adapter
  COMMAND gguf_model_test
          --good ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model.gguf
          --broken ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_broken.gguf
          --unknown ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_unknown.gguf
          --out-dir ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME gguf_engine_wiring
  COMMAND gguf_engine_test
          --good ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model.gguf
          --broken ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_broken.gguf
          --unknown ${CMAKE_CURRENT_BINARY_DIR}/fixture/gguf_model_unknown.gguf
          --out-dir ${CMAKE_CURRENT_BINARY_DIR}
)

# --- GGUF-backed container test ---------------------------------------------
# GgufContainer (src/gguf/gguf_container.cpp) materialises a GGUF file into the
# in-memory image src/loader.cpp's Model::load reads. It links the reader, the
# adapter (read_config), the name mapper (validate/coverage live in the same
# translation unit) and the encoder. Host only -- no CUDA, no GPU.
#
# src/format.hpp pulls in nlohmann/json.hpp, so third_party/ is on the include
# path even though this test never parses a YSKN container.
add_executable(gguf_container_test
  gguf_container_test.cpp
  ${YSKN_GGUF_SRC}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_quant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_container.cpp
)

target_include_directories(gguf_container_test PRIVATE
  ${YSKN_GGUF_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
)

if(MSVC)
  target_compile_options(gguf_container_test PRIVATE /W4 /utf-8)
else()
  target_compile_options(gguf_container_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# The container fixtures. tools/make_tiny_gguf.py writes a Qwen3-shaped GGUF
# plus a .json sidecar of expected values. Five variants:
#
#   tiny        2 layers, the loadable model
#   tiny_bad    hidden 96, not a multiple of 256: must be rejected
#   tiny_miss   no blk.1.ffn_down.weight: must be rejected
#   tiny_tie    tied embeddings (no output.weight)
#   tiny_nobias no attn_q/k/v.bias (the container synthesises zeros)
set(GGUF_CONTAINER_FIXTURES
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny.gguf
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_bad.gguf
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_missing.gguf
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_tie.gguf
  ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_nobias.gguf
)

add_custom_command(
  OUTPUT ${GGUF_CONTAINER_FIXTURES}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fixture
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny.gguf
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_bad.gguf --variant badshape
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_missing.gguf --variant missing
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_tie.gguf --variant tie --layers 1
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
          --out ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_nobias.gguf --variant nobias --layers 1
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
  COMMENT "Generating tiny GGUF container fixtures"
  VERBATIM
)

add_custom_target(gguf_container_fixture ALL DEPENDS ${GGUF_CONTAINER_FIXTURES})
add_dependencies(gguf_container_test gguf_container_fixture)

add_test(NAME gguf_container
  COMMAND gguf_container_test
          --good ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny.gguf
          --bad ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_bad.gguf
          --missing ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_missing.gguf
          --tie ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_tie.gguf
          --nobias ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny_nobias.gguf
)

# --- GGUF -> Model::load dispatch test --------------------------------------
# gguf_load_test checks the last wiring slice: Model::open_backing() picks the
# Container or the GgufContainer from the magic, cfg follows the live one, and
# T()/V()/at<>() (src/model.hpp) resolve the same names either way.
#
# No CUDA *link* is needed: open_backing() and the three accessors are all
# inline in src/model.hpp, and the test never calls Model::load() (whose upload
# half is the only thing that touches cudaMalloc). The CUDA toolkit has to be on
# the include path because model.hpp -> kernels.hpp/post_kernels.hpp include
# <cuda_runtime.h>, but the binary links nothing from it.
#
# Fixtures: tiny.gguf (head_dim 64, from tools/make_tiny_gguf.py) for the
# dispatch checks, plus a head_dim-128 and a head_dim-64 pair for the gate.
find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
  message(WARNING "CUDA toolkit not found: skipping gguf_load_test")
else()
  add_executable(gguf_load_test
    gguf_load_test.cpp
    ${YSKN_GGUF_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_map.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_container.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gguf/gguf_engine.cpp
  )

  target_include_directories(gguf_load_test PRIVATE
    ${YSKN_GGUF_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  if(MSVC)
    target_compile_options(gguf_load_test PRIVATE /W4 /utf-8)
  else()
    target_compile_options(gguf_load_test PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  # The gate fixtures: head_dim 128 (accepted) and head_dim 64 (rejected).
  set(GGUF_LOAD_HD128 ${CMAKE_CURRENT_BINARY_DIR}/fixture/load_hd128.gguf)
  set(GGUF_LOAD_HD64 ${CMAKE_CURRENT_BINARY_DIR}/fixture/load_hd64.gguf)

  add_custom_command(
    OUTPUT ${GGUF_LOAD_HD128} ${GGUF_LOAD_HD128}.json
           ${GGUF_LOAD_HD64} ${GGUF_LOAD_HD64}.json
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fixture
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
            --out ${GGUF_LOAD_HD128} --layers 1 --hidden 512 --ffn 512
            --heads 4 --kv-heads 1 --head-dim 128 --vocab 16
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
            --out ${GGUF_LOAD_HD64} --layers 1 --hidden 256 --ffn 512
            --heads 4 --kv-heads 2 --head-dim 64 --vocab 16
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/make_tiny_gguf.py
    COMMENT "Generating GGUF load-dispatch fixtures"
    VERBATIM
  )

  add_custom_target(gguf_load_fixture ALL DEPENDS ${GGUF_LOAD_HD128} ${GGUF_LOAD_HD64})
  add_dependencies(gguf_load_test gguf_load_fixture)
  add_dependencies(gguf_load_test gguf_container_fixture)

  add_test(NAME gguf_load_dispatch
    COMMAND gguf_load_test
            --gguf ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny.gguf
            --gguf-sidecar ${CMAKE_CURRENT_BINARY_DIR}/fixture/tiny.gguf.json
            --hd128 ${GGUF_LOAD_HD128}
            --hd128-sidecar ${GGUF_LOAD_HD128}.json
            --hd64 ${GGUF_LOAD_HD64}
            --hd64-sidecar ${GGUF_LOAD_HD64}.json
            --out-dir ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
