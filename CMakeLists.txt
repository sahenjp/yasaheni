cmake_minimum_required(VERSION 3.24)
project(yasakani_engine LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- target architectures ---------------------------------------------------------------------------------------------------------
# The upstream build pinned sm_120 (RTX 5090) only. We keep that as the default but let the
# caller widen it, and we drop architectures the installed toolkit cannot emit.
#   -DCMAKE_CUDA_ARCHITECTURES="120;86;80"     explicit list
#   -DYSKN_CUDA_ARCH=native                    ask the driver for this machine's GPU
#   -DYSKN_CUDA_ARCH=portable                  "120;86;80;75"
set(YSKN_CUDA_ARCH "" CACHE STRING "Comma/semicolon separated SM list, 'native' or 'portable'")
if(YSKN_CUDA_ARCH)
  if(YSKN_CUDA_ARCH STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
  elseif(YSKN_CUDA_ARCH STREQUAL "portable")
    set(CMAKE_CUDA_ARCHITECTURES "120;90;86;80;75")
  else()
    string(REPLACE "," ";" _yskn_arch_list "${YSKN_CUDA_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${_yskn_arch_list}")
  endif()
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)
message(STATUS "yskn: CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")

add_library(yskn SHARED
  src/attention_gqa.cu
  src/attention_kv4q.cu
  src/c_api.cpp
  src/devmem.cpp
  src/draft_kernels.cu
  src/engine.cpp
  src/engine35.cpp
  src/gemm_compressed.cu
  src/gemv_sigma_int.cu
  src/kvx_probe.cu
  src/lnord.cu
  src/loader.cpp
  src/misc_kernels.cu
  src/pfa.cu
  src/pm_test.cu
  src/post_kernels.cu
  src/q35_atx.cu
  src/q35_atxpk.cu
  src/q35_bdraft.cu
  src/q35_chk.cu
  src/q35_fa.cu
  src/q35_flow.cu
  src/q35_g6.cu
  src/q35_kernels.cu
  src/q35_kz.cu
  src/q35_prefill.cu
  src/q35_sample.cu
  src/repack.cu
  src/specx.cpp
  src/test_hooks.cpp
)
target_include_directories(yskn PRIVATE src include third_party)
target_compile_definitions(yskn PRIVATE YSKN_BUILD YSKN_KVX_MODE=3 NOMINMAX _CRT_SECURE_NO_WARNINGS)

# --fmad=false is not cosmetic: the engine's numerics were tuned and verified with fused
# multiply-add contraction disabled, so we keep it on every platform.
set(YSKN_CUDA_FLAGS -O3 --fmad=false -Xptxas -O3)
set(YSKN_CUDA_FLAGS ${YSKN_CUDA_FLAGS} -Xptxas --allow-expensive-optimizations=true)

if(MSVC)
  target_compile_options(yskn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj;/O2;/d1trimfile:${CMAKE_SOURCE_DIR}/>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8;-Xcompiler=/bigobj;-Xcompiler=/d1trimfile:${CMAKE_SOURCE_DIR}/;${YSKN_CUDA_FLAGS}>)
else()
  target_compile_options(yskn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-fno-math-errno;-fvisibility=hidden;-fvisibility-inlines-hidden>
    $<$<COMPILE_LANGUAGE:CUDA>:${YSKN_CUDA_FLAGS};-Xcompiler=-fPIC,-fvisibility=hidden>)
  # Keep the C ABI exported even though the rest of the shared object is hidden.
  set_target_properties(yskn PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
  target_link_options(yskn PRIVATE -Wl,--no-undefined)
endif()

# Link the CUDA runtime statically so the .so needs only the driver, not the toolkit.
if(NOT MSVC)
  set_target_properties(yskn PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(yskn PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS} pthread)
else()
  target_link_libraries(yskn PRIVATE CUDA::cudart)
endif()
