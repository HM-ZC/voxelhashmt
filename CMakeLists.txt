cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

project(tsdf_voxel_hash_mc LANGUAGES CXX)

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
  string(APPEND CMAKE_CUDA_FLAGS_INIT " --allow-unsupported-compiler")
endif()

include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
endif()

set(TSDFMC_HAVE_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  set(TSDFMC_HAVE_CUDA ON)
endif()

add_executable(tsdf_voxel_hash_demo
  src/main.cpp
  src/reconstruction_backend.cpp
)

if(TSDFMC_HAVE_CUDA)
  target_sources(tsdf_voxel_hash_demo PRIVATE
    src/cuda_block_allocator.cu
    src/cuda_mesh_extractor.cu
    src/cuda_tsdf_integrator.cu
  )
endif()

target_include_directories(tsdf_voxel_hash_demo
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(tsdf_voxel_hash_demo
  PRIVATE
    TSDFMC_DEFAULT_DATASET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/datasets"
    TSDFMC_HAVE_CUDA=$<IF:$<BOOL:${TSDFMC_HAVE_CUDA}>,1,0>
)

set_target_properties(tsdf_voxel_hash_demo PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
if(MSVC)
  target_compile_options(tsdf_voxel_hash_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:/O2>
  )
else()
  target_compile_options(tsdf_voxel_hash_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-O2>
  )
endif()

if(TSDFMC_HAVE_CUDA)
  set_target_properties(tsdf_voxel_hash_demo PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
  )
  message(STATUS "CUDA backend enabled for tsdf_voxel_hash_demo")
else()
  message(STATUS "CUDA toolkit not found; building CPU-only backend")
endif()

if(WIN32)
  target_link_libraries(tsdf_voxel_hash_demo PRIVATE windowscodecs)
endif()
