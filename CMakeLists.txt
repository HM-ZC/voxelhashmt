cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

project(tsdf_voxel_hash_mc LANGUAGES CXX)

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
  string(APPEND CMAKE_CUDA_FLAGS_INIT " --allow-unsupported-compiler")
endif()

include(CheckLanguage)

if(WIN32 AND CMAKE_GENERATOR MATCHES "^Visual Studio ")
  set(_tsdfmc_use_legacy_findcuda_default ON)
else()
  set(_tsdfmc_use_legacy_findcuda_default OFF)
endif()

option(TSDFMC_USE_LEGACY_FINDCUDA
       "Use legacy FindCUDA/cuda_add_executable integration. This is useful for Visual Studio solution generation with standalone CUDA toolkits."
       ${_tsdfmc_use_legacy_findcuda_default})

set(TSDFMC_HAVE_CUDA OFF)
set(TSDFMC_USING_LEGACY_FINDCUDA OFF)

set(TSDFMC_CXX_SOURCES
  src/main.cpp
  src/reconstruction_backend.cpp
)

set(TSDFMC_CUDA_SOURCES
  src/cuda_block_allocator.cu
  src/cuda_mesh_extractor.cu
  src/cuda_tsdf_integrator.cu
)

if(TSDFMC_USE_LEGACY_FINDCUDA)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

  if(CMAKE_CUDA_COMPILER AND NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    get_filename_component(_tsdfmc_cuda_bin_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${_tsdfmc_cuda_bin_dir}/.." ABSOLUTE)
  endif()

  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    set(TSDFMC_HAVE_CUDA ON)
    set(TSDFMC_USING_LEGACY_FINDCUDA ON)
    list(APPEND CUDA_NVCC_FLAGS --std=c++17)
    if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
      list(APPEND CUDA_NVCC_FLAGS --allow-unsupported-compiler)
    endif()
    if(MSVC)
      string(APPEND CUDA_NVCC_FLAGS_DEBUG " -Xcompiler=/MDd")
      string(APPEND CUDA_NVCC_FLAGS_RELEASE " -Xcompiler=/MD")
      string(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO " -Xcompiler=/MD")
      string(APPEND CUDA_NVCC_FLAGS_MINSIZEREL " -Xcompiler=/MD")
    endif()
  endif()
else()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(TSDFMC_HAVE_CUDA ON)
  endif()
endif()

if(TSDFMC_USING_LEGACY_FINDCUDA)
  cuda_add_executable(tsdf_voxel_hash_demo
    ${TSDFMC_CXX_SOURCES}
    ${TSDFMC_CUDA_SOURCES}
  )
elseif(TSDFMC_HAVE_CUDA)
  add_executable(tsdf_voxel_hash_demo
    ${TSDFMC_CXX_SOURCES}
    ${TSDFMC_CUDA_SOURCES}
  )
else()
  add_executable(tsdf_voxel_hash_demo
    ${TSDFMC_CXX_SOURCES}
  )
endif()

target_include_directories(tsdf_voxel_hash_demo
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(TSDFMC_USING_LEGACY_FINDCUDA)
  cuda_include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
endif()

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

if(TSDFMC_HAVE_CUDA AND NOT TSDFMC_USING_LEGACY_FINDCUDA)
  set_target_properties(tsdf_voxel_hash_demo PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

if(TSDFMC_HAVE_CUDA)
  if(TSDFMC_USING_LEGACY_FINDCUDA)
    message(STATUS "CUDA backend enabled for tsdf_voxel_hash_demo (legacy FindCUDA)")
  else()
    message(STATUS "CUDA backend enabled for tsdf_voxel_hash_demo")
  endif()
else()
  message(STATUS "CUDA toolkit not found; building CPU-only backend")
endif()

if(WIN32)
  if(TSDFMC_USING_LEGACY_FINDCUDA)
    target_link_libraries(tsdf_voxel_hash_demo windowscodecs)
  else()
    target_link_libraries(tsdf_voxel_hash_demo PRIVATE windowscodecs)
  endif()
endif()
