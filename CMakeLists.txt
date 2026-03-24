cmake_minimum_required(VERSION 3.16)

project(tsdf_voxel_hash_mc LANGUAGES CXX)

add_executable(tsdf_voxel_hash_demo
  src/main.cpp
)

target_include_directories(tsdf_voxel_hash_demo
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(tsdf_voxel_hash_demo
  PRIVATE
    TSDFMC_DEFAULT_DATASET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/datasets"
)

target_compile_features(tsdf_voxel_hash_demo PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(tsdf_voxel_hash_demo PRIVATE
    /W4
    /permissive-
    $<$<NOT:$<CONFIG:Debug>>:/O2>
  )
else()
  target_compile_options(tsdf_voxel_hash_demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    $<$<NOT:$<CONFIG:Debug>>:-O2>
  )
endif()

if(WIN32)
  target_link_libraries(tsdf_voxel_hash_demo PRIVATE windowscodecs)
endif()
