cmake_minimum_required(VERSION 3.16)

project(tsdf_voxel_hash_mc LANGUAGES CXX)

add_executable(tsdf_voxel_hash_demo
  src/main.cpp
)

target_include_directories(tsdf_voxel_hash_demo
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tsdf_voxel_hash_demo PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(tsdf_voxel_hash_demo PRIVATE /O2 /W4 /permissive-)
else()
  target_compile_options(tsdf_voxel_hash_demo PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
