cmake_minimum_required ( VERSION 3.18 )

project(pierogi LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/GraphBLAS/cmake_modules)
add_subdirectory(GraphBLAS)

add_subdirectory(../combblas/cnpy ./cnpy)

add_executable(blas main.cpp)
add_executable(bench run_bench.cpp)

target_link_libraries(blas graphblas cnpy)
target_link_libraries(bench graphblas cnpy)

