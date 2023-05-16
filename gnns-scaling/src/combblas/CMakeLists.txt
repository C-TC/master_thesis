cmake_minimum_required ( VERSION 3.18 )

project(pierogi2 LANGUAGES C CXX)

add_subdirectory(CombBLAS)
add_subdirectory(cnpy)

add_executable(va va.cpp)

target_link_libraries(va CombBLAS cnpy)
