cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0 LANGUAGES CXX)

# Eigen library
set(Eigen3_DIR third-party/eigen-5.0.0)
find_package(Eigen3 REQUIRED NO_MODULE)

# Main library
add_library(sqp_solver STATIC source/qp.cpp source/sqp.cpp)
target_link_libraries(sqp_solver Eigen3::Eigen)
target_include_directories(sqp_solver PUBLIC source)
target_compile_features(sqp_solver PUBLIC cxx_std_11)

# Custom test of constrained optimization via SQP solver
enable_testing()
add_executable(test_001 tests/test_001.cpp)
target_link_libraries(test_001 sqp_solver Eigen3::Eigen)
set_target_properties(test_001 PROPERTIES CMAKE_BUILD_TYPE "Debug")
