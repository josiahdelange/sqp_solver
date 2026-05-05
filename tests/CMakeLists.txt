cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(solver_test VERSION 0.1.0 LANGUAGES CXX)

#find_package(GTest REQUIRED)
set(CMAKE_BUILD_TYPE Debug)

# Custom test of constrained optimization via SQP solver
add_executable(test_001 test_001.cpp)
target_link_libraries(test_001 sqp_solver Eigen3::Eigen)
set_target_properties(test_001 PROPERTIES CMAKE_BUILD_TYPE "Debug")