cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0 LANGUAGES CXX)
enable_testing()

# Eigen library
set(Eigen3_DIR third-party/eigen-5.0.0)
find_package(Eigen3 REQUIRED NO_MODULE)

# Main library
add_library(sqp_solver STATIC source/qp.cpp source/sqp.cpp)
target_link_libraries(sqp_solver Eigen3::Eigen)
target_include_directories(sqp_solver PUBLIC source)
target_compile_features(sqp_solver PUBLIC cxx_std_11)

# Test program #1: Rosenbrock function constrained to a disk
add_executable(test_001 tests/test_001.cpp)
target_link_libraries(test_001 sqp_solver Eigen3::Eigen)
set_target_properties(test_001 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_001 COMMAND test_001 WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})

# Test program #2: Mishra's Bird function - constrained
add_executable(test_002 tests/test_002.cpp)
target_link_libraries(test_002 sqp_solver Eigen3::Eigen)
set_target_properties(test_002 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_002 COMMAND test_002 WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})

# Test program #3: Townsend function (modified)
add_executable(test_003 tests/test_003.cpp)
target_link_libraries(test_003 sqp_solver Eigen3::Eigen)
set_target_properties(test_003 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_003 COMMAND test_003 WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})
