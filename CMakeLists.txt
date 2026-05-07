cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0 LANGUAGES CXX)
enable_testing()
message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

# Eigen library
set(Eigen3_DIR third-party/eigen-5.0.0)
find_package(Eigen3 REQUIRED NO_MODULE)

# Main SQP library
add_library(sqp_solver STATIC source/qp.cpp source/sqp.cpp)
target_link_libraries(sqp_solver Eigen3::Eigen)
target_include_directories(sqp_solver PUBLIC
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/third-party/osqp-v1.0.0/include/public
    ${CMAKE_CURRENT_BINARY_DIR}/third-party/osqp-v1.0.0/include/public
)
target_compile_features(sqp_solver PUBLIC cxx_std_11)

# Test program #1: Rosenbrock function constrained to a disk
add_executable(test_001 tests/test_001.cpp)
target_link_libraries(test_001 sqp_solver Eigen3::Eigen)
set_target_properties(test_001 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_001 COMMAND test_001)

# Test program #2: Mishra's Bird function - constrained
add_executable(test_002 tests/test_002.cpp)
target_link_libraries(test_002 sqp_solver Eigen3::Eigen)
set_target_properties(test_002 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_002 COMMAND test_002)

# Test program #3: Townsend function (modified)
add_executable(test_003 tests/test_003.cpp)
target_link_libraries(test_003 sqp_solver Eigen3::Eigen)
set_target_properties(test_003 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_003 COMMAND test_003)

# Test program #4: Matyas function
add_executable(test_004 tests/test_004.cpp)
target_link_libraries(test_004 sqp_solver Eigen3::Eigen)
set_target_properties(test_004 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_004 COMMAND test_003)

# Test program #5: Booth function
add_executable(test_005 tests/test_005.cpp)
target_link_libraries(test_005 sqp_solver Eigen3::Eigen)
set_target_properties(test_005 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_005 COMMAND test_005)

# Test program #6: Goldstein-Price function
add_executable(test_006 tests/test_006.cpp)
target_link_libraries(test_006 sqp_solver Eigen3::Eigen)
set_target_properties(test_006 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_006 COMMAND test_006)
