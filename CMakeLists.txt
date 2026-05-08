cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0 LANGUAGES CXX)

enable_testing()
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Eigen library
#set(Eigen3_DIR third-party/eigen-5.0.0)
#find_package(Eigen3 REQUIRED NO_MODULE)

# Main SQP library
add_library(sqp_solver STATIC source/qp.cpp source/sqp.cpp)
target_include_directories(sqp_solver PUBLIC
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/third-party/eigen-5.0.0
)
target_compile_features(sqp_solver PUBLIC cxx_std_14)

# Test program #1: Rosenbrock function constrained to a disk
add_executable(test_001 tests/test_001.cpp)
target_link_libraries(test_001 sqp_solver)
set_target_properties(test_001 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_001 COMMAND test_001)

# Test program #2: Mishra's Bird function - constrained
add_executable(test_002 tests/test_002.cpp)
target_link_libraries(test_002 sqp_solver)
set_target_properties(test_002 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_002 COMMAND test_002)

# Test program #3: Townsend function (modified)
add_executable(test_003 tests/test_003.cpp)
target_link_libraries(test_003 sqp_solver)
set_target_properties(test_003 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_003 COMMAND test_003)

# Test program #4: Matyas function
add_executable(test_004 tests/test_004.cpp)
target_link_libraries(test_004 sqp_solver)
set_target_properties(test_004 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_004 COMMAND test_004)

# Test program #5: Booth function
add_executable(test_005 tests/test_005.cpp)
target_link_libraries(test_005 sqp_solver)
set_target_properties(test_005 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_005 COMMAND test_005)

# Test program #6: Goldstein-Price function
add_executable(test_006 tests/test_006.cpp)
target_link_libraries(test_006 sqp_solver)
set_target_properties(test_006 PROPERTIES CMAKE_BUILD_TYPE "Debug")
add_test(NAME test_006 COMMAND test_006)
