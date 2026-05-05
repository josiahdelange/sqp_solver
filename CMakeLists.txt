cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0 LANGUAGES CXX)

option(SQP_SOLVER_BUILD_TESTS "Enable testing" ON)
option(SQP_SOLVER_INSTALL "Enable generation of install target" ON)

# Eigen library
find_package(Eigen3 REQUIRED NO_MODULE)

# Source files
add_subdirectory(source)

# Test files
if(SQP_SOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
