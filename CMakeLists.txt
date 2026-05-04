cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0 LANGUAGES CXX)

option(SQP_SOLVER_BUILD_TESTS "Enable testing" ON)
option(SQP_SOLVER_INSTALL "Enable generation of install target" ON)

# Eigen library
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME} STATIC src/qp.cpp src/sqp.cpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_include_directories(${PROJECT_NAME} PUBLIC include/)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

if(SQP_SOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
