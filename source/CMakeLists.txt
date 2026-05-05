# Main library
add_library(${PROJECT_NAME} STATIC qp.cpp sqp.cpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
