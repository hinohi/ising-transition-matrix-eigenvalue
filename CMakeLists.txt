cmake_minimum_required(VERSION 3.26)
project(ising_transition_matrix_eigenvalue)

set(CMAKE_CXX_STANDARD 20)

add_executable(ising_transition_matrix_eigenvalue main.cpp)

add_subdirectory(third-party/eigen)
add_subdirectory(third-party/spectra)

target_link_libraries(ising_transition_matrix_eigenvalue PUBLIC Eigen3::Eigen)
target_link_libraries(ising_transition_matrix_eigenvalue PUBLIC Spectra)
