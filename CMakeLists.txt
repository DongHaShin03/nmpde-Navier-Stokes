cmake_minimum_required(VERSION 3.12.0)
project(nmpde-Navier-Stokes LANGUAGES CXX C)

include(common/cmake-common.cmake)

set(NAVIER_STOKES_SOURCES
  src/benchmarks/BenchmarkRecorder.cpp
  src/NavierStokes.cpp
  src/NavierStokes2D.cpp
  src/NavierStokes3D.cpp
  src/FlowPastCylinder2D.cpp
  src/FlowPastCylinder3D.cpp
)

add_executable(main src/main.cpp ${NAVIER_STOKES_SOURCES})
target_include_directories(main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
deal_ii_setup_target(main)

# For Makefile generators, extend `make clean` so the build tree is left
# essentially empty after cleaning.
if(CMAKE_GENERATOR MATCHES "Makefiles")
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_BINARY_DIR}/CMakeCache.txt"
    "${CMAKE_BINARY_DIR}/CMakeFiles"
    "${CMAKE_BINARY_DIR}/Makefile"
    "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_BINARY_DIR}/main"
    "${CMAKE_BINARY_DIR}/results"
    "${CMAKE_BINARY_DIR}/results_3d"
    "${CMAKE_BINARY_DIR}/benchmark_results"
    "${CMAKE_BINARY_DIR}/coefficients.txt"
    "${CMAKE_BINARY_DIR}/coefficients_3d.txt"
    "${CMAKE_BINARY_DIR}/solution.pvd"
  )
endif()
