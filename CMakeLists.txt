cmake_minimum_required(VERSION 3.12.0)
project(nmpde-Navier-Stokes LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(main2 src/main2.cpp src/NavierStokes.cpp)
deal_ii_setup_target(main2)

# For Makefile generators, extend `make clean` so the build tree is left
# essentially empty after cleaning.
if(CMAKE_GENERATOR MATCHES "Makefiles")
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_BINARY_DIR}/CMakeCache.txt"
    "${CMAKE_BINARY_DIR}/CMakeFiles"
    "${CMAKE_BINARY_DIR}/Makefile"
    "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_BINARY_DIR}/main2"
    "${CMAKE_BINARY_DIR}/results"
    "${CMAKE_BINARY_DIR}/coefficients.txt"
    "${CMAKE_BINARY_DIR}/solution.pvd"
  )
endif()
