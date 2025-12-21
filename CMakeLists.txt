cmake_minimum_required(VERSION 3.12.0)
project(nmpde-Navier-Stokes LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(main src/main.cpp src/NavierStokes2D.cpp)
deal_ii_setup_target(main)
