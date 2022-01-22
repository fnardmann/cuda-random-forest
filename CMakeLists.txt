cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(cuda-random-forest)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(main ${SOURCES})
