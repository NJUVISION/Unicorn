cmake_minimum_required(VERSION 2.6)
project(PCQM)


if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /openmp /std:c++17")
endif(MSVC)

if (NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp -std=c++17")
endif(NOT MSVC)

add_executable(PCQM main.cpp)

include_directories(eigen3/)

include_directories(include/)

FILE(COPY resources/L_data.txt DESTINATION "${CMAKE_BINARY_DIR}")
FILE(COPY resources/RegularGrid_0_0_1.txt DESTINATION "${CMAKE_BINARY_DIR}")
FILE(COPY resources/RegularGrid_0_0_2.txt DESTINATION "${CMAKE_BINARY_DIR}")
FILE(COPY resources/RegularGridInit_0_0_1.txt DESTINATION "${CMAKE_BINARY_DIR}")
FILE(COPY resources/RegularGridInit_0_0_2.txt DESTINATION "${CMAKE_BINARY_DIR}")

install(TARGETS PCQM RUNTIME DESTINATION bin)
