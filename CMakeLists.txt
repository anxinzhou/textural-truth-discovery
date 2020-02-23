cmake_minimum_required(VERSION 3.13)
project(texttruth)
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -I /usr/local/include   -L /usr/local/lib")

add_executable(${PROJECT_NAME}   main.cpp texttruth.cpp)

