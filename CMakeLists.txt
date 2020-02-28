cmake_minimum_required(VERSION 3.13)
project(texttruth)
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -I /usr/local/include   -L /usr/local/lib -lboost_serialization -lboost_filesystem -march=sandybridge -march=haswell ")

#add_executable(${PROJECT_NAME}   main.cpp texttruth.cpp util.cpp)


# make install
set(CMAKE_BUILD_TYPE  Release)
include(GNUInstallDirs)
add_library(ttruth SHARED  texttruth.cpp util.cpp)
set(MY_HEADER texttruth.h util.h)
target_include_directories(ttruth PUBLIC ${PROJECT_SOURCE_DIR})
set_target_properties(ttruth PROPERTIES PUBLIC_HEADER "${MY_HEADER}")
install(TARGETS ttruth
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include)