cmake_minimum_required(VERSION 2.8)

project(Farneback)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(Farneback Farneback.cpp)

target_link_libraries(Farneback ${OpenCV_LIBS})