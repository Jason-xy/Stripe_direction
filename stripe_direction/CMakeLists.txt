cmake_minimum_required(VERSION 2.8)

project(stripe_direction)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(stripe_direction stripe_direction.cpp)

target_link_libraries(stripe_direction ${OpenCV_LIBS})