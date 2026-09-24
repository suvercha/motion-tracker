cmake_minimum_required(VERSION 3.16)
project(motion_tracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(motion_tracking src/main.cpp)
target_link_libraries(motion_tracking PRIVATE ${OpenCV_LIBS})
target_include_directories(motion_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
