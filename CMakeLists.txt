cmake_minimum_required(VERSION 3.16)
project(motion_tracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(motion_tracking src/main.cpp)
target_link_libraries(motion_tracking PRIVATE ${OpenCV_LIBS})
target_include_directories(motion_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})

# Object learning/detection shared by find_object and its tests.
add_library(object_finder STATIC src/object_finder.cpp)
target_include_directories(object_finder PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_finder PUBLIC ${OpenCV_LIBS})

add_executable(find_object src/find_object.cpp)
target_link_libraries(find_object PRIVATE object_finder)

# Offline tests (no camera needed). Run with: ctest --test-dir build --output-on-failure
enable_testing()
add_executable(find_object_test tests/find_object_test.cpp)
target_link_libraries(find_object_test PRIVATE object_finder)
add_test(NAME find_object_orb
         COMMAND find_object_test orb ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)
add_test(NAME find_object_sift
         COMMAND find_object_test sift ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)
