cmake_minimum_required(VERSION 3.16)
project(motion_tracking CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only the OpenCV modules the code uses; linking all of them makes every program
# load about 60 libraries at startup.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio features geometry)

# Code shared by the programs and the tests: camera access, object segmentation,
# and learning/finding an object.
add_library(motion_common STATIC
    src/camera.cpp
    src/segmentation.cpp
    src/object_finder.cpp)
target_include_directories(motion_common PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(motion_common PUBLIC ${OpenCV_LIBS})

add_executable(capture_object src/capture_object.cpp)
target_link_libraries(capture_object PRIVATE motion_common)

add_executable(find_object src/find_object.cpp)
target_link_libraries(find_object PRIVATE motion_common)

# Offline tests (no camera needed). Run with: ctest --test-dir build --output-on-failure
enable_testing()
add_executable(find_object_test tests/find_object_test.cpp)
target_link_libraries(find_object_test PRIVATE motion_common)
add_test(NAME find_object_orb
         COMMAND find_object_test orb ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)
add_test(NAME find_object_sift
         COMMAND find_object_test sift ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)
