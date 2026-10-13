cmake_minimum_required(VERSION 3.16)
project(motion_tracking CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only the OpenCV modules the code uses; linking all of them makes every program
# load about 60 libraries at startup.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio video tracking features geometry dnn)

# Compiler warnings, applied to every target that links this.
add_library(motion_warnings INTERFACE)
if(MSVC)
    target_compile_options(motion_warnings INTERFACE /W4)
else()
    target_compile_options(motion_warnings INTERFACE -Wall -Wextra -Wdeprecated-declarations)
endif()

# Code shared by the programs and the tests: camera access, object segmentation,
# learning/finding an object, following it, and drawing its path and speed.
# Public headers are in include/motion/.
add_library(motion_common STATIC
    src/camera.cpp
    src/segmentation.cpp
    src/object_finder.cpp
    src/object_tracker.cpp
    src/overlay.cpp
    src/path_history.cpp)
target_compile_features(motion_common PUBLIC cxx_std_20)
target_include_directories(motion_common PUBLIC include)
target_include_directories(motion_common SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
# Where ObjectTracker looks for the TrackerNano models by default (see models/README.md).
target_compile_definitions(motion_common PUBLIC MOTION_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")
target_link_libraries(motion_common PUBLIC ${OpenCV_LIBS} PRIVATE motion_warnings)

# Programs.
add_executable(capture_object apps/capture_object.cpp)
target_link_libraries(capture_object PRIVATE motion_common motion_warnings)

add_executable(find_object apps/find_object.cpp)
target_link_libraries(find_object PRIVATE motion_common motion_warnings)

# Offline tests (no camera needed). Run with: ctest --test-dir build --output-on-failure
# Turn them off with -DBUILD_TESTING=OFF.
include(CTest)
if(BUILD_TESTING)
    add_executable(find_object_test tests/find_object_test.cpp)
    target_link_libraries(find_object_test PRIVATE motion_common motion_warnings)
    add_test(NAME find_object_orb
             COMMAND find_object_test orb ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)
    add_test(NAME find_object_sift
             COMMAND find_object_test sift ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)

    add_executable(tracking_test tests/tracking_test.cpp)
    target_link_libraries(tracking_test PRIVATE motion_common motion_warnings)
    add_test(NAME tracking_orb
             COMMAND tracking_test orb ${CMAKE_SOURCE_DIR}/object.jpg ${CMAKE_BINARY_DIR}/test_output)

    add_executable(path_history_test tests/path_history_test.cpp)
    target_link_libraries(path_history_test PRIVATE motion_common motion_warnings)
    add_test(NAME path_history COMMAND path_history_test)
endif()
