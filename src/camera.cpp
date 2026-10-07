#include "camera.hpp"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <system_error>
#include <thread>

namespace {

constexpr int MAX_EMPTY_FRAMES = 100;
constexpr auto RETRY_DELAY = std::chrono::milliseconds(30);

}  // namespace

std::optional<cv::Mat> Camera::read() {
    cv::Mat frame;
    for (int attempt = 0; attempt <= MAX_EMPTY_FRAMES; ++attempt) {
        cap_ >> frame;
        if (!frame.empty()) return frame;
        std::this_thread::sleep_for(RETRY_DELAY);
    }
    std::cerr << "Error: camera " << index_ << " opened but returned no frames. "
                 "Check camera permission for your terminal app." << std::endl;
    return std::nullopt;
}

std::optional<Camera> openCamera(int index) {
    Camera camera(index);
    if (!camera.isOpened()) {
        std::cerr << "Error: Could not open camera " << index << "." << std::endl;
        return std::nullopt;
    }
    return camera;
}

std::optional<int> cameraIndexFromArgs(int argc, char** argv) {
    if (argc < 2) return 0;

    const char* arg = argv[1];
    const char* end = arg + std::strlen(arg);
    int index = 0;
    auto [ptr, error] = std::from_chars(arg, end, index);
    if (error != std::errc() || ptr != end || index < 0) {
        std::cerr << "Error: camera index must be a non-negative integer, got '" << arg << "'." << std::endl;
        return std::nullopt;
    }
    return index;
}
