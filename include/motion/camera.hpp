#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <string_view>

namespace motion {

constexpr int DEFAULT_CAMERA_INDEX = 0;

// A webcam that tolerates the empty frames cameras often return while warming up.
class Camera {
public:
    explicit Camera(int index) : cap_(index), index_(index) {}

    [[nodiscard]] bool isOpened() const { return cap_.isOpened(); }
    [[nodiscard]] int index() const { return index_; }

    // Returns the next frame. Waits through warm-up; if the camera still gives
    // nothing, prints an error and returns nullopt.
    [[nodiscard]] std::optional<cv::Mat> read();

private:
    cv::VideoCapture cap_;
    int index_;
};

// Opens camera `index`, printing an error and returning nullopt on failure.
[[nodiscard]] std::optional<Camera> openCamera(int index);

// Parses a camera index from command-line text. Prints an error and returns
// nullopt if it is not a non-negative integer.
[[nodiscard]] std::optional<int> parseCameraIndex(std::string_view text);

}  // namespace motion
