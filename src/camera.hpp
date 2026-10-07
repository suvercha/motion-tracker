#pragma once

#include <opencv2/videoio.hpp>
#include <optional>

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

// The camera index from the first command-line argument (default 0). Prints an
// error and returns nullopt if the argument is not a non-negative integer.
[[nodiscard]] std::optional<int> cameraIndexFromArgs(int argc, char** argv);
