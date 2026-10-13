#pragma once

// Pieces shared by the offline tests for building synthetic camera frames.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace test_scene {

inline const cv::Size FRAME_SIZE(1280, 720);
constexpr int JPEG_QUALITY = 70;  // webcam-like compression
constexpr int CLUTTER_BOXES = 25;
constexpr int CLUTTER_TEXTS = 6;

// Background made from the photo's own background (its top-left corner, which
// is assumed to be plain background), plus colored boxes and text as clutter.
inline cv::Mat makeBackground(const cv::Mat& photo, cv::RNG& rng) {
    cv::Mat patch = photo(cv::Rect(0, 0, photo.cols / 6, photo.rows / 5)).clone();
    cv::Mat bg;
    cv::resize(patch, bg, FRAME_SIZE);
    for (int i = 0; i < CLUTTER_BOXES; i++) {
        cv::Rect r(rng.uniform(0, FRAME_SIZE.width - 100), rng.uniform(0, FRAME_SIZE.height - 100),
                   rng.uniform(30, 160), rng.uniform(30, 160));
        cv::rectangle(bg, r, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
                      cv::FILLED);
    }
    for (int i = 0; i < CLUTTER_TEXTS; i++) {
        cv::putText(bg, "HELPS DRY",
                    cv::Point(rng.uniform(0, FRAME_SIZE.width - 300), rng.uniform(40, FRAME_SIZE.height - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
    }
    return bg;
}

// Compresses and decompresses `frame` as a JPEG, like a webcam does.
inline cv::Mat jpegRoundTrip(const cv::Mat& frame) {
    std::vector<uchar> jpg;
    cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});
    return cv::imdecode(jpg, cv::IMREAD_COLOR);
}

}  // namespace test_scene
