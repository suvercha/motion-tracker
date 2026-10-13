#pragma once

#include "motion/clock.hpp"
#include "motion/object_finder.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>  // defines HAVE_OPENCV_DNN, which the video headers check
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include <filesystem>
#include <optional>

// Where the TrackerNano model files are looked for by default. CMake sets this to
// the project's models/ folder.
#ifndef MOTION_MODELS_DIR
#define MOTION_MODELS_DIR "models"
#endif

namespace motion {

constexpr const char* DEFAULT_MODELS_DIR = MOTION_MODELS_DIR;

// Where the object is in a frame.
struct Track {
    cv::Rect box;
    cv::Point2f center;
    bool detected;  // false: the detector missed this frame and the position comes from the tracker
};

// Follows a learned object through a video.
//
// The detector (ObjectFinder) runs on every frame. It never drifts, but it
// misses frames when the object is blurred by motion. On a miss, a TrackerNano
// (a small neural-network tracker, about 4 ms a frame) starts from the last
// detected position and follows the object until the detector finds it again.
// A tracker drifts and cannot tell when it has lost the object, so it is only
// trusted for a short time (see MAX_TRACK_ONLY_SECONDS in object_tracker.cpp).
// See docs/tracking-design.md.
class ObjectTracker {
public:
    // Loads the TrackerNano models (nanotrack_backbone_sim.onnx and
    // nanotrack_head_sim.onnx) from `modelsDir`. Prints an error and returns
    // nullopt if they are missing or can't be loaded; scripts/download_models.sh
    // fetches them.
    [[nodiscard]] static std::optional<ObjectTracker> create(ObjectFinder finder,
                                                             const std::filesystem::path& modelsDir);

    // Processes the next frame, taken at `now`. Returns nullopt if the object is
    // lost. `frame` must stay unmodified until the call returns.
    [[nodiscard]] std::optional<Track> update(const cv::Mat& frame, Clock::time_point now);

private:
    ObjectTracker(ObjectFinder finder, cv::dnn::Net backbone, cv::dnn::Net neckhead)
        : finder_(std::move(finder)), backbone_(std::move(backbone)), neckhead_(std::move(neckhead)) {}

    void reset();
    [[nodiscard]] std::optional<Track> followWithTracker(const cv::Mat& frame);

    ObjectFinder finder_;

    // Loaded once, so starting a tracker mid-video doesn't read from disk.
    cv::dnn::Net backbone_;
    cv::dnn::Net neckhead_;

    // The last detection, and the frame it was found in.
    std::optional<Detection> lastDetection_;
    Clock::time_point lastDetectionTime_;
    cv::Mat lastDetectionFrame_;

    // Running only while the detector is missing the object.
    cv::Ptr<cv::Tracker> tracker_;
    cv::Point2f centerOffset_;  // object center minus box center, kept constant while tracking
};

}  // namespace motion
