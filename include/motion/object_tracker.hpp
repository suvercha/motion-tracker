#pragma once

#include "motion/clock.hpp"
#include "motion/object_finder.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>  // defines HAVE_OPENCV_DNN, which the video headers check
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string_view>

// Where the tracker model files are looked for by default. CMake sets this to the
// project's models/ folder.
#ifndef MOTION_MODELS_DIR
#define MOTION_MODELS_DIR "models"
#endif

namespace motion {

constexpr const char* DEFAULT_MODELS_DIR = MOTION_MODELS_DIR;

// The trackers that can follow the object between detections. See docs/tracking-design.md
// for how they compare.
enum class TrackerType { Nano, Vit, Mil, Kcf, Csrt };

constexpr std::array<TrackerType, 5> ALL_TRACKER_TYPES = {TrackerType::Nano, TrackerType::Vit, TrackerType::Mil,
                                                          TrackerType::Kcf, TrackerType::Csrt};
constexpr TrackerType DEFAULT_TRACKER = TrackerType::Nano;

[[nodiscard]] constexpr std::string_view toString(TrackerType type) {
    switch (type) {
        case TrackerType::Nano: return "nano";
        case TrackerType::Vit: return "vit";
        case TrackerType::Mil: return "mil";
        case TrackerType::Kcf: return "kcf";
        case TrackerType::Csrt: return "csrt";
    }
    return "";
}

// "nano", "vit", "mil", "kcf" or "csrt" (lowercase), or nullopt for anything else.
[[nodiscard]] std::optional<TrackerType> parseTrackerType(std::string_view text);

// Creates a fresh tracker each time it is called. Loading a tracker's model files is done
// once, when the factory is made.
using TrackerFactory = std::function<cv::Ptr<cv::Tracker>()>;

// Makes a factory for `type`. Nano and Vit need model files in `modelsDir` (the other
// trackers need none). Prints an error and returns nullopt if they are missing or can't be
// loaded; scripts/download_models.sh fetches them.
[[nodiscard]] std::optional<TrackerFactory> createTrackerFactory(TrackerType type,
                                                                 const std::filesystem::path& modelsDir);

// Where the object is in a frame.
struct Track {
    cv::Rect box;
    cv::Point2f center;
    bool detected;  // false: the detector missed this frame and the position comes from the tracker
    // The object's widest dimension in the frame, in pixels. While only the tracker is
    // following it, this is the size from the last detection, because a tracker's box
    // size drifts (the object is assumed to stay at about the same distance).
    double apparentSizePx;

    // The real size of one pixel at the object, in centimetres, if the object is
    // `objectWidthCm` across. Nullopt if either is unknown (zero).
    [[nodiscard]] std::optional<double> cmPerPixel(double objectWidthCm) const {
        if (objectWidthCm <= 0 || apparentSizePx <= 0) return std::nullopt;
        return objectWidthCm / apparentSizePx;
    }
};

// Follows a learned object through a video.
//
// The detector (ObjectFinder) runs on every frame. It never drifts, but it
// misses frames when the object is blurred by motion. On a miss, a tracker
// (TrackerNano by default, about 4 ms a frame) starts from the last detected
// position and follows the object until the detector finds it again. A tracker
// drifts and cannot tell when it has lost the object, so it is only trusted for
// a short time (see MAX_TRACK_ONLY_SECONDS in object_tracker.cpp).
// See docs/tracking-design.md.
class ObjectTracker {
public:
    // Follows with a tracker of the given type; see createTrackerFactory for the model
    // files it needs. Prints an error and returns nullopt if they can't be loaded.
    [[nodiscard]] static std::optional<ObjectTracker> create(ObjectFinder finder,
                                                             const std::filesystem::path& modelsDir,
                                                             TrackerType type = DEFAULT_TRACKER);

    // Processes the next frame, taken at `now`. Returns nullopt if the object is
    // lost. `frame` must stay unmodified until the call returns.
    [[nodiscard]] std::optional<Track> update(const cv::Mat& frame, Clock::time_point now);

private:
    ObjectTracker(ObjectFinder finder, TrackerFactory makeTracker)
        : finder_(std::move(finder)), makeTracker_(std::move(makeTracker)) {}

    void reset();
    [[nodiscard]] std::optional<Track> followWithTracker(const cv::Mat& frame);

    ObjectFinder finder_;
    TrackerFactory makeTracker_;

    // The last detection, and the frame it was found in.
    std::optional<Detection> lastDetection_;
    Clock::time_point lastDetectionTime_;
    cv::Mat lastDetectionFrame_;

    // Running only while the detector is missing the object.
    cv::Ptr<cv::Tracker> tracker_;
    cv::Point2f centerOffset_;  // object center minus box center, kept constant while tracking
};

}  // namespace motion
