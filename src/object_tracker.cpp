#include "motion/object_tracker.hpp"

#include <chrono>
#include <iostream>

namespace motion {

namespace {

// Without a detection for this long, the object is declared lost. Trackers drift
// onto the background, and TrackerNano kept reporting a box (with an unchanged
// confidence score) for as long as it was measured after the object had left, so
// it must not be trusted for long.
constexpr double MAX_TRACK_ONLY_SECONDS = 1.0;

constexpr const char* BACKBONE_FILE = "nanotrack_backbone_sim.onnx";
constexpr const char* NECKHEAD_FILE = "nanotrack_head_sim.onnx";

// Loads a network. OpenCV runs it on the CPU by default. Throws cv::Exception if the
// file is missing or invalid.
cv::dnn::Net loadNet(const std::filesystem::path& path) { return cv::dnn::readNet(path.string()); }

// A tracked box smaller than this (pixels), or whose area differs from the last
// detection's by more than this factor, is the tracker having lost the object.
constexpr int MIN_BOX_AREA = 400;
constexpr double MAX_AREA_CHANGE = 2.5;

cv::Point2f boxCenter(const cv::Rect& box) {
    return {box.x + box.width / 2.0f, box.y + box.height / 2.0f};
}

}  // namespace

std::optional<ObjectTracker> ObjectTracker::create(ObjectFinder finder, const std::filesystem::path& modelsDir) {
    try {
        return ObjectTracker(std::move(finder), loadNet(modelsDir / BACKBONE_FILE), loadNet(modelsDir / NECKHEAD_FILE));
    } catch (const cv::Exception&) {
        std::cerr << "Error: could not load the tracker models (" << BACKBONE_FILE << " and " << NECKHEAD_FILE
                  << ") from '" << modelsDir.string() << "'.\n"
                  << "Run scripts/download_models.sh to download them, or pass --models=DIR." << std::endl;
        return std::nullopt;
    }
}

void ObjectTracker::reset() {
    tracker_.release();
    lastDetection_.reset();
    lastDetectionFrame_.release();
}

std::optional<Track> ObjectTracker::update(const cv::Mat& frame, Clock::time_point now) {
    if (std::optional<Detection> detection = finder_.find(frame)) {
        lastDetection_ = *detection;
        lastDetectionTime_ = now;
        lastDetectionFrame_ = frame.clone();
        centerOffset_ = detection->center - boxCenter(detection->box);
        tracker_.release();  // stale; restarted from this detection if the next frame is a miss
        return Track{detection->box, detection->center, true};
    }

    if (!lastDetection_) return std::nullopt;
    if (std::chrono::duration<double>(now - lastDetectionTime_).count() > MAX_TRACK_ONLY_SECONDS) {
        reset();
        return std::nullopt;
    }
    return followWithTracker(frame);
}

std::optional<Track> ObjectTracker::followWithTracker(const cv::Mat& frame) {
    // First miss after a detection: start following from where it was last seen.
    // That was the previous frame, so the tracker is started on that frame.
    if (!tracker_) {
        tracker_ = cv::TrackerNano::create(backbone_, neckhead_);
        tracker_->init(lastDetectionFrame_, lastDetection_->box);
    }

    cv::Rect box;
    if (!tracker_->update(frame, box)) {
        reset();
        return std::nullopt;
    }

    box &= cv::Rect(0, 0, frame.cols, frame.rows);
    double areaChange = static_cast<double>(box.area()) / lastDetection_->box.area();
    if (box.area() < MIN_BOX_AREA || areaChange > MAX_AREA_CHANGE || areaChange < 1.0 / MAX_AREA_CHANGE) {
        reset();
        return std::nullopt;
    }
    return Track{box, boxCenter(box) + centerOffset_, false};
}

}  // namespace motion
