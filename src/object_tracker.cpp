#include "motion/object_tracker.hpp"

#include <opencv2/tracking.hpp>
#include <chrono>
#include <iostream>
#include <vector>

namespace motion {

namespace {

// Without a detection for this long, the object is declared lost. Trackers drift
// onto the background, and all but KCF kept reporting a box (TrackerNano with an
// unchanged confidence score) for as long as it was measured after the object had
// left, so a tracker must not be trusted for long.
constexpr double MAX_TRACK_ONLY_SECONDS = 1.0;

constexpr const char* NANO_BACKBONE_FILE = "nanotrack_backbone_sim.onnx";
constexpr const char* NANO_NECKHEAD_FILE = "nanotrack_head_sim.onnx";
constexpr const char* VIT_FILE = "object_tracking_vittrack_2023sep.onnx";

// The model files a tracker type needs (the older trackers need none).
std::vector<const char*> modelFiles(TrackerType type) {
    switch (type) {
        case TrackerType::Nano: return {NANO_BACKBONE_FILE, NANO_NECKHEAD_FILE};
        case TrackerType::Vit: return {VIT_FILE};
        default: return {};
    }
}

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

std::optional<TrackerType> parseTrackerType(std::string_view text) {
    for (TrackerType type : ALL_TRACKER_TYPES) {
        if (text == toString(type)) return type;
    }
    return std::nullopt;
}

std::optional<TrackerFactory> createTrackerFactory(TrackerType type, const std::filesystem::path& modelsDir) {
    try {
        switch (type) {
            case TrackerType::Nano: {
                cv::dnn::Net backbone = loadNet(modelsDir / NANO_BACKBONE_FILE);
                cv::dnn::Net neckhead = loadNet(modelsDir / NANO_NECKHEAD_FILE);
                return TrackerFactory([backbone, neckhead]() -> cv::Ptr<cv::Tracker> {
                    return cv::TrackerNano::create(backbone, neckhead);
                });
            }
            case TrackerType::Vit: {
                cv::dnn::Net net = loadNet(modelsDir / VIT_FILE);
                return TrackerFactory([net]() -> cv::Ptr<cv::Tracker> { return cv::TrackerVit::create(net); });
            }
            case TrackerType::Mil:
                return TrackerFactory([]() -> cv::Ptr<cv::Tracker> { return cv::TrackerMIL::create(); });
            case TrackerType::Kcf:
                return TrackerFactory([]() -> cv::Ptr<cv::Tracker> { return cv::TrackerKCF::create(); });
            case TrackerType::Csrt:
                return TrackerFactory([]() -> cv::Ptr<cv::Tracker> { return cv::TrackerCSRT::create(); });
        }
    } catch (const cv::Exception&) {
        std::cerr << "Error: could not load the " << toString(type) << " tracker's model files (";
        const char* separator = "";
        for (const char* file : modelFiles(type)) {
            std::cerr << separator << file;
            separator = " and ";
        }
        std::cerr << ") from '" << modelsDir.string() << "'.\n"
                  << "Run scripts/download_models.sh to download them, or pass --models=DIR." << std::endl;
    }
    return std::nullopt;
}

std::optional<ObjectTracker> ObjectTracker::create(ObjectFinder finder, const std::filesystem::path& modelsDir,
                                                   TrackerType type) {
    std::optional<TrackerFactory> factory = createTrackerFactory(type, modelsDir);
    if (!factory) return std::nullopt;
    return ObjectTracker(std::move(finder), std::move(*factory));
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
        return Track{detection->box, detection->center, true, detection->apparentSizePx};
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
        tracker_ = makeTracker_();
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
    return Track{box, boxCenter(box) + centerOffset_, false, lastDetection_->apparentSizePx};
}

}  // namespace motion
