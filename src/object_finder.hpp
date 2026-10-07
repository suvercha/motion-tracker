#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

// See docs/feature-detector-decision.md for the trade-offs between the two.
enum class DetectorType { ORB, SIFT };

// An object learned from a photo, which can then be found in other images.
class ObjectFinder {
public:
    // Learns the object from a photo of it against a plain background, roughly
    // centered. Returns nullopt if it can't be isolated or has too few features.
    [[nodiscard]] static std::optional<ObjectFinder> fromPhoto(const cv::Mat& photo, DetectorType type);

    // Looks for the learned object in `frame` and returns its bounding box in
    // frame coordinates, or nullopt if it isn't found.
    [[nodiscard]] std::optional<cv::Rect> find(const cv::Mat& frame) const;

    [[nodiscard]] size_t featureCount() const { return keypoints_.size(); }

private:
    explicit ObjectFinder(DetectorType type);
    bool learn(const cv::Mat& photo);

    DetectorType type_;
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;

    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    std::vector<cv::Point2f> outline_;  // object outline, in reference-image coordinates
    cv::Size size_;
};
