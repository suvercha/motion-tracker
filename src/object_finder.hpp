#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// See docs/feature-detector-decision.md for the trade-offs between the two.
enum class DetectorType { ORB, SIFT };

// Learns an object from a photo, then finds it in other images.
class ObjectFinder {
public:
    explicit ObjectFinder(DetectorType type);

    // Learns the object from a photo of it against a plain background, roughly
    // centered. Returns false if it can't be isolated or has too few features.
    bool learn(const cv::Mat& photo);

    // Looks for the learned object in `frame`. On success returns true and sets
    // `box` to the object's bounding box in frame coordinates.
    bool find(const cv::Mat& frame, cv::Rect& box) const;

    size_t featureCount() const { return keypoints_.size(); }

    // Binary mask (255 = object) of the object in `photo`. Empty on failure.
    static cv::Mat isolateObject(const cv::Mat& photo);

private:
    DetectorType type_;
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;

    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    std::vector<cv::Point2f> outline_;  // object outline, in reference-image coordinates
    cv::Size size_;
};
