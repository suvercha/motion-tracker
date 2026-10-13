#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features.hpp>
#include <optional>
#include <string_view>
#include <vector>

namespace motion {

// See docs/feature-detector-decision.md for the trade-offs between the two.
enum class DetectorType { ORB, SIFT };

[[nodiscard]] constexpr std::string_view toString(DetectorType type) {
    return type == DetectorType::SIFT ? "sift" : "orb";
}

// "orb" or "sift" (lowercase), or nullopt for anything else.
[[nodiscard]] std::optional<DetectorType> parseDetectorType(std::string_view text);

// Where the object was found in an image.
struct Detection {
    cv::Rect box;        // bounding box of the object's outline
    cv::Point2f center;  // the object's centroid; steadier than the box center when it rotates
    // The object's widest dimension as it appears in the frame, in pixels. With the
    // object's real width, this gives the real size of a pixel at the object's distance.
    double apparentSizePx;
};

// An object learned from a photo, which can then be found in other images.
class ObjectFinder {
public:
    // Learns the object from a photo of it against a plain background, roughly
    // centered. Returns nullopt if it can't be isolated or has too few features.
    [[nodiscard]] static std::optional<ObjectFinder> fromPhoto(const cv::Mat& photo, DetectorType type);

    // Looks for the learned object in `frame` and returns where it is, in frame
    // coordinates, or nullopt if it isn't found.
    [[nodiscard]] std::optional<Detection> find(const cv::Mat& frame) const;

    [[nodiscard]] size_t featureCount() const { return keypoints_.size(); }

private:
    explicit ObjectFinder(DetectorType type);
    [[nodiscard]] bool learn(const cv::Mat& photo);

    DetectorType type_;
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;

    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    std::vector<cv::Point2f> outline_;  // object outline, in reference-image coordinates
    cv::Point2f centroid_;              // object centroid, in reference-image coordinates
    cv::Size size_;
};

}  // namespace motion
