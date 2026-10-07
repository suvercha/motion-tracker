#include "object_finder.hpp"

#include "segmentation.hpp"

#include <opencv2/geometry.hpp>
#include <algorithm>

namespace {

constexpr float RATIO_THRESHOLD = 0.75f;
constexpr int MIN_INLIERS = 12;

// The reference is shrunk to roughly the size the object appears on screen.
// ORB has a narrow scale range and misses objects far from the reference size;
// SIFT copes with a larger reference.
int referenceMaxDim(DetectorType type) { return type == DetectorType::SIFT ? 640 : 320; }

}  // namespace

ObjectFinder::ObjectFinder(DetectorType type) : type_(type) {
    if (type == DetectorType::SIFT) {
        detector_ = cv::SIFT::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_L2);
    } else {
        detector_ = cv::ORB::create(1500);
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
    }
}

std::optional<ObjectFinder> ObjectFinder::fromPhoto(const cv::Mat& photo, DetectorType type) {
    ObjectFinder finder(type);
    if (!finder.learn(photo)) return std::nullopt;
    return finder;
}

bool ObjectFinder::learn(const cv::Mat& photo) {
    cv::Mat mask = isolateObject(photo, centeredRect(photo.size(), PHOTO_SEED_FRACTION));
    if (mask.empty()) return false;

    cv::Rect box = cv::boundingRect(mask);
    cv::Mat crop = photo(box).clone();
    cv::Mat cropMask = mask(box).clone();

    double scale = std::min(1.0, static_cast<double>(referenceMaxDim(type_)) / std::max(crop.cols, crop.rows));
    if (scale < 1.0) {
        cv::resize(crop, crop, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(cropMask, cropMask, crop.size(), 0, 0, cv::INTER_NEAREST);
    }
    size_ = crop.size();

    // Outline used later to draw a tight box around the object.
    auto contour = largestContour(cropMask);
    if (!contour) return false;
    outline_.clear();
    for (const auto& p : *contour) outline_.emplace_back(p);

    // Shrink the mask a little so features on the object's edge, which mix in
    // background and shadow, are not learned.
    cv::Mat featureMask;
    cv::erode(cropMask, featureMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    detector_->detectAndCompute(gray, featureMask, keypoints_, descriptors_);
    return static_cast<int>(keypoints_.size()) >= MIN_INLIERS;
}

std::optional<cv::Rect> ObjectFinder::find(const cv::Mat& frame) const {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    if (descriptors.rows < 2) return std::nullopt;

    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(descriptors_, descriptors, knn, 2);

    std::vector<cv::Point2f> src, dst;
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < RATIO_THRESHOLD * m[1].distance) {
            src.push_back(keypoints_[m[0].queryIdx].pt);
            dst.push_back(keypoints[m[0].trainIdx].pt);
        }
    }
    if (static_cast<int>(src.size()) < MIN_INLIERS) return std::nullopt;

    cv::Mat inlierMask;
    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 3.0, inlierMask);
    if (H.empty() || cv::countNonZero(inlierMask) < MIN_INLIERS) return std::nullopt;

    // Reject twisted/flipped mappings: the reference's corners must stay convex.
    std::vector<cv::Point2f> corners = {
        {0, 0}, {static_cast<float>(size_.width), 0},
        {static_cast<float>(size_.width), static_cast<float>(size_.height)},
        {0, static_cast<float>(size_.height)}};
    std::vector<cv::Point2f> projectedCorners;
    cv::perspectiveTransform(corners, projectedCorners, H);
    if (!cv::isContourConvex(projectedCorners)) return std::nullopt;

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(outline_, projected, H);
    cv::Rect found = cv::boundingRect(projected) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (found.area() < 400) return std::nullopt;

    return found;
}
