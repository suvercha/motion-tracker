#include "motion/object_finder.hpp"

#include "motion/segmentation.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace motion {

namespace {

constexpr int ORB_MAX_FEATURES = 1500;

// A reference feature is kept only if its best match is clearly better than the
// second best (Lowe's ratio test).
constexpr float RATIO_THRESHOLD = 0.75f;
// Matches needed, and that must agree on one homography, to call it found.
constexpr int MIN_INLIERS = 12;
constexpr double RANSAC_REPROJECTION_THRESHOLD = 3.0;
// A detected box smaller than this (in pixels) is noise, not the object.
constexpr int MIN_BOX_AREA = 400;
// Features near the object's edge mix in background and shadow, so the mask
// they are taken from is eroded by a kernel this wide.
constexpr int FEATURE_MASK_ERODE_SIZE = 15;

// The reference is shrunk to roughly the size the object appears on screen.
// ORB has a narrow scale range and misses objects far from the reference size;
// SIFT copes with a larger reference.
int referenceMaxDim(DetectorType type) { return type == DetectorType::SIFT ? 640 : 320; }

}  // namespace

std::optional<DetectorType> parseDetectorType(std::string_view text) {
    if (text == toString(DetectorType::ORB)) return DetectorType::ORB;
    if (text == toString(DetectorType::SIFT)) return DetectorType::SIFT;
    return std::nullopt;
}

ObjectFinder::ObjectFinder(DetectorType type) : type_(type) {
    if (type == DetectorType::SIFT) {
        detector_ = cv::SIFT::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_L2);
    } else {
        detector_ = cv::ORB::create(ORB_MAX_FEATURES);
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

    cv::Mat featureMask;
    cv::erode(cropMask, featureMask,
              cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(FEATURE_MASK_ERODE_SIZE, FEATURE_MASK_ERODE_SIZE)));

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
    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, RANSAC_REPROJECTION_THRESHOLD, inlierMask);
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
    if (found.area() < MIN_BOX_AREA) return std::nullopt;

    return found;
}

}  // namespace motion
