#include "object_finder.hpp"

#include <opencv2/geometry.hpp>
#include <algorithm>
#include <cmath>

namespace {

constexpr float RATIO_THRESHOLD = 0.75f;
constexpr int MIN_INLIERS = 12;

// The reference is shrunk to roughly the size the object appears on screen.
// ORB has a narrow scale range and misses objects far from the reference size;
// SIFT copes with a larger reference.
int referenceMaxDim(DetectorType type) { return type == DetectorType::SIFT ? 640 : 320; }

// GrabCut tends to keep the object's shadow, because a shadow is only a darker
// copy of the background. A shadow keeps the background's hue, so drop every
// pixel that matches the background's hue and saturation, whatever its
// brightness. The background is estimated from a strip around the photo's
// edge. Does nothing if that strip is not clearly colored (hue is meaningless
// for gray backgrounds).
void removeBackgroundHue(const cv::Mat& image, cv::Mat& fg) {
    constexpr int MAX_HUE_DIFF = 10;  // OpenCV hue range is 0..179
    constexpr double MIN_BG_SATURATION = 30;

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int bw = std::max(4, static_cast<int>(std::min(image.cols, image.rows) * 0.03));
    cv::Mat border(image.size(), CV_8UC1, cv::Scalar(255));
    border(cv::Rect(bw, bw, image.cols - 2 * bw, image.rows - 2 * bw)).setTo(0);

    // Circular mean of the hue, and mean saturation, over the border strip.
    double sumSin = 0, sumCos = 0, sumSat = 0;
    int n = 0;
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            if (!border.at<uchar>(y, x)) continue;
            cv::Vec3b p = hsv.at<cv::Vec3b>(y, x);
            double angle = p[0] * CV_PI / 90.0;
            sumSin += std::sin(angle);
            sumCos += std::cos(angle);
            sumSat += p[1];
            n++;
        }
    }
    double bgSat = sumSat / n;
    if (bgSat < MIN_BG_SATURATION) return;
    double bgHue = std::atan2(sumSin, sumCos) * 90.0 / CV_PI;
    if (bgHue < 0) bgHue += 180;

    cv::Mat bgLike(image.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            cv::Vec3b p = hsv.at<cv::Vec3b>(y, x);
            double diff = std::abs(p[0] - bgHue);
            diff = std::min(diff, 180.0 - diff);
            if (diff <= MAX_HUE_DIFF && p[1] >= 0.5 * bgSat) bgLike.at<uchar>(y, x) = 255;
        }
    }
    fg.setTo(0, bgLike);
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
}

const std::vector<cv::Point>& largestContour(const std::vector<std::vector<cv::Point>>& contours) {
    return *std::max_element(
        contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
}

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

// Separates the object from its background with GrabCut, assuming it sits
// roughly in the middle of the photo with some background around it.
cv::Mat ObjectFinder::isolateObject(const cv::Mat& image) {
    int mx = image.cols * 0.08;
    int my = image.rows * 0.08;
    cv::Rect seed(mx, my, image.cols - 2 * mx, image.rows - 2 * my);

    cv::Mat mask, bgdModel, fgdModel;
    cv::grabCut(image, mask, seed, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);
    cv::Mat fg = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);

    removeBackgroundHue(image, fg);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return cv::Mat();

    cv::Mat clean = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::drawContours(clean, std::vector<std::vector<cv::Point>>{largestContour(contours)}, 0,
                     cv::Scalar(255), cv::FILLED);
    return clean;
}

bool ObjectFinder::learn(const cv::Mat& photo) {
    cv::Mat mask = isolateObject(photo);
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
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cropMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;
    outline_.clear();
    for (const auto& p : largestContour(contours)) outline_.emplace_back(p);

    // Shrink the mask a little so features on the object's edge, which mix in
    // background and shadow, are not learned.
    cv::Mat featureMask;
    cv::erode(cropMask, featureMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    detector_->detectAndCompute(gray, featureMask, keypoints_, descriptors_);
    return static_cast<int>(keypoints_.size()) >= MIN_INLIERS;
}

bool ObjectFinder::find(const cv::Mat& frame, cv::Rect& box) const {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    if (descriptors.rows < 2) return false;

    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(descriptors_, descriptors, knn, 2);

    std::vector<cv::Point2f> src, dst;
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < RATIO_THRESHOLD * m[1].distance) {
            src.push_back(keypoints_[m[0].queryIdx].pt);
            dst.push_back(keypoints[m[0].trainIdx].pt);
        }
    }
    if (static_cast<int>(src.size()) < MIN_INLIERS) return false;

    cv::Mat inlierMask;
    cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 3.0, inlierMask);
    if (H.empty() || cv::countNonZero(inlierMask) < MIN_INLIERS) return false;

    // Reject twisted/flipped mappings: the reference's corners must stay convex.
    std::vector<cv::Point2f> corners = {
        {0, 0}, {static_cast<float>(size_.width), 0},
        {static_cast<float>(size_.width), static_cast<float>(size_.height)},
        {0, static_cast<float>(size_.height)}};
    std::vector<cv::Point2f> projectedCorners;
    cv::perspectiveTransform(corners, projectedCorners, H);
    if (!cv::isContourConvex(projectedCorners)) return false;

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(outline_, projected, H);
    cv::Rect found = cv::boundingRect(projected) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (found.area() < 400) return false;

    box = found;
    return true;
}
