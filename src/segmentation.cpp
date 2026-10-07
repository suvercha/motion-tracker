#include "segmentation.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <algorithm>
#include <cmath>

namespace {

// GrabCut tends to keep the object's shadow, because a shadow is only a darker
// copy of the background. A shadow keeps the background's hue, so drop every
// pixel that matches the background's hue and saturation, whatever its
// brightness. The background is estimated from a strip around the image's
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

}  // namespace

cv::Rect centeredRect(cv::Size size, double fraction) {
    int width = static_cast<int>(size.width * fraction);
    int height = static_cast<int>(size.height * fraction);
    return {(size.width - width) / 2, (size.height - height) / 2, width, height};
}

std::optional<std::vector<cv::Point>> largestContour(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return std::nullopt;

    return *std::max_element(
        contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
}

cv::Mat isolateObject(const cv::Mat& image, const cv::Rect& seed) {
    cv::Mat mask, bgdModel, fgdModel;
    cv::grabCut(image, mask, seed, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);
    cv::Mat fg = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);

    removeBackgroundHue(image, fg);

    // GrabCut can leave small speckles; the largest blob is the object.
    auto contour = largestContour(fg);
    if (!contour) return cv::Mat();

    cv::Mat clean = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::drawContours(clean, std::vector<std::vector<cv::Point>>{*contour}, 0, cv::Scalar(255), cv::FILLED);
    return clean;
}

cv::Mat cutOutObject(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat cutout(image.size(), image.type(), cv::Scalar(255, 255, 255));
    image.copyTo(cutout, mask);
    return cutout(cv::boundingRect(mask)).clone();
}
