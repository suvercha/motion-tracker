#include "motion/segmentation.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace motion {

namespace {

constexpr int GRABCUT_ITERATIONS = 5;

// OpenCV stores hue as 0..179, so a full turn of the color wheel is 180.
constexpr double HUE_RANGE = 180.0;

// A pixel counts as background if its hue is within this of the background's...
constexpr double MAX_HUE_DIFF = 10;
// ...and it is at least this fraction as saturated as the background.
constexpr double MIN_SATURATION_RATIO = 0.5;
// Hue is meaningless for gray backgrounds; below this saturation, do nothing.
constexpr double MIN_BG_SATURATION = 30;
// The background is sampled from a strip this fraction of the image's short side wide.
constexpr double BORDER_FRACTION = 0.03;
constexpr int MIN_BORDER_PIXELS = 4;

constexpr int NOISE_KERNEL_SIZE = 9;

// GrabCut tends to keep the object's shadow, because a shadow is only a darker
// copy of the background. A shadow keeps the background's hue, so drop every
// pixel that matches the background's hue and saturation, whatever its
// brightness. The background is estimated from a strip around the image's
// edge. Does nothing if that strip is not clearly colored.
void removeBackgroundHue(const cv::Mat& image, cv::Mat& fg) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat channels[3];
    cv::split(hsv, channels);
    cv::Mat hue, saturation;
    channels[0].convertTo(hue, CV_32F);
    channels[1].convertTo(saturation, CV_32F);

    int borderWidth = std::max(MIN_BORDER_PIXELS,
                               static_cast<int>(std::min(image.cols, image.rows) * BORDER_FRACTION));
    cv::Mat border(image.size(), CV_8UC1, cv::Scalar(255));
    border(cv::Rect(borderWidth, borderWidth, image.cols - 2 * borderWidth, image.rows - 2 * borderWidth))
        .setTo(0);

    double bgSaturation = cv::mean(saturation, border)[0];
    if (bgSaturation < MIN_BG_SATURATION) return;

    // Hue is an angle (0 and 179 are neighbors), so average it as a point on a circle.
    cv::Mat angle, cosHue, sinHue;
    hue.convertTo(angle, CV_32F, 2.0 * CV_PI / HUE_RANGE);
    cv::polarToCart(cv::noArray(), angle, cosHue, sinHue);
    double bgHue = std::atan2(cv::mean(sinHue, border)[0], cv::mean(cosHue, border)[0]) * HUE_RANGE / (2.0 * CV_PI);
    if (bgHue < 0) bgHue += HUE_RANGE;

    // Distance around the circle: the shorter of the two ways between the hues.
    cv::Mat direct, wrapped, hueDistance;
    cv::absdiff(hue, cv::Scalar(bgHue), direct);
    wrapped = HUE_RANGE - direct;
    cv::min(direct, wrapped, hueDistance);

    cv::Mat bgLike = (hueDistance <= MAX_HUE_DIFF) & (saturation >= MIN_SATURATION_RATIO * bgSaturation);
    fg.setTo(0, bgLike);
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(NOISE_KERNEL_SIZE, NOISE_KERNEL_SIZE)));
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
    cv::grabCut(image, mask, seed, bgdModel, fgdModel, GRABCUT_ITERATIONS, cv::GC_INIT_WITH_RECT);
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

}  // namespace motion
