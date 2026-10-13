#include "motion/overlay.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace motion {

namespace {

// The trail's line is this many pixels thick at its thickest, times sqrt(TRAIL_LENGTH / 2).
constexpr double TRAIL_THICKNESS_FACTOR = 2.5;

constexpr int LABEL_MARGIN_PX = 20;
constexpr int LABEL_PADDING_PX = 8;
constexpr double LABEL_FONT_SCALE = 0.8;
constexpr int LABEL_THICKNESS = 2;

}  // namespace

int trailThickness(size_t index) {
    double thickness = std::sqrt(static_cast<double>(TRAIL_LENGTH) / static_cast<double>(index + 1)) * TRAIL_THICKNESS_FACTOR;
    return std::max(1, static_cast<int>(thickness));
}

void drawTrail(cv::Mat& image, const std::vector<std::optional<cv::Point2f>>& trail, const cv::Scalar& color) {
    for (size_t i = 1; i < trail.size(); i++) {
        // No line to or from a frame where the object was lost.
        if (!trail[i - 1] || !trail[i]) continue;

        cv::Point newer(cvRound(trail[i - 1]->x), cvRound(trail[i - 1]->y));
        cv::Point older(cvRound(trail[i]->x), cvRound(trail[i]->y));
        cv::line(image, newer, older, color, trailThickness(i), cv::LINE_AA);
    }
}

void drawSpeedLabel(cv::Mat& image, std::optional<double> pixelsPerSecond, std::optional<double> cmPerSecond) {
    std::ostringstream label;
    label << "Speed: ";
    if (!pixelsPerSecond) {
        label << "--";
    } else if (cmPerSecond) {
        label << std::fixed << std::setprecision(1) << *cmPerSecond << " cm/s  (" << cvRound(*pixelsPerSecond) << " px/s)";
    } else {
        label << cvRound(*pixelsPerSecond) << " px/s";
    }
    std::string text = label.str();

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS, &baseline);
    cv::Point origin(LABEL_MARGIN_PX, LABEL_MARGIN_PX + textSize.height);

    cv::Rect background(origin.x - LABEL_PADDING_PX, origin.y - textSize.height - LABEL_PADDING_PX,
                        textSize.width + 2 * LABEL_PADDING_PX, textSize.height + baseline + 2 * LABEL_PADDING_PX);
    cv::rectangle(image, background, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, cv::Scalar(255, 255, 255),
                LABEL_THICKNESS, cv::LINE_AA);
}

}  // namespace motion
