#include "motion/overlay.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <string>

namespace motion {

namespace {

constexpr double DOT_SPACING_PX = 12.0;
constexpr int DOT_RADIUS_PX = 3;
// The oldest part of the trail is still faintly visible.
constexpr double MIN_ALPHA = 0.15;

constexpr int LABEL_MARGIN_PX = 20;
constexpr int LABEL_PADDING_PX = 8;
constexpr double LABEL_FONT_SCALE = 0.8;
constexpr int LABEL_THICKNESS = 2;

double alphaForAge(double ageSeconds) {
    return std::clamp(1.0 - ageSeconds / TRAIL_SECONDS, MIN_ALPHA, 1.0);
}

// Draws a filled circle that blends into the image by `alpha`.
void blendDot(cv::Mat& image, cv::Point center, int radius, const cv::Scalar& color, double alpha) {
    cv::Rect region(center.x - radius, center.y - radius, 2 * radius + 1, 2 * radius + 1);
    region &= cv::Rect(0, 0, image.cols, image.rows);
    if (region.empty()) return;

    cv::Mat target = image(region);
    cv::Mat withDot = target.clone();
    cv::circle(withDot, center - region.tl(), radius, color, cv::FILLED, cv::LINE_AA);
    cv::addWeighted(withDot, alpha, target, 1.0 - alpha, 0, target);
}

}  // namespace

std::vector<Dot> dottedPath(const std::vector<TrailPoint>& trail, double spacingPx) {
    std::vector<Dot> dots;
    double untilNextDot = 0;  // each segment starts with a dot

    for (size_t i = 1; i < trail.size(); i++) {
        const TrailPoint& a = trail[i - 1];
        const TrailPoint& b = trail[i];
        if (a.segment != b.segment) {
            untilNextDot = 0;
            continue;
        }

        double length = cv::norm(b.position - a.position);
        double distance = untilNextDot;
        while (distance <= length) {
            double fraction = length > 0 ? distance / length : 0;
            cv::Point2f position = a.position + (b.position - a.position) * static_cast<float>(fraction);
            double age = a.ageSeconds + (b.ageSeconds - a.ageSeconds) * fraction;
            dots.push_back({position, alphaForAge(age)});
            distance += spacingPx;
        }
        untilNextDot = distance - length;
    }
    return dots;
}

void drawDottedTrail(cv::Mat& image, const std::vector<TrailPoint>& trail, const cv::Scalar& color) {
    for (const Dot& dot : dottedPath(trail, DOT_SPACING_PX)) {
        blendDot(image, cv::Point(cvRound(dot.position.x), cvRound(dot.position.y)), DOT_RADIUS_PX, color,
                 dot.alpha);
    }
}

void drawSpeedLabel(cv::Mat& image, std::optional<double> speedPxPerSecond) {
    std::string text = speedPxPerSecond ? "Speed: " + std::to_string(cvRound(*speedPxPerSecond)) + " px/s"
                                        : "Speed: --";

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
