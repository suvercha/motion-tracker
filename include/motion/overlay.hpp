#pragma once

#include "motion/path_history.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace motion {

// One dot of a dotted line. `alpha` is 1 for the newest part of the trail and
// fades toward 0 for the oldest.
struct Dot {
    cv::Point2f position;
    double alpha;
};

// Places dots at a fixed distance `spacingPx` along the trail, so they stay
// evenly spaced however fast the object moves. Segments are not connected to
// each other.
[[nodiscard]] std::vector<Dot> dottedPath(const std::vector<TrailPoint>& trail, double spacingPx);

// Draws the trail as a dotted line that fades with age.
void drawDottedTrail(cv::Mat& image, const std::vector<TrailPoint>& trail, const cv::Scalar& color);

// Draws the speed in the top-left corner, or "--" if it is not known.
void drawSpeedLabel(cv::Mat& image, std::optional<double> speedPxPerSecond);

}  // namespace motion
