#pragma once

#include "motion/path_history.hpp"

#include <opencv2/core.hpp>
#include <cstddef>
#include <optional>
#include <vector>

namespace motion {

// How thick the trail's line is at segment `index`, where 1 is the newest segment (between
// the newest and the second newest point) and TRAIL_LENGTH - 1 is the oldest. The trail
// tapers: thick at the object, thin at its tail.
[[nodiscard]] int trailThickness(size_t index);

// Draws the trail (as returned by PathHistory::trail, newest first) as connected line
// segments that get thinner with age. Frames where the object was lost leave a gap.
void drawTrail(cv::Mat& image, const std::vector<std::optional<cv::Point2f>>& trail, const cv::Scalar& color);

// Draws the speed in the top-left corner: centimetres per second as well as pixels per
// second if the real size is known, or "--" if the speed is not known.
void drawSpeedLabel(cv::Mat& image, std::optional<double> pixelsPerSecond,
                    std::optional<double> cmPerSecond = std::nullopt);

}  // namespace motion
