#pragma once

#include "motion/clock.hpp"

#include <opencv2/core.hpp>
#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

namespace motion {

// How many past frames the trail remembers. A smaller number gives a shorter tail and a
// larger number a longer one (at 30 fps, 64 frames is about 2 seconds).
constexpr size_t TRAIL_LENGTH = 64;

// The recent path of a moving object, one entry per camera frame: gives the points of its
// trail, and its speed.
class PathHistory {
public:
    // Records that the object was at `position` at `time`. Times must not go backwards.
    // `cmPerPixel` is the real size of a pixel at the object at that moment, if known;
    // it is only needed for metricSpeed.
    void add(Clock::time_point time, cv::Point2f position, std::optional<double> cmPerPixel = std::nullopt);

    // Records that the object was not found in the frame taken at `time`. It takes a
    // place in the trail like any other frame, so a lost object's tail fades away, and
    // the trail is not joined across the gap.
    void markLost(Clock::time_point time);

    // The positions from the last TRAIL_LENGTH frames, newest first. Frames where the
    // object was not found are nullopt.
    [[nodiscard]] std::vector<std::optional<cv::Point2f>> trail() const;

    // Speed in pixels per second, from a straight-line fit over the last fraction of a
    // second (single-frame differences are far too noisy). Returns 0 below a small
    // threshold so detection jitter on a still object doesn't read as motion, and
    // nullopt if the object is lost or there are too few recent points to say.
    [[nodiscard]] std::optional<double> speed(Clock::time_point now) const;

    // The same speed in centimetres per second: the pixel speed times the real size of a
    // pixel (the median over the same window). Only right for motion across the view at
    // about constant distance; motion toward or away from the camera is not measured.
    // Nullopt if there is no pixel speed, or no point in the window gave a pixel size.
    [[nodiscard]] std::optional<double> metricSpeed(Clock::time_point now) const;

private:
    struct Sample {
        Clock::time_point time;
        std::optional<cv::Point2f> position;  // nullopt: the object was lost in this frame
        std::optional<double> cmPerPixel;
    };

    // The newest samples that the speed is fitted to, newest first, up to the first lost
    // frame. Empty if there are too few, or the newest is lost or too old.
    [[nodiscard]] std::vector<const Sample*> speedWindow(Clock::time_point now) const;

    void push(Sample sample);

    std::deque<Sample> samples_;
};

}  // namespace motion
