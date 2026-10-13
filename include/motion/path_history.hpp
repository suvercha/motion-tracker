#pragma once

#include "motion/clock.hpp"

#include <opencv2/core.hpp>
#include <deque>
#include <optional>
#include <vector>

namespace motion {

// How far back the drawn trail reaches.
constexpr double TRAIL_SECONDS = 3.0;

// One point of the drawn trail.
struct TrailPoint {
    cv::Point2f position;
    double ageSeconds;
    int segment;  // points in different segments are not connected (the object was lost between them)
};

// The recent path of a moving object: gives the points of its trail, and its speed.
class PathHistory {
public:
    // Records that the object was at `position` at `time`. Times must not go backwards.
    void add(Clock::time_point time, cv::Point2f position);

    // The object was lost. The next `add` starts a new, unconnected segment.
    void breakPath();

    // Points from the last TRAIL_SECONDS, oldest first.
    [[nodiscard]] std::vector<TrailPoint> trail(Clock::time_point now) const;

    // Speed in pixels per second, from a straight-line fit over the last fraction of a
    // second (single-frame differences are far too noisy). Returns 0 below a small
    // threshold so detection jitter on a still object doesn't read as motion, and
    // nullopt if there are too few recent points to say.
    [[nodiscard]] std::optional<double> speed(Clock::time_point now) const;

private:
    struct Sample {
        Clock::time_point time;
        cv::Point2f position;
        int segment;
    };

    std::deque<Sample> samples_;
    int segment_ = 0;
    bool startNewSegment_ = false;
};

}  // namespace motion
