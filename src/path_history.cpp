#include "motion/path_history.hpp"

#include <cmath>

namespace motion {

namespace {

// The speed is fitted over this much of the most recent path.
constexpr double SPEED_WINDOW_SECONDS = 0.4;
// Need at least this many points, spanning at least this long, to fit a speed.
constexpr int MIN_SPEED_SAMPLES = 3;
constexpr double MIN_SPEED_SPAN_SECONDS = 0.15;
// If the newest point is older than this, the object has been lost: no speed.
constexpr double MAX_SAMPLE_AGE_SECONDS = 0.3;
// Slower than this reads as standing still (detection jitter is a few pixels).
constexpr double STATIONARY_SPEED_PX_PER_S = 15.0;

double secondsBetween(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

}  // namespace

void PathHistory::add(Clock::time_point time, cv::Point2f position) {
    if (startNewSegment_) {
        segment_++;
        startNewSegment_ = false;
    }
    samples_.push_back({time, position, segment_});

    while (!samples_.empty() && secondsBetween(samples_.front().time, time) > TRAIL_SECONDS) {
        samples_.pop_front();
    }
}

void PathHistory::breakPath() {
    if (!samples_.empty()) startNewSegment_ = true;
}

std::vector<TrailPoint> PathHistory::trail(Clock::time_point now) const {
    std::vector<TrailPoint> points;
    for (const Sample& s : samples_) {
        double age = secondsBetween(s.time, now);
        if (age <= TRAIL_SECONDS) points.push_back({s.position, age, s.segment});
    }
    return points;
}

std::optional<double> PathHistory::speed(Clock::time_point now) const {
    if (samples_.empty()) return std::nullopt;
    const Sample& newest = samples_.back();
    if (secondsBetween(newest.time, now) > MAX_SAMPLE_AGE_SECONDS) return std::nullopt;

    // Newest samples of the current segment, within the speed window.
    std::vector<const Sample*> window;
    for (auto it = samples_.rbegin(); it != samples_.rend(); ++it) {
        if (it->segment != newest.segment) break;
        if (secondsBetween(it->time, newest.time) > SPEED_WINDOW_SECONDS) break;
        window.push_back(&*it);
    }
    if (static_cast<int>(window.size()) < MIN_SPEED_SAMPLES) return std::nullopt;
    if (secondsBetween(window.back()->time, newest.time) < MIN_SPEED_SPAN_SECONDS) return std::nullopt;

    // Least-squares slope of x and of y against time. It uses every point in the
    // window, so one noisy point barely moves it.
    double n = static_cast<double>(window.size());
    double sumT = 0, sumX = 0, sumY = 0, sumTT = 0, sumTX = 0, sumTY = 0;
    for (const Sample* s : window) {
        double t = secondsBetween(newest.time, s->time);
        sumT += t;
        sumX += s->position.x;
        sumY += s->position.y;
        sumTT += t * t;
        sumTX += t * s->position.x;
        sumTY += t * s->position.y;
    }
    double denominator = n * sumTT - sumT * sumT;
    if (denominator <= 0) return std::nullopt;
    double vx = (n * sumTX - sumT * sumX) / denominator;
    double vy = (n * sumTY - sumT * sumY) / denominator;

    double speed = std::hypot(vx, vy);
    return speed < STATIONARY_SPEED_PX_PER_S ? 0.0 : speed;
}

}  // namespace motion
