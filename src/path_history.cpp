#include "motion/path_history.hpp"

#include <algorithm>
#include <cmath>

namespace motion {

namespace {

// The speed is fitted over this much of the most recent path.
constexpr double SPEED_WINDOW_SECONDS = 0.4;
// Need at least this many points, spanning at least this long, to fit a speed.
constexpr int MIN_SPEED_SAMPLES = 3;
constexpr double MIN_SPEED_SPAN_SECONDS = 0.15;
// If the newest point is older than this, nothing is being recorded: no speed.
constexpr double MAX_SAMPLE_AGE_SECONDS = 0.3;
// Slower than this reads as standing still (detection jitter is a few pixels).
constexpr double STATIONARY_SPEED_PX_PER_S = 15.0;

double secondsBetween(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

}  // namespace

void PathHistory::push(Sample sample) {
    samples_.push_back(std::move(sample));
    while (samples_.size() > TRAIL_LENGTH) samples_.pop_front();
}

void PathHistory::add(Clock::time_point time, cv::Point2f position, std::optional<double> cmPerPixel) {
    push({time, position, cmPerPixel});
}

void PathHistory::markLost(Clock::time_point time) { push({time, std::nullopt, std::nullopt}); }

std::vector<std::optional<cv::Point2f>> PathHistory::trail() const {
    std::vector<std::optional<cv::Point2f>> positions;
    positions.reserve(samples_.size());
    for (auto it = samples_.rbegin(); it != samples_.rend(); ++it) positions.push_back(it->position);
    return positions;
}

std::vector<const PathHistory::Sample*> PathHistory::speedWindow(Clock::time_point now) const {
    if (samples_.empty()) return {};
    const Sample& newest = samples_.back();
    if (!newest.position) return {};
    if (secondsBetween(newest.time, now) > MAX_SAMPLE_AGE_SECONDS) return {};

    std::vector<const Sample*> window;
    for (auto it = samples_.rbegin(); it != samples_.rend(); ++it) {
        if (!it->position) break;
        if (secondsBetween(it->time, newest.time) > SPEED_WINDOW_SECONDS) break;
        window.push_back(&*it);
    }
    if (static_cast<int>(window.size()) < MIN_SPEED_SAMPLES) return {};
    if (secondsBetween(window.back()->time, newest.time) < MIN_SPEED_SPAN_SECONDS) return {};
    return window;
}

std::optional<double> PathHistory::speed(Clock::time_point now) const {
    std::vector<const Sample*> window = speedWindow(now);
    if (window.empty()) return std::nullopt;
    const Sample& newest = *window.front();

    // Least-squares slope of x and of y against time. It uses every point in the
    // window, so one noisy point barely moves it.
    double n = static_cast<double>(window.size());
    double sumT = 0, sumX = 0, sumY = 0, sumTT = 0, sumTX = 0, sumTY = 0;
    for (const Sample* s : window) {
        double t = secondsBetween(newest.time, s->time);
        sumT += t;
        sumX += s->position->x;
        sumY += s->position->y;
        sumTT += t * t;
        sumTX += t * s->position->x;
        sumTY += t * s->position->y;
    }
    double denominator = n * sumTT - sumT * sumT;
    if (denominator <= 0) return std::nullopt;
    double vx = (n * sumTX - sumT * sumX) / denominator;
    double vy = (n * sumTY - sumT * sumY) / denominator;

    double speed = std::hypot(vx, vy);
    return speed < STATIONARY_SPEED_PX_PER_S ? 0.0 : speed;
}

std::optional<double> PathHistory::metricSpeed(Clock::time_point now) const {
    std::optional<double> pixelsPerSecond = speed(now);
    if (!pixelsPerSecond) return std::nullopt;

    std::vector<double> scales;
    for (const Sample* s : speedWindow(now)) {
        if (s->cmPerPixel) scales.push_back(*s->cmPerPixel);
    }
    if (scales.empty()) return std::nullopt;

    // The median, so one badly measured size doesn't swing the result.
    std::nth_element(scales.begin(), scales.begin() + scales.size() / 2, scales.end());
    return *pixelsPerSecond * scales[scales.size() / 2];
}

}  // namespace motion
