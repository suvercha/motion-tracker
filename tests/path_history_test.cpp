// Unit tests for PathHistory (trail and speed) and the dotted-line placement.
// Uses made-up timestamps, so it is fast and exact. No camera or photo needed.

#include "motion/overlay.hpp"
#include "motion/path_history.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace motion;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    std::cout << (condition ? "ok:   " : "FAIL: ") << what << '\n';
    if (!condition) failures++;
}

Clock::time_point at(double seconds) {
    return Clock::time_point{} + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

constexpr double FRAME_SECONDS = 1.0 / 30;

// An object moving at a constant velocity (pixels per second), sampled at 30 fps for `seconds`.
PathHistory straightPath(cv::Point2f velocity, double seconds, cv::Point2f start = {100, 100}) {
    PathHistory path;
    for (double t = 0; t <= seconds + 1e-9; t += FRAME_SECONDS) {
        path.add(at(t), start + velocity * static_cast<float>(t));
    }
    return path;
}

void testConstantVelocity() {
    cv::Point2f velocity(180, 135);  // 225 px/s
    PathHistory path = straightPath(velocity, 1.0);
    std::optional<double> speed = path.speed(at(1.0));
    check(speed.has_value() && std::abs(*speed - 225.0) < 225.0 * 0.02, "constant velocity: speed within 2% of 225 px/s");

    PathHistory slow = straightPath({-40, 0}, 1.0);
    speed = slow.speed(at(1.0));
    check(speed.has_value() && std::abs(*speed - 40.0) < 2.0, "slow leftward motion: 40 px/s, direction does not matter");
}

void testStationaryJitter() {
    cv::RNG rng(5);
    PathHistory path;
    for (double t = 0; t <= 1.0; t += FRAME_SECONDS) {
        // Detection noise of about 1.5 px around a fixed spot.
        path.add(at(t), {300 + static_cast<float>(rng.gaussian(1.5)), 200 + static_cast<float>(rng.gaussian(1.5))});
    }
    std::optional<double> speed = path.speed(at(1.0));
    check(speed.has_value() && *speed == 0.0, "still object with 1.5 px jitter reads 0 px/s");
}

void testNotEnoughData() {
    PathHistory empty;
    check(!empty.speed(at(0)).has_value(), "no points: no speed");

    PathHistory two;
    two.add(at(0), {0, 0});
    two.add(at(FRAME_SECONDS), {5, 0});
    check(!two.speed(at(FRAME_SECONDS)).has_value(), "two points: no speed yet");

    PathHistory path = straightPath({100, 0}, 1.0);
    check(!path.speed(at(1.0 + 0.5)).has_value(), "newest point half a second old: speed unknown (object lost)");
}

void testBreakPath() {
    // Moves fast, is lost, and reappears far away standing still.
    PathHistory path = straightPath({300, 0}, 1.0);
    path.breakPath();
    for (double t = 1.5; t <= 2.0; t += FRAME_SECONDS) path.add(at(t), {900, 400});

    std::optional<double> speed = path.speed(at(2.0));
    check(speed.has_value() && *speed == 0.0, "after a break, the jump across the gap does not count as speed");

    PathHistory fresh = straightPath({300, 0}, 1.0);
    fresh.breakPath();
    fresh.add(at(1.5), {900, 400});
    fresh.add(at(1.5 + FRAME_SECONDS), {900, 400});
    check(!fresh.speed(at(1.5 + FRAME_SECONDS)).has_value(), "right after a break: no speed until enough new points");
}

void testTrail() {
    PathHistory path = straightPath({50, 0}, 5.0);  // 5 s of path
    std::vector<TrailPoint> trail = path.trail(at(5.0));
    check(!trail.empty(), "trail has points");
    check(std::all_of(trail.begin(), trail.end(), [](const TrailPoint& p) { return p.ageSeconds <= TRAIL_SECONDS + 1e-6; }),
          "trail only reaches back TRAIL_SECONDS");
    check(std::is_sorted(trail.begin(), trail.end(), [](const TrailPoint& a, const TrailPoint& b) { return a.ageSeconds > b.ageSeconds; }),
          "trail is oldest first");
    check(std::abs(trail.back().ageSeconds) < 1e-6, "newest trail point has age 0");

    path.breakPath();
    path.add(at(5.2), {0, 0});
    trail = path.trail(at(5.2));
    check(trail.back().segment != trail.front().segment, "points after a break are in a new segment");
}

void testDottedPath() {
    // A straight 300 px line, once with many samples and once with just two.
    auto line = [](int samples) {
        std::vector<TrailPoint> trail;
        for (int i = 0; i < samples; i++) {
            double f = static_cast<double>(i) / (samples - 1);
            trail.push_back({{static_cast<float>(300 * f), 50}, 1.0 * (1 - f), 0});
        }
        return trail;
    };

    for (int samples : {2, 31}) {
        std::vector<Dot> dots = dottedPath(line(samples), 12.0);
        bool evenlySpaced = true;
        for (size_t i = 1; i < dots.size(); i++) {
            if (std::abs(cv::norm(dots[i].position - dots[i - 1].position) - 12.0) > 0.01) evenlySpaced = false;
        }
        check(dots.size() == 26 && evenlySpaced,
              "300 px line from " + std::to_string(samples) + " samples: 26 dots, 12 px apart");
    }

    std::vector<Dot> dots = dottedPath(line(31), 12.0);
    check(dots.front().alpha < dots.back().alpha, "older dots are fainter than newer ones");

    // Two segments: no dots between them, and each starts with a dot.
    std::vector<TrailPoint> broken = {{{0, 0}, 2.0, 0}, {{100, 0}, 1.9, 0}, {{500, 300}, 0.5, 1}, {{600, 300}, 0.4, 1}};
    dots = dottedPath(broken, 12.0);
    bool anyBetween = std::any_of(dots.begin(), dots.end(), [](const Dot& d) { return d.position.x > 100.5f && d.position.x < 499.5f; });
    check(!anyBetween, "no dots across a break");
    check(std::any_of(dots.begin(), dots.end(), [](const Dot& d) { return d.position == cv::Point2f(500, 300); }),
          "a new segment starts with a dot");

    check(dottedPath({}, 12.0).empty() && dottedPath({{{1, 1}, 0, 0}}, 12.0).empty(), "empty or single-point trail: no dots");
}

void testDrawing() {
    cv::Mat image = cv::Mat::zeros(240, 320, CV_8UC3);
    std::vector<TrailPoint> trail = {{{-30, -30}, 1.0, 0}, {{5, 5}, 0.8, 0}, {{160, 120}, 0.4, 0}, {{400, 300}, 0.0, 0}};
    drawDottedTrail(image, trail, cv::Scalar(0, 140, 255));
    check(cv::countNonZero(image.reshape(1)) > 0, "trail draws something, including near and past the image edge");

    cv::Mat labelled = cv::Mat::zeros(240, 320, CV_8UC3);
    drawSpeedLabel(labelled, 123.4);
    cv::Mat withUnknown = cv::Mat::zeros(240, 320, CV_8UC3);
    drawSpeedLabel(withUnknown, std::nullopt);
    check(cv::countNonZero(labelled.reshape(1)) > 0 && cv::countNonZero(withUnknown.reshape(1)) > 0,
          "speed label draws for a known and an unknown speed");
    cv::Mat difference;
    cv::absdiff(labelled, withUnknown, difference);
    check(cv::countNonZero(difference.reshape(1)) > 0, "known and unknown speed look different");
}

}  // namespace

int main() {
    testConstantVelocity();
    testStationaryJitter();
    testNotEnoughData();
    testBreakPath();
    testTrail();
    testDottedPath();
    testDrawing();

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return failures == 0 ? 0 : 1;
}
