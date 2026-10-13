// Unit tests for PathHistory (trail and speeds) and the trail drawing.
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
PathHistory straightPath(cv::Point2f velocity, double seconds, std::optional<double> cmPerPixel = std::nullopt,
                         cv::Point2f start = {100, 100}) {
    PathHistory path;
    for (double t = 0; t <= seconds + 1e-9; t += FRAME_SECONDS) {
        path.add(at(t), start + velocity * static_cast<float>(t), cmPerPixel);
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
    check(!path.speed(at(1.0 + 0.5)).has_value(), "newest point half a second old: speed unknown");
}

void testLostFrames() {
    // Moves fast, is lost for half a second, and reappears far away standing still.
    PathHistory path = straightPath({300, 0}, 1.0);
    check(path.speed(at(1.0)).has_value(), "moving object has a speed");

    for (double t = 1.0 + FRAME_SECONDS; t <= 1.5; t += FRAME_SECONDS) path.markLost(at(t));
    check(!path.speed(at(1.5)).has_value(), "while the object is lost there is no speed");

    double t = 1.5 + FRAME_SECONDS;
    path.add(at(t), {900, 400});
    path.add(at(t += FRAME_SECONDS), {900, 400});
    check(!path.speed(at(t)).has_value(), "right after it reappears: no speed until there are enough new points");

    for (t += FRAME_SECONDS; t <= 2.2; t += FRAME_SECONDS) path.add(at(t), {900, 400});
    std::optional<double> speed = path.speed(at(t));
    check(speed.has_value() && *speed == 0.0, "after a gap, the jump across it does not count as speed");

    // A single missed frame in the middle of steady motion must not poison the speed for long.
    PathHistory dropout;
    double x = 0;
    for (double u = 0; u <= 1.5; u += FRAME_SECONDS, x += 10) {
        if (std::abs(u - 0.5) < FRAME_SECONDS / 2) dropout.markLost(at(u));
        else dropout.add(at(u), {static_cast<float>(x), 50});
    }
    std::optional<double> afterDropout = dropout.speed(at(1.5));
    check(afterDropout.has_value() && std::abs(*afterDropout - 300.0) < 15.0, "a single missed frame: speed recovers to 300 px/s");
}

void testTrail() {
    // 100 frames, more than the trail holds.
    PathHistory path = straightPath({60, 0}, 100 * FRAME_SECONDS);
    std::vector<std::optional<cv::Point2f>> trail = path.trail();
    check(trail.size() == TRAIL_LENGTH, "the trail keeps exactly TRAIL_LENGTH frames");
    check(trail.front().has_value() && trail.back().has_value() && trail.front()->x > trail.back()->x,
          "the trail is newest first");

    // Lost frames take a place in the trail, as gaps.
    PathHistory gappy;
    gappy.add(at(0), {10, 10});
    gappy.markLost(at(FRAME_SECONDS));
    gappy.add(at(2 * FRAME_SECONDS), {30, 10});
    trail = gappy.trail();
    check(trail.size() == 3 && trail[0] && !trail[1] && trail[2], "a lost frame is a gap in the trail");

    // Once the object has been lost for TRAIL_LENGTH frames, its tail has faded away.
    PathHistory fading = straightPath({60, 0}, 1.0);
    for (size_t i = 1; i <= TRAIL_LENGTH; i++) fading.markLost(at(1.0 + i * FRAME_SECONDS));
    trail = fading.trail();
    check(std::none_of(trail.begin(), trail.end(), [](const auto& p) { return p.has_value(); }),
          "after TRAIL_LENGTH lost frames, no tail is left");
}

void testTrailThickness() {
    bool matchesFormula = true, neverBelowOne = true, nonIncreasing = true;
    for (size_t i = 1; i < TRAIL_LENGTH; i++) {
        int expected = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(TRAIL_LENGTH) / (i + 1.0)) * 2.5));
        matchesFormula = matchesFormula && trailThickness(i) == expected;
        neverBelowOne = neverBelowOne && trailThickness(i) >= 1;
        if (i > 1) nonIncreasing = nonIncreasing && trailThickness(i) <= trailThickness(i - 1);
    }
    check(matchesFormula, "thickness = sqrt(TRAIL_LENGTH / (i + 1)) * 2.5");
    check(neverBelowOne, "the line is always at least 1 px thick");
    check(nonIncreasing && trailThickness(1) > trailThickness(TRAIL_LENGTH - 1), "the trail tapers: thick at the object, thin at the tail");
}

void testMetricSpeed() {
    cv::Point2f velocity(180, 135);  // 225 px/s
    PathHistory path = straightPath(velocity, 1.0, 0.02);
    std::optional<double> cm = path.metricSpeed(at(1.0));
    check(cm.has_value() && std::abs(*cm - 4.5) < 4.5 * 0.02, "225 px/s at 0.02 cm per pixel: 4.5 cm/s within 2%");

    PathHistory noScale = straightPath(velocity, 1.0);
    check(noScale.speed(at(1.0)).has_value() && !noScale.metricSpeed(at(1.0)).has_value(),
          "without a pixel size there is a px/s speed but no cm/s speed");

    PathHistory still;
    for (double t = 0; t <= 1.0; t += FRAME_SECONDS) still.add(at(t), {300, 200}, 0.02);
    std::optional<double> stillCm = still.metricSpeed(at(1.0));
    check(stillCm.has_value() && *stillCm == 0.0, "a still object reads 0 cm/s");

    // One badly measured size must not swing the result: the median is used.
    PathHistory outlier;
    int frame = 0;
    for (double t = 0; t <= 1.0; t += FRAME_SECONDS, frame++) {
        outlier.add(at(t), cv::Point2f(100, 100) + velocity * static_cast<float>(t), frame == 25 ? 0.2 : 0.02);
    }
    std::optional<double> outlierCm = outlier.metricSpeed(at(1.0));
    check(outlierCm.has_value() && std::abs(*outlierCm - 4.5) < 4.5 * 0.02, "one wrong pixel size is ignored");

    // Closer to the camera means a smaller real size per pixel; the speed uses the size in the window.
    PathHistory approaching;
    for (double t = 0; t <= 1.0; t += FRAME_SECONDS) {
        approaching.add(at(t), cv::Point2f(100, 100) + velocity * static_cast<float>(t), 0.02 - 0.004 * t);
    }
    std::optional<double> approachCm = approaching.metricSpeed(at(1.0));
    check(approachCm.has_value() && *approachCm > 225 * 0.016 && *approachCm < 225 * 0.0172,
          "a pixel size that changes during the move: the speed uses the recent sizes");

    for (double t = 1.0 + FRAME_SECONDS; t <= 1.5; t += FRAME_SECONDS) path.markLost(at(t));
    check(!path.metricSpeed(at(1.5)).has_value(), "no cm/s speed while the object is lost");
}

void testDrawing() {
    cv::Mat image = cv::Mat::zeros(240, 320, CV_8UC3);
    std::vector<std::optional<cv::Point2f>> trail = {cv::Point2f(400, 300), cv::Point2f(160, 120), cv::Point2f(5, 5),
                                                     cv::Point2f(-30, -30)};
    drawTrail(image, trail, cv::Scalar(0, 0, 255));
    check(cv::countNonZero(image.reshape(1)) > 0, "the trail draws something, including near and past the image edge");

    cv::Mat gap = cv::Mat::zeros(240, 320, CV_8UC3);
    drawTrail(gap, {cv::Point2f(100, 100), std::nullopt, cv::Point2f(200, 100)}, cv::Scalar(0, 0, 255));
    check(cv::countNonZero(gap.reshape(1)) == 0, "no line is drawn across a lost frame");

    cv::Mat none = cv::Mat::zeros(240, 320, CV_8UC3);
    drawTrail(none, {}, cv::Scalar(0, 0, 255));
    drawTrail(none, {cv::Point2f(10, 10)}, cv::Scalar(0, 0, 255));
    check(cv::countNonZero(none.reshape(1)) == 0, "an empty or single-point trail draws nothing");

    cv::Mat labelled = cv::Mat::zeros(240, 320, CV_8UC3);
    drawSpeedLabel(labelled, 123.4);
    cv::Mat withUnknown = cv::Mat::zeros(240, 320, CV_8UC3);
    drawSpeedLabel(withUnknown, std::nullopt);
    cv::Mat withMetric = cv::Mat::zeros(240, 320, CV_8UC3);
    drawSpeedLabel(withMetric, 123.4, 2.5);
    check(cv::countNonZero(labelled.reshape(1)) > 0 && cv::countNonZero(withUnknown.reshape(1)) > 0 &&
              cv::countNonZero(withMetric.reshape(1)) > 0,
          "the speed label draws for a known, an unknown and a metric speed");

    auto differ = [](const cv::Mat& a, const cv::Mat& b) {
        cv::Mat difference;
        cv::absdiff(a, b, difference);
        return cv::countNonZero(difference.reshape(1)) > 0;
    };
    check(differ(labelled, withUnknown) && differ(labelled, withMetric), "the three speed labels look different");
}

}  // namespace

int main() {
    testConstantVelocity();
    testStationaryJitter();
    testNotEnoughData();
    testLostFrames();
    testTrail();
    testTrailThickness();
    testMetricSpeed();
    testDrawing();

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return failures == 0 ? 0 : 1;
}
