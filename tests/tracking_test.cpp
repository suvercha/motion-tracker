// End-to-end offline test of following a moving object: no camera needed.
//
// Renders a short synthetic video of the object moving at a known speed across a
// cluttered background, with motion blur like a real camera's, and runs it through
// the same pipeline as find_object: the detector plus tracker (ObjectTracker), and
// PathHistory for the speed. Checks the positions and the speed against the truth,
// and reports how often the detector alone would have found the object compared
// with the detector plus tracker.
//
// Usage: tracking_test [orb|sift] [photo] [output_dir] [models_dir] [nano|vit|mil|kcf|csrt]
//   defaults: orb, object.jpg, test_output, the project's models/ folder, nano
// The nano and vit trackers need their model files in the models folder: run
// scripts/download_models.sh. Whatever tracker is chosen, a quick check first makes sure that
// every tracker type starts and follows the object.
// Writes a few annotated frames per scenario to output_dir.

#include "motion/object_finder.hpp"
#include "motion/object_tracker.hpp"
#include "motion/overlay.hpp"
#include "motion/path_history.hpp"
#include "motion/segmentation.hpp"
#include "test_scene.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace motion;
using test_scene::FRAME_SIZE;

namespace {

constexpr double FPS = 30;
constexpr double FRAME_SECONDS = 1.0 / FPS;
// A moving object smears across this many pixels per rendered sub-frame, at most.
constexpr double BLUR_STEP_PX = 2.0;
constexpr int MAX_BLUR_SAMPLES = 12;
constexpr double NOISE_SD = 4;
// The object sits still this long, so it is found while sharp (as when someone
// holds it up to the camera) before it starts moving.
constexpr double STILL_SECONDS = 0.6;
// The speed is fitted over a window of path; frames are scored from this long after the object starts moving.
constexpr double SCORE_AFTER_MOVE_SECONDS = 0.6;
constexpr int RNG_SEED = 21;
// The tin's real width, which turns pixels into centimetres.
constexpr double OBJECT_WIDTH_CM = 4.5;
// How long the tracker is expected to carry the object after the detector loses it
// (matches MAX_TRACK_ONLY_SECONDS in object_tracker.cpp).
constexpr double EXPECTED_TRACKER_SECONDS = 1.0;
// How far off the speed may be while only the tracker is following a blurred object.
// TrackerNano reads the speed 5-11% low at 12 px of blur in repeated runs (tracking
// trackers damp the movement they report), so this leaves room for run-to-run variation.
constexpr double MAX_BLURRED_SPEED_ERROR = 0.15;

// What the object does in a scenario.
struct Scenario {
    const char* id;
    const char* description;
    cv::Point2f start;     // the object's center in the frame at t = 0
    cv::Point2f velocity;  // pixels per second
    double exposureSeconds = 0;  // motion blur: the object smears by speed * exposure
    double vanishAtSeconds = -1; // the object leaves the scene at this time (negative: never)
    double seconds = 3.0;
    bool required = true;
};

const std::vector<Scenario> SCENARIOS = {
    {.id = "sharp",      .description = "250 px/s, no blur",             .start = {300, 250}, .velocity = {240, 70}},
    {.id = "blur12",     .description = "250 px/s, 12 px blur",          .start = {300, 250}, .velocity = {240, 70}, .exposureSeconds = 0.05},
    {.id = "blur25",     .description = "250 px/s, 25 px blur",          .start = {300, 250}, .velocity = {240, 70}, .exposureSeconds = 0.10, .required = false},
    {.id = "blur50",     .description = "250 px/s, 50 px blur",          .start = {300, 250}, .velocity = {240, 70}, .exposureSeconds = 0.20, .required = false},
    {.id = "stationary", .description = "not moving",                    .start = {640, 360}, .velocity = {0, 0}},
    {.id = "vanish",     .description = "250 px/s, leaves scene at 1.6 s", .start = {300, 250}, .velocity = {240, 70}, .vanishAtSeconds = 1.6, .seconds = 4.0},
};

// The photo's object, ready to be drawn into frames.
struct Subject {
    cv::Mat photo;
    cv::Mat mask;
    cv::Point2f centroid;
    double radius;    // farthest distance from the centroid to the object's edge, in photo pixels
    double diameter;  // the object's widest dimension, in photo pixels
    double scale;     // object size in the frame relative to the photo
};

Subject makeSubject(const cv::Mat& photo, const cv::Mat& mask, double scale) {
    cv::Moments m = cv::moments(mask, /*binaryImage=*/true);
    cv::Point2f centroid(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
    double radius = 0;
    std::vector<cv::Point> outline = largestContour(mask).value();
    std::vector<cv::Point2f> outlinePoints;
    for (const cv::Point& p : outline) {
        radius = std::max(radius, cv::norm(cv::Point2f(p) - centroid));
        outlinePoints.emplace_back(p);
    }
    return {photo, mask, centroid, radius, maxDiameter(outlinePoints), scale};
}

Clock::time_point at(double seconds) {
    return Clock::time_point{} + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

cv::Point2f positionAt(const Scenario& sc, double t) {
    return sc.start + sc.velocity * static_cast<float>(std::max(0.0, t - STILL_SECONDS));
}

// The frame at time `t`. The object turns slowly as it moves, and is blurred
// along its path by averaging copies of it drawn across the exposure.
cv::Mat renderFrame(const Subject& subject, const cv::Mat& background, const Scenario& sc, double t, cv::RNG& rng) {
    cv::Mat frame = background.clone();

    bool present = sc.vanishAtSeconds < 0 || t < sc.vanishAtSeconds;
    double speed = std::hypot(sc.velocity.x, sc.velocity.y);
    double blurLength = speed * sc.exposureSeconds;
    int samples = std::clamp(static_cast<int>(std::ceil(blurLength / BLUR_STEP_PX)), 1, MAX_BLUR_SAMPLES);

    // Only the area the object covers during the exposure needs redrawing.
    double half = subject.radius * subject.scale + 4;
    auto squareAround = [&](double time) {
        cv::Point2f c = positionAt(sc, time);
        return cv::Rect(cvFloor(c.x - half), cvFloor(c.y - half), cvCeil(2 * half), cvCeil(2 * half));
    };
    cv::Rect roi = (squareAround(t - sc.exposureSeconds / 2) | squareAround(t + sc.exposureSeconds / 2))
                   & cv::Rect(0, 0, frame.cols, frame.rows);

    if (present && !roi.empty()) {
        cv::Mat objectSum = cv::Mat::zeros(roi.size(), CV_32FC3);
        cv::Mat coverageSum = cv::Mat::zeros(roi.size(), CV_32FC1);
        for (int k = 0; k < samples; k++) {
            double time = t + sc.exposureSeconds * ((k + 0.5) / samples - 0.5);
            cv::Point2f center = positionAt(sc, time);
            double angle = 20 + 30 * time;

            // Turn about the object's centroid, then move that centroid to `center`.
            cv::Mat M = cv::getRotationMatrix2D(subject.centroid, angle, subject.scale);
            M.at<double>(0, 2) += center.x - subject.centroid.x - roi.x;
            M.at<double>(1, 2) += center.y - subject.centroid.y - roi.y;

            cv::Mat warped, warpedMask, warpedF, coverage, coverage3;
            cv::warpAffine(subject.photo, warped, M, roi.size());
            cv::warpAffine(subject.mask, warpedMask, M, roi.size());
            warped.convertTo(warpedF, CV_32F);
            warpedMask.convertTo(coverage, CV_32F, 1.0 / 255.0);
            cv::merge(std::vector<cv::Mat>{coverage, coverage, coverage}, coverage3);
            cv::multiply(warpedF, coverage3, warpedF);
            objectSum += warpedF;
            coverageSum += coverage;
        }
        cv::Mat objectAverage = objectSum / samples;
        cv::Mat coverage = coverageSum / samples, coverage3, background3F;
        cv::merge(std::vector<cv::Mat>{coverage, coverage, coverage}, coverage3);
        background(roi).convertTo(background3F, CV_32F);
        cv::multiply(background3F, cv::Scalar::all(1.0) - coverage3, background3F);
        cv::Mat combined = objectAverage + background3F, blended;
        combined.convertTo(blended, CV_8U);
        blended.copyTo(frame(roi));
    }

    cv::Mat noise(frame.size(), CV_16SC3), wide;
    rng.fill(noise, cv::RNG::NORMAL, 0, NOISE_SD);
    frame.convertTo(wide, CV_16SC3);
    wide += noise;
    wide.convertTo(frame, CV_8UC3);
    return test_scene::jpegRoundTrip(frame);
}

struct Metrics {
    int frames = 0;
    int detected = 0;     // the detector found the object
    int trackerOnly = 0;  // the detector missed; the tracker followed it
    double detectedCenterErrorPx = 0;
    double trackerCenterErrorPx = 0;
    int scored = 0;                 // frames after the warm-up, while the object is in the scene
    int speedAvailable = 0;
    std::vector<double> speedErrors;      // relative to the true speed, moving scenarios
    std::vector<double> signedSpeedErrors;  // (speed - true) / true: negative means the speed reads too low
    std::vector<double> metricSpeedErrors;  // the same for the speed in cm/s, which also tests the size estimate
    int metricSpeedReadsZero = 0;           // stationary scenario
    int speedReadsZero = 0;               // stationary scenario
    double lastTrackAfterVanish = -1;     // vanish scenario: latest time with a track after the object left
    int speedAfterVanish = 0;             // vanish scenario: frames showing a speed long after the object left
};

double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Frames to save for a visual check.
bool shouldSave(double t) {
    for (double target : {0.5, 1.2, 2.1, 2.7}) {
        if (std::abs(t - target) < FRAME_SECONDS / 2) return true;
    }
    return false;
}

Metrics runScenario(const Scenario& sc, const Subject& subject, const cv::Mat& background, const ObjectFinder& finder,
                    const std::string& modelsDir, TrackerType trackerType, const std::string& outDir) {
    Metrics metrics;
    ObjectTracker tracker = ObjectTracker::create(finder, modelsDir, trackerType).value();
    PathHistory path;
    cv::RNG rng(RNG_SEED);
    double trueSpeed = std::hypot(sc.velocity.x, sc.velocity.y);
    double trueCmPerPixel = OBJECT_WIDTH_CM / (subject.diameter * subject.scale);
    double trueMetricSpeed = trueSpeed * trueCmPerPixel;

    int totalFrames = static_cast<int>(sc.seconds * FPS);
    for (int i = 0; i < totalFrames; i++) {
        double t = i * FRAME_SECONDS;
        Clock::time_point now = at(t);
        cv::Mat frame = renderFrame(subject, background, sc, t, rng);
        bool present = sc.vanishAtSeconds < 0 || t < sc.vanishAtSeconds;

        std::optional<Track> track = tracker.update(frame, now);
        if (track) {
            path.add(now, track->center, track->cmPerPixel(OBJECT_WIDTH_CM));
            double error = cv::norm(track->center - positionAt(sc, t));
            if (track->detected) {
                metrics.detected++;
                metrics.detectedCenterErrorPx += error;
            } else {
                metrics.trackerOnly++;
                metrics.trackerCenterErrorPx += error;
            }
            if (!present) metrics.lastTrackAfterVanish = t;
        } else {
            path.markLost(now);
        }
        metrics.frames++;

        std::optional<double> speed = path.speed(now);
        std::optional<double> metricSpeed = path.metricSpeed(now);
        if (present && t >= STILL_SECONDS + SCORE_AFTER_MOVE_SECONDS) {
            metrics.scored++;
            if (speed) {
                metrics.speedAvailable++;
                if (trueSpeed > 0) {
                    metrics.speedErrors.push_back(std::abs(*speed - trueSpeed) / trueSpeed);
                    metrics.signedSpeedErrors.push_back((*speed - trueSpeed) / trueSpeed);
                }
                if (*speed == 0.0) metrics.speedReadsZero++;
            }
            if (metricSpeed) {
                if (trueMetricSpeed > 0) metrics.metricSpeedErrors.push_back(std::abs(*metricSpeed - trueMetricSpeed) / trueMetricSpeed);
                if (*metricSpeed == 0.0) metrics.metricSpeedReadsZero++;
            }
        }
        // Long enough after the object left that no speed should still be showing.
        if (!present && t > sc.vanishAtSeconds + 1.6 && speed) metrics.speedAfterVanish++;

        if (shouldSave(t)) {
            cv::Mat annotated = frame.clone();
            drawTrail(annotated, path.trail(), cv::Scalar(0, 0, 255));
            drawSpeedLabel(annotated, speed, metricSpeed);
            if (track) {
                cv::rectangle(annotated, track->box, track->detected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255), 2);
            }
            cv::circle(annotated, positionAt(sc, t), 5, cv::Scalar(255, 0, 0), cv::FILLED);
            cv::imwrite(outDir + "/tracking_" + sc.id + "_" + std::to_string(cvRound(t * 1000)) + "ms.jpg", annotated);
        }
    }
    if (metrics.detected) metrics.detectedCenterErrorPx /= metrics.detected;
    if (metrics.trackerOnly) metrics.trackerCenterErrorPx /= metrics.trackerOnly;
    return metrics;
}

// How long the detector and a tracker each take per frame, on sharp frames of the
// object moving. Printed for information; this is what decides how often the
// detector needs to run.
void reportCosts(const Subject& subject, const cv::Mat& background, const ObjectFinder& finder,
                 const TrackerFactory& makeTracker, TrackerType trackerType) {
    const Scenario& sc = SCENARIOS.front();
    cv::RNG rng(RNG_SEED);
    constexpr int FRAMES = 20;
    constexpr double START = STILL_SECONDS + 0.2;

    std::vector<cv::Mat> frames;
    for (int i = 0; i < FRAMES; i++) frames.push_back(renderFrame(subject, background, sc, START + i * FRAME_SECONDS, rng));

    double detectMs = 0;
    for (const cv::Mat& frame : frames) {
        auto begin = std::chrono::steady_clock::now();
        (void)finder.find(frame);
        detectMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }

    std::optional<Detection> first = finder.find(frames.front());
    if (!first) return;
    cv::Ptr<cv::Tracker> tracker = makeTracker();
    auto initBegin = std::chrono::steady_clock::now();
    tracker->init(frames.front(), first->box);
    double initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initBegin).count();
    double trackMs = 0;
    for (size_t i = 1; i < frames.size(); i++) {
        cv::Rect box;
        auto begin = std::chrono::steady_clock::now();
        (void)tracker->update(frames[i], box);
        trackMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }

    std::cout << std::fixed << std::setprecision(1) << "cost per frame: detector " << detectMs / frames.size()
              << " ms, " << toString(trackerType) << " tracker " << trackMs / (frames.size() - 1) << " ms (plus " << initMs << " ms to start it)\n\n";
}

// Checks that every tracker type can be created, started and used to follow the object.
// This is a working check, not a comparison of the trackers. The object is found while it
// sits still, then moves and blurs, so the detector may miss and the tracker has to carry it.
// Returns the number of failures.
int checkEveryTrackerType(const Subject& subject, const cv::Mat& background, const ObjectFinder& finder,
                          const std::string& modelsDir) {
    constexpr double MAX_ERROR_PX = 80;  // the tin is about 220 px across
    const Scenario& sc = SCENARIOS[2];
    cv::RNG rng(RNG_SEED);
    const std::vector<double> times = {0.4, 0.5, 0.7};
    std::vector<cv::Mat> frames;
    for (double t : times) frames.push_back(renderFrame(subject, background, sc, t, rng));

    int failures = 0;
    std::cout << "every tracker type:\n";
    for (TrackerType type : ALL_TRACKER_TYPES) {
        std::optional<ObjectTracker> tracker = ObjectTracker::create(finder, modelsDir, type);
        if (!tracker) {
            std::cout << "  " << std::left << std::setw(5) << toString(type) << " FAIL: could not be created\n";
            failures++;
            continue;
        }
        std::optional<Track> last;
        for (size_t i = 0; i < frames.size(); i++) last = tracker->update(frames[i], at(times[i]));

        double error = last ? cv::norm(last->center - positionAt(sc, times.back())) : -1;
        bool ok = last && error < MAX_ERROR_PX;
        std::cout << "  " << std::left << std::setw(5) << toString(type) << (ok ? " ok" : " FAIL");
        if (last) {
            std::cout << ": " << (last->detected ? "found by the detector" : "followed by the tracker") << ", "
                      << std::fixed << std::setprecision(0) << error << " px from the true position";
        } else {
            std::cout << ": lost the object";
        }
        std::cout << '\n';
        if (!ok) failures++;
    }
    std::cout << '\n';
    return failures;
}

// Prints the scenario's row and returns the number of failures.
int report(const Scenario& sc, const Metrics& m) {
    bool vanishing = sc.vanishAtSeconds >= 0;
    bool stationary = sc.velocity == cv::Point2f(0, 0);
    double frames = std::max(1, m.frames);
    double coverageWithTracker = (m.detected + m.trackerOnly) / frames;
    double coverageDetectorOnly = m.detected / frames;
    double scored = std::max(1, m.scored);

    std::vector<std::string> problems;
    if (vanishing) {
        // The tracker may follow for a short while after the object leaves, but not for long.
        if (m.lastTrackAfterVanish > sc.vanishAtSeconds + 1.0 + 2 * FRAME_SECONDS) {
            problems.push_back("still tracking " + std::to_string(m.lastTrackAfterVanish - sc.vanishAtSeconds) + " s after the object left");
        }
        if (m.speedAfterVanish > 0) problems.push_back("speed still shown long after the object left");
    } else if (sc.exposureSeconds > 0) {
        // Motion blur defeats the detector once the object moves, so only the tracker can follow
        // it, and only for EXPECTED_TRACKER_SECONDS. Check that it does, and that the speed it
        // gives is accurate. Following the whole blurred movement is a known limit
        // (docs/tracking-design.md), so it is not required here.
        double expectedCoverage = (STILL_SECONDS + EXPECTED_TRACKER_SECONDS - 0.15) / sc.seconds;
        if (coverageWithTracker < expectedCoverage) problems.push_back("tracker did not carry the object for about a second");
        if (m.speedAvailable < 10) problems.push_back("too few frames with a speed reading");
        if (median(m.speedErrors) > MAX_BLURRED_SPEED_ERROR) problems.push_back("median speed error over 15%");
        if (median(m.metricSpeedErrors) > MAX_BLURRED_SPEED_ERROR) problems.push_back("median cm/s error over 15%");
    } else {
        if (coverageWithTracker < 0.95) problems.push_back("object followed in under 95% of frames");
        if (m.speedAvailable < 0.9 * scored) problems.push_back("speed available in under 90% of frames");
        if (stationary) {
            if (m.speedReadsZero < 0.9 * scored) problems.push_back("still object does not read 0 px/s");
            if (m.metricSpeedReadsZero < 0.9 * scored) problems.push_back("still object does not read 0 cm/s");
        } else {
            if (median(m.speedErrors) > 0.10) problems.push_back("median speed error over 10%");
            if (median(m.metricSpeedErrors) > 0.10) problems.push_back("median cm/s error over 10%");
        }
    }

    std::cout << std::left << std::setw(11) << sc.id << std::setw(33) << sc.description << std::fixed;
    if (vanishing) {
        std::cout << "tracked until +" << std::setprecision(2) << std::max(0.0, m.lastTrackAfterVanish - sc.vanishAtSeconds)
                  << " s after it left";
    } else {
        std::cout << "detector alone " << std::setprecision(0) << std::setw(4) << coverageDetectorOnly * 100 << "%, with tracker "
                  << std::setw(4) << coverageWithTracker * 100 << "%  position err " << std::setprecision(1)
                  << m.detectedCenterErrorPx << " px";
        if (m.trackerOnly) std::cout << " (tracker frames " << m.trackerCenterErrorPx << " px)";
        if (stationary) {
            std::cout << "  reads 0 px/s " << std::setprecision(0) << 100.0 * m.speedReadsZero / scored << "%";
        } else {
            std::cout << "  median speed err " << std::setprecision(1) << median(m.speedErrors) * 100 << "% px/s, "
                      << median(m.metricSpeedErrors) * 100 << "% cm/s (bias " << std::showpos
                      << median(m.signedSpeedErrors) * 100 << std::noshowpos << "%)";
        }
    }
    std::cout << "  ";
    if (problems.empty()) {
        std::cout << "ok\n";
        return 0;
    }
    std::cout << (sc.required ? "FAIL" : "(informational)") << '\n';
    for (const std::string& problem : problems) std::cout << "              - " << problem << '\n';
    return sc.required ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string detectorName = (argc > 1) ? argv[1] : "orb";
    std::string photoPath = (argc > 2) ? argv[2] : "object.jpg";
    std::string outDir = (argc > 3) ? argv[3] : "test_output";
    std::string modelsDir = (argc > 4) ? argv[4] : DEFAULT_MODELS_DIR;
    std::string trackerName = (argc > 5) ? argv[5] : std::string(toString(DEFAULT_TRACKER));

    std::optional<DetectorType> detector = parseDetectorType(detectorName);
    std::optional<TrackerType> trackerType = parseTrackerType(trackerName);
    if (!detector || !trackerType) {
        std::cerr << "Usage: tracking_test [orb|sift] [photo] [output_dir] [models_dir] [nano|vit|mil|kcf|csrt]"
                  << std::endl;
        return 2;
    }
    cv::Mat photo = cv::imread(photoPath);
    if (photo.empty()) {
        std::cerr << "Error: could not read photo '" << photoPath << "'." << std::endl;
        return 2;
    }
    std::filesystem::create_directories(outDir);

    cv::Mat mask = isolateObject(photo, centeredRect(photo.size(), PHOTO_SEED_FRACTION));
    std::optional<ObjectFinder> finder = ObjectFinder::fromPhoto(photo, *detector);
    if (mask.empty() || !finder) {
        std::cout << "FAIL: could not learn the object\n";
        return 1;
    }

    constexpr double OBJECT_SCALE = 0.25;
    Subject subject = makeSubject(photo, mask, OBJECT_SCALE);
    cv::RNG backgroundRng(RNG_SEED);
    cv::Mat background = test_scene::makeBackground(photo, backgroundRng);

    std::cout << "=== " << toString(*detector) << ": following a moving object (" << FPS << " fps, "
              << "object about " << cvRound(2 * subject.radius * OBJECT_SCALE) << " px across) ===\n";
    // The trackers need their model files; say so clearly if they are missing.
    std::optional<TrackerFactory> factory = createTrackerFactory(*trackerType, modelsDir);
    if (!factory) return 2;
    int failures = 0;
    if (ObjectTracker::create(*finder, "/nonexistent-models-dir", TrackerType::Nano)) {
        std::cout << "FAIL: ObjectTracker::create accepted a models folder that does not exist\n";
        failures++;
    }

    failures += checkEveryTrackerType(subject, background, *finder, modelsDir);
    std::cout << "following with the " << toString(*trackerType) << " tracker:\n";
    reportCosts(subject, background, *finder, *factory, *trackerType);
    for (const Scenario& sc : SCENARIOS) {
        failures += report(sc, runScenario(sc, subject, background, *finder, modelsDir, *trackerType, outDir));
    }

    std::cout << "\nFrames written to '" << outDir << "/' (tracking_<scenario>_<time>.jpg: green box = detector, "
              << "yellow = tracker, blue dot = true center).\n"
              << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return failures == 0 ? 0 : 1;
}
