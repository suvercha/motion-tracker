// Offline test for ObjectFinder: no camera needed.
//
// Learns the object from a photo, pastes the isolated object onto a cluttered
// background under different conditions (rotation, perspective tilt, darkness,
// noise, blur, small size), and checks that the object is found with a tight
// box. Also checks that object-free frames produce no false detections.
//
// Usage: find_object_test [orb|sift|both] [photo] [output_dir]
//   defaults: both, object.jpg, test_output
// Writes the isolated cutout and one annotated example per case to output_dir
// (green = detected box, blue = true box).

#include "motion/object_finder.hpp"
#include "motion/segmentation.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace motion;

namespace {

constexpr int TRIALS = 6;
constexpr int REQUIRED_HITS = 5;   // out of TRIALS, for required cases
constexpr double MIN_IOU = 0.5;    // detected box vs true box, to count as a hit
constexpr int FALSE_POSITIVE_FRAMES = 20;
constexpr int RNG_SEED = 11;       // fixed: same scenes every run

const cv::Size FRAME_SIZE(1280, 720);
constexpr int JPEG_QUALITY = 70;   // webcam-like compression
constexpr int CLUTTER_BOXES = 25;
constexpr int CLUTTER_TEXTS = 6;
constexpr double TRIAL_ROTATION_STEP = 47;  // degrees added per trial, so trials differ

// How the object appears in a test frame.
struct Conditions {
    double angle = 20;       // degrees
    double scale = 0.45;     // relative to the photo
    double blur = 0;         // Gaussian sigma
    double brightness = 1.0;
    double noise = 0;        // Gaussian standard deviation
    double tilt = 0;         // perspective tilt, 0 = none
};

struct Case {
    const char* id;
    const char* description;
    Conditions conditions;
    bool required;  // failing a required case fails the test
};

// The "informational" cases are known hard: they are reported but don't fail.
const std::vector<Case> CASES = {
    {.id = "baseline", .description = "rotated, 0.45x",             .conditions = {},                   .required = true},
    {.id = "tilt25",   .description = "perspective tilt (mild)",    .conditions = {.tilt = 0.25},       .required = true},
    {.id = "tilt40",   .description = "perspective tilt (strong)",  .conditions = {.tilt = 0.45},       .required = true},
    {.id = "noise",    .description = "sensor noise (sd 12)",       .conditions = {.noise = 12},        .required = true},
    {.id = "blur",     .description = "blur (sigma 3.5)",           .conditions = {.blur = 3.5},        .required = true},
    {.id = "small",    .description = "small in frame (0.15x)",     .conditions = {.scale = 0.15},      .required = true},
    {.id = "dark",     .description = "dark (x0.55 brightness)",    .conditions = {.brightness = 0.55}, .required = false},
    {.id = "combo",    .description = "small+blur+dark+noise+tilt",
     .conditions = {.angle = 70, .scale = 0.25, .blur = 2.0, .brightness = 0.7, .noise = 8, .tilt = 0.25},
     .required = false},
};

// Background made from the photo's own background (its top-left corner, which
// is assumed to be plain background), plus colored boxes and text as clutter.
cv::Mat makeBackground(const cv::Mat& photo, cv::RNG& rng) {
    cv::Mat patch = photo(cv::Rect(0, 0, photo.cols / 6, photo.rows / 5)).clone();
    cv::Mat bg;
    cv::resize(patch, bg, FRAME_SIZE);
    for (int i = 0; i < CLUTTER_BOXES; i++) {
        cv::Rect r(rng.uniform(0, FRAME_SIZE.width - 100), rng.uniform(0, FRAME_SIZE.height - 100),
                   rng.uniform(30, 160), rng.uniform(30, 160));
        cv::rectangle(bg, r, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
                      cv::FILLED);
    }
    for (int i = 0; i < CLUTTER_TEXTS; i++) {
        cv::putText(bg, "HELPS DRY",
                    cv::Point(rng.uniform(0, FRAME_SIZE.width - 300), rng.uniform(40, FRAME_SIZE.height - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
    }
    return bg;
}

struct Scene {
    cv::Mat frame;
    cv::Rect truth;
};

// The homography that rotates and scales the photo to a random spot near the
// middle of the frame, with optional perspective tilt.
cv::Mat placementTransform(const cv::Size& photoSize, const Conditions& c, int trial, cv::RNG& rng) {
    cv::Point2f center(photoSize.width / 2.f, photoSize.height / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(center, c.angle + trial * TRIAL_ROTATION_STEP, c.scale);
    M.at<double>(0, 2) += FRAME_SIZE.width / 2.0 - center.x + rng.uniform(-200, 200);
    M.at<double>(1, 2) += FRAME_SIZE.height / 2.0 - center.y + rng.uniform(-100, 100);
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    M.copyTo(H(cv::Rect(0, 0, 3, 2)));

    if (c.tilt > 0) {
        double sign = (trial % 2) ? 1 : -1;
        cv::Mat P = cv::Mat::eye(3, 3, CV_64F);
        P.at<double>(2, 0) = sign * c.tilt * 0.0006 / c.scale;
        P.at<double>(2, 1) = -sign * c.tilt * 0.0003 / c.scale;
        cv::Mat toCenter = cv::Mat::eye(3, 3, CV_64F);
        toCenter.at<double>(0, 2) = -FRAME_SIZE.width / 2.0;
        toCenter.at<double>(1, 2) = -FRAME_SIZE.height / 2.0;
        cv::Mat fromCenter = cv::Mat::eye(3, 3, CV_64F);
        fromCenter.at<double>(0, 2) = FRAME_SIZE.width / 2.0;
        fromCenter.at<double>(1, 2) = FRAME_SIZE.height / 2.0;
        H = fromCenter * P * toCenter * H;
    }
    return H;
}

Scene makeScene(const cv::Mat& photo, const cv::Mat& mask, const Conditions& c, int trial, cv::RNG& rng) {
    Scene s;
    s.frame = makeBackground(photo, rng);

    cv::Mat H = placementTransform(photo.size(), c, trial, rng);
    cv::Mat warped, warpedMask;
    cv::warpPerspective(photo, warped, H, FRAME_SIZE);
    cv::warpPerspective(mask, warpedMask, H, FRAME_SIZE, cv::INTER_NEAREST);
    warped.convertTo(warped, -1, c.brightness, 0);
    warped.copyTo(s.frame, warpedMask);
    s.truth = cv::boundingRect(warpedMask);

    if (c.blur > 0) cv::GaussianBlur(s.frame, s.frame, cv::Size(0, 0), c.blur);
    if (c.noise > 0) {
        cv::Mat noise(s.frame.size(), CV_16SC3), wide;
        rng.fill(noise, cv::RNG::NORMAL, 0, c.noise);
        s.frame.convertTo(wide, CV_16SC3);
        wide += noise;
        wide.convertTo(s.frame, CV_8UC3);
    }

    std::vector<uchar> jpg;
    cv::imencode(".jpg", s.frame, jpg, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});
    s.frame = cv::imdecode(jpg, cv::IMREAD_COLOR);
    return s;
}

double intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    return static_cast<double>((a & b).area()) / static_cast<double>((a | b).area());
}

// Checks the cutout is a sensible blob that doesn't touch the photo's edge.
// Returns the number of failures.
int checkIsolation(const cv::Mat& photo, const cv::Mat& mask, const std::string& outDir) {
    std::cout << "=== isolation ===\n";
    cv::Rect bounds = cv::boundingRect(mask);
    double fraction = cv::countNonZero(mask) / static_cast<double>(mask.total());
    bool touchesEdge = bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= photo.cols || bounds.br().y >= photo.rows;
    bool sensible = fraction > 0.05 && fraction < 0.8 && !touchesEdge;
    std::cout << "object covers " << std::fixed << std::setprecision(0) << fraction * 100 << "% of the photo, "
              << (touchesEdge ? "touches" : "clear of") << " the photo edge: " << (sensible ? "ok" : "FAIL") << '\n';

    cv::Mat cutout(photo.size(), photo.type(), cv::Scalar(255, 255, 255));
    photo.copyTo(cutout, mask);
    cv::imwrite(outDir + "/cutout.jpg", cutout);
    return sensible ? 0 : 1;
}

struct CaseResult {
    int hits = 0;
    double meanIou = 0;
    double meanMs = 0;
};

CaseResult runCase(const ObjectFinder& finder, const cv::Mat& photo, const cv::Mat& mask, const Case& c,
                   DetectorType type, const std::string& outDir, cv::RNG& rng) {
    CaseResult result;
    double iouSum = 0, msSum = 0;
    for (int t = 0; t < TRIALS; t++) {
        Scene scene = makeScene(photo, mask, c.conditions, t, rng);
        auto start = std::chrono::steady_clock::now();
        std::optional<cv::Rect> found = finder.find(scene.frame);
        msSum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        double iou = found ? intersectionOverUnion(*found, scene.truth) : 0;
        if (iou > MIN_IOU) {
            result.hits++;
            iouSum += iou;
        }
        if (t == 0) {
            cv::Mat annotated = scene.frame.clone();
            cv::rectangle(annotated, scene.truth, cv::Scalar(255, 0, 0), 2);
            if (found) cv::rectangle(annotated, *found, cv::Scalar(0, 255, 0), 2);
            cv::imwrite(outDir + "/" + std::string(toString(type)) + "_" + c.id + ".jpg", annotated);
        }
    }
    result.meanIou = result.hits ? iouSum / result.hits : 0.0;
    result.meanMs = msSum / TRIALS;
    return result;
}

// Returns the number of failures.
int runDetector(DetectorType type, const cv::Mat& photo, const cv::Mat& mask, const std::string& outDir) {
    std::cout << "\n=== " << toString(type) << " ===\n";

    std::optional<ObjectFinder> finder = ObjectFinder::fromPhoto(photo, type);
    if (!finder) {
        std::cout << "FAIL: could not learn the object\n";
        return 1;
    }
    std::cout << "learned " << finder->featureCount() << " features\n\n";
    std::cout << std::left << std::setw(10) << "case" << std::setw(31) << "description" << std::setw(7) << "hits"
              << std::setw(9) << "meanIoU" << std::setw(9) << "meanMs" << "result\n";

    int failures = 0;
    cv::RNG rng(RNG_SEED);
    for (const Case& c : CASES) {
        CaseResult r = runCase(*finder, photo, mask, c, type, outDir, rng);
        bool ok = r.hits >= REQUIRED_HITS;
        if (!ok && c.required) failures++;
        const char* verdict = ok ? "ok" : (c.required ? "FAIL" : "(informational)");
        std::cout << std::left << std::setw(10) << c.id << std::setw(31) << c.description
                  << std::setw(7) << (std::to_string(r.hits) + "/" + std::to_string(TRIALS))
                  << std::fixed << std::setprecision(2) << std::setw(9) << r.meanIou
                  << std::setprecision(0) << std::setw(9) << r.meanMs << verdict << '\n';
    }

    int falsePositives = 0;
    for (int i = 0; i < FALSE_POSITIVE_FRAMES; i++) {
        if (finder->find(makeBackground(photo, rng))) falsePositives++;
    }
    std::cout << "\nfalse positives on " << FALSE_POSITIVE_FRAMES << " object-free frames: " << falsePositives
              << (falsePositives == 0 ? " (ok)" : " (FAIL)") << '\n';
    return failures + (falsePositives > 0 ? 1 : 0);
}

constexpr const char* USAGE = "Usage: find_object_test [orb|sift|both] [photo] [output_dir]";

std::optional<std::vector<DetectorType>> parseDetectors(const std::string& which) {
    if (which == "both") return std::vector<DetectorType>{DetectorType::ORB, DetectorType::SIFT};
    if (std::optional<DetectorType> type = parseDetectorType(which)) return std::vector<DetectorType>{*type};
    return std::nullopt;
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = (argc > 1) ? argv[1] : "both";
    std::string photoPath = (argc > 2) ? argv[2] : "object.jpg";
    std::string outDir = (argc > 3) ? argv[3] : "test_output";

    std::optional<std::vector<DetectorType>> detectors = parseDetectors(which);
    if (!detectors) {
        std::cerr << USAGE << std::endl;
        return 2;
    }
    cv::Mat photo = cv::imread(photoPath);
    if (photo.empty()) {
        std::cerr << "Error: could not read photo '" << photoPath << "'." << std::endl;
        return 2;
    }
    std::filesystem::create_directories(outDir);

    cv::Mat mask = isolateObject(photo, centeredRect(photo.size(), PHOTO_SEED_FRACTION));
    if (mask.empty()) {
        std::cout << "=== isolation ===\nFAIL: could not isolate the object\n";
        return 1;
    }

    int failures = checkIsolation(photo, mask, outDir);
    for (DetectorType type : *detectors) failures += runDetector(type, photo, mask, outDir);

    std::cout << "\nImages written to '" << outDir << "/' (cutout.jpg, and <detector>_<case>.jpg examples).\n"
              << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return failures == 0 ? 0 : 1;
}
