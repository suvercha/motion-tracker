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

#include "object_finder.hpp"

#include <opencv2/geometry.hpp>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int TRIALS = 6;
constexpr int REQUIRED_HITS = 5;   // out of TRIALS, for required cases
constexpr double MIN_IOU = 0.5;    // detected box vs true box, to count as a hit
constexpr int FALSE_POSITIVE_FRAMES = 20;

struct Case {
    const char* id;
    const char* description;
    double angle, scale, blur, brightness, noise, tilt;
    bool required;  // failing a required case fails the test
};

// The "informational" cases are known hard: they are reported but don't fail.
const std::vector<Case> CASES = {
    {"baseline",  "rotated, 0.45x",              20, 0.45, 0,   1.0,  0,  0,    true},
    {"tilt25",    "perspective tilt (mild)",     20, 0.45, 0,   1.0,  0,  0.25, true},
    {"tilt40",    "perspective tilt (strong)",   20, 0.45, 0,   1.0,  0,  0.45, true},
    {"noise",     "sensor noise (sd 12)",        20, 0.45, 0,   1.0,  12, 0,    true},
    {"blur",      "blur (sigma 3.5)",            20, 0.45, 3.5, 1.0,  0,  0,    true},
    {"small",     "small in frame (0.15x)",      20, 0.15, 0,   1.0,  0,  0,    true},
    {"dark",      "dark (x0.55 brightness)",     20, 0.45, 0,   0.55, 0,  0,    false},
    {"combo",     "small+blur+dark+noise+tilt",  70, 0.25, 2.0, 0.7,  8,  0.25, false},
};

// Background made from the photo's own background (its top-left corner, which
// is assumed to be plain background), plus colored boxes and text as clutter.
cv::Mat makeBackground(const cv::Mat& photo, cv::Size size, cv::RNG& rng) {
    cv::Mat patch = photo(cv::Rect(0, 0, photo.cols / 6, photo.rows / 5)).clone();
    cv::Mat bg;
    cv::resize(patch, bg, size);
    for (int i = 0; i < 25; i++) {
        cv::Rect r(rng.uniform(0, size.width - 100), rng.uniform(0, size.height - 100),
                   rng.uniform(30, 160), rng.uniform(30, 160));
        cv::rectangle(bg, r, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)),
                      cv::FILLED);
    }
    for (int i = 0; i < 6; i++) {
        cv::putText(bg, "HELPS DRY",
                    cv::Point(rng.uniform(0, size.width - 300), rng.uniform(40, size.height - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 3);
    }
    return bg;
}

struct Scene {
    cv::Mat frame;
    cv::Rect truth;
};

Scene makeScene(const cv::Mat& photo, const cv::Mat& mask, const Case& c, int trial, cv::RNG& rng) {
    const cv::Size size(1280, 720);
    Scene s;
    s.frame = makeBackground(photo, size, rng);

    // Rotate/scale the object to a random spot near the middle of the frame.
    cv::Point2f center(photo.cols / 2.f, photo.rows / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(center, c.angle + trial * 47, c.scale);
    M.at<double>(0, 2) += size.width / 2.0 - center.x + rng.uniform(-200, 200);
    M.at<double>(1, 2) += size.height / 2.0 - center.y + rng.uniform(-100, 100);
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    M.copyTo(H(cv::Rect(0, 0, 3, 2)));

    if (c.tilt > 0) {
        double sign = (trial % 2) ? 1 : -1;
        cv::Mat P = cv::Mat::eye(3, 3, CV_64F);
        P.at<double>(2, 0) = sign * c.tilt * 0.0006 / c.scale;
        P.at<double>(2, 1) = -sign * c.tilt * 0.0003 / c.scale;
        cv::Mat toCenter = cv::Mat::eye(3, 3, CV_64F);
        toCenter.at<double>(0, 2) = -size.width / 2.0;
        toCenter.at<double>(1, 2) = -size.height / 2.0;
        cv::Mat fromCenter = cv::Mat::eye(3, 3, CV_64F);
        fromCenter.at<double>(0, 2) = size.width / 2.0;
        fromCenter.at<double>(1, 2) = size.height / 2.0;
        H = fromCenter * P * toCenter * H;
    }

    cv::Mat warped, warpedMask;
    cv::warpPerspective(photo, warped, H, size);
    cv::warpPerspective(mask, warpedMask, H, size, cv::INTER_NEAREST);
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
    // Webcam-like compression.
    std::vector<uchar> jpg;
    cv::imencode(".jpg", s.frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 70});
    s.frame = cv::imdecode(jpg, cv::IMREAD_COLOR);
    return s;
}

// Returns the number of failures.
int runDetector(DetectorType type, const cv::Mat& photo, const cv::Mat& mask,
                const std::string& outDir) {
    const char* name = (type == DetectorType::SIFT) ? "sift" : "orb";
    std::printf("\n=== %s ===\n", name);

    ObjectFinder finder(type);
    if (!finder.learn(photo)) {
        std::printf("FAIL: could not learn the object\n");
        return 1;
    }
    std::printf("learned %zu features\n\n", finder.featureCount());
    std::printf("%-9s %-30s %-5s %-8s %-8s %s\n", "case", "description", "hits", "meanIoU", "meanMs", "result");

    int failures = 0;
    cv::RNG rng(11);  // fixed seed: same scenes every run
    for (const Case& c : CASES) {
        int hits = 0;
        double iouSum = 0, msSum = 0;
        for (int t = 0; t < TRIALS; t++) {
            Scene scene = makeScene(photo, mask, c, t, rng);
            cv::Rect box;
            auto start = std::chrono::steady_clock::now();
            bool found = finder.find(scene.frame, box);
            msSum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

            double iou = found ? double((box & scene.truth).area()) / double((box | scene.truth).area()) : 0;
            if (found && iou > MIN_IOU) {
                hits++;
                iouSum += iou;
            }
            if (t == 0) {
                cv::Mat annotated = scene.frame.clone();
                cv::rectangle(annotated, scene.truth, cv::Scalar(255, 0, 0), 2);
                if (found) cv::rectangle(annotated, box, cv::Scalar(0, 255, 0), 2);
                cv::imwrite(outDir + "/" + name + "_" + c.id + ".jpg", annotated);
            }
        }
        bool ok = hits >= REQUIRED_HITS;
        const char* result = ok ? "ok" : (c.required ? "FAIL" : "(informational)");
        if (!ok && c.required) failures++;
        std::printf("%-9s %-30s %d/%-3d %-8.2f %-8.0f %s\n", c.id, c.description, hits, TRIALS,
                    hits ? iouSum / hits : 0.0, msSum / TRIALS, result);
    }

    int falsePositives = 0;
    for (int i = 0; i < FALSE_POSITIVE_FRAMES; i++) {
        cv::Rect box;
        if (finder.find(makeBackground(photo, cv::Size(1280, 720), rng), box)) falsePositives++;
    }
    std::printf("\nfalse positives on %d object-free frames: %d %s\n", FALSE_POSITIVE_FRAMES,
                falsePositives, falsePositives == 0 ? "(ok)" : "(FAIL)");
    if (falsePositives > 0) failures++;
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = (argc > 1) ? argv[1] : "both";
    std::string photoPath = (argc > 2) ? argv[2] : "object.jpg";
    std::string outDir = (argc > 3) ? argv[3] : "test_output";

    if (which != "orb" && which != "sift" && which != "both") {
        std::cerr << "Usage: find_object_test [orb|sift|both] [photo] [output_dir]" << std::endl;
        return 2;
    }
    cv::Mat photo = cv::imread(photoPath);
    if (photo.empty()) {
        std::cerr << "Error: could not read photo '" << photoPath << "'." << std::endl;
        return 2;
    }
    std::filesystem::create_directories(outDir);

    // Isolation check: the cutout should be a sensible blob that doesn't touch the photo's edge.
    std::printf("=== isolation ===\n");
    cv::Mat mask = ObjectFinder::isolateObject(photo);
    int failures = 0;
    if (mask.empty()) {
        std::printf("FAIL: could not isolate the object\n");
        return 1;
    }
    cv::Rect bounds = cv::boundingRect(mask);
    double fraction = cv::countNonZero(mask) / static_cast<double>(mask.total());
    bool touchesEdge = bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= photo.cols || bounds.br().y >= photo.rows;
    bool sensible = fraction > 0.05 && fraction < 0.8 && !touchesEdge;
    std::printf("object covers %.0f%% of the photo, %s the photo edge: %s\n", fraction * 100,
                touchesEdge ? "touches" : "clear of", sensible ? "ok" : "FAIL");
    if (!sensible) failures++;
    cv::Mat cutout(photo.size(), photo.type(), cv::Scalar(255, 255, 255));
    photo.copyTo(cutout, mask);
    cv::imwrite(outDir + "/cutout.jpg", cutout);

    if (which == "orb" || which == "both") failures += runDetector(DetectorType::ORB, photo, mask, outDir);
    if (which == "sift" || which == "both") failures += runDetector(DetectorType::SIFT, photo, mask, outDir);

    std::printf("\nImages written to '%s/' (cutout.jpg, and <detector>_<case>.jpg examples).\n", outDir.c_str());
    std::printf("%s\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
