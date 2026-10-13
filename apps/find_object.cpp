// Learns an object from a photo, then looks for it in the camera feed. Once found,
// it follows the object, draws its recent path as a dotted line, and shows its speed.
//
// Usage: find_object [--detector=orb|sift] [--models=DIR] [camera_index] [image_path]
//   defaults: orb, the project's models/ folder, camera 0, object.jpg
// The tracker's model files must be in the models folder: run scripts/download_models.sh.
// See docs/feature-detector-decision.md for how to choose a detector, and
// docs/tracking-design.md for how the object is followed.

#include "motion/camera.hpp"
#include "motion/clock.hpp"
#include "motion/exit_code.hpp"
#include "motion/object_finder.hpp"
#include "motion/object_tracker.hpp"
#include "motion/overlay.hpp"
#include "motion/path_history.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace motion;

namespace {

constexpr DetectorType DEFAULT_DETECTOR = DetectorType::ORB;
constexpr double SEARCH_TIMEOUT_SECONDS = 15.0;
constexpr const char* DEFAULT_IMAGE_PATH = "object.jpg";
constexpr const char* WINDOW_NAME = "Find Object";
// OpenCV colors are BGR.
const cv::Scalar DETECTED_BOX_COLOR(0, 255, 0);   // green: found by the detector
const cv::Scalar TRACKED_BOX_COLOR(0, 255, 255);  // yellow: detector missed, following with the tracker
const cv::Scalar TRAIL_COLOR(0, 140, 255);        // orange
constexpr std::string_view DETECTOR_OPTION = "--detector=";
constexpr std::string_view MODELS_OPTION = "--models=";
constexpr const char* USAGE = "Usage: find_object [--detector=orb|sift] [--models=DIR] [camera_index] [image_path]";

struct Options {
    DetectorType detector = DEFAULT_DETECTOR;
    std::string modelsDir = DEFAULT_MODELS_DIR;
    int cameraIndex = DEFAULT_CAMERA_INDEX;
    std::string imagePath = DEFAULT_IMAGE_PATH;
};

// Prints an error and returns nullopt if the arguments are invalid.
std::optional<Options> parseArgs(int argc, char** argv) {
    Options options;
    std::vector<std::string_view> positional;

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg.substr(0, DETECTOR_OPTION.size()) == DETECTOR_OPTION) {
            std::string_view value = arg.substr(DETECTOR_OPTION.size());
            std::optional<DetectorType> detector = parseDetectorType(value);
            if (!detector) {
                std::cerr << "Error: unknown detector '" << value << "' (use orb or sift).\n" << USAGE << std::endl;
                return std::nullopt;
            }
            options.detector = *detector;
        } else if (arg.substr(0, MODELS_OPTION.size()) == MODELS_OPTION) {
            options.modelsDir = std::string(arg.substr(MODELS_OPTION.size()));
        } else if (arg.substr(0, 2) == "--") {
            std::cerr << "Error: unknown option '" << arg << "'.\n" << USAGE << std::endl;
            return std::nullopt;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 2) {
        std::cerr << "Error: too many arguments.\n" << USAGE << std::endl;
        return std::nullopt;
    }
    if (!positional.empty()) {
        std::optional<int> index = parseCameraIndex(positional[0]);
        if (!index) return std::nullopt;
        options.cameraIndex = *index;
    }
    if (positional.size() > 1) options.imagePath = positional[1];
    return options;
}

std::optional<ObjectFinder> learnObject(const std::string& imagePath, DetectorType detector) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: could not read image '" << imagePath << "'." << std::endl;
        return std::nullopt;
    }

    std::optional<ObjectFinder> finder = ObjectFinder::fromPhoto(image, detector);
    if (!finder) {
        std::cerr << "Error: could not isolate the object or find features on it in '"
                  << imagePath << "'." << std::endl;
        return std::nullopt;
    }
    std::cout << "Learned object from '" << imagePath << "' (" << finder->featureCount()
              << " features, " << toString(detector) << ")." << std::endl;
    return finder;
}

// Shows the camera feed. Until the object is first found it counts down the
// search time, and exits if the object never appears. After that it follows the
// object, drawing its box, its recent path, and its speed, until the user presses 'q'.
ExitCode searchForObject(Camera& camera, ObjectTracker& tracker) {
    // Started at the first real frame so camera warm-up doesn't count.
    std::optional<Clock::time_point> searchStart;
    PathHistory path;
    bool everFound = false;

    std::cout << "Looking for the object for up to " << SEARCH_TIMEOUT_SECONDS
              << " seconds. Press 'q' to quit." << std::endl;

    while (true) {
        std::optional<cv::Mat> frame = camera.read();
        if (!frame) return ExitCode::Error;
        Clock::time_point now = Clock::now();
        if (!searchStart) searchStart = now;

        // Draw on a copy: the tracker must see the clean frame.
        cv::Mat display = frame->clone();
        std::optional<Track> track = tracker.update(*frame, now);

        if (track) {
            if (!everFound) std::cout << "Object found." << std::endl;
            everFound = true;
            path.add(now, track->center);
        } else {
            path.breakPath();
        }

        if (everFound) {
            drawDottedTrail(display, path.trail(now), TRAIL_COLOR);
            drawSpeedLabel(display, path.speed(now));
        } else {
            double elapsed = std::chrono::duration<double>(now - *searchStart).count();
            if (elapsed >= SEARCH_TIMEOUT_SECONDS) {
                std::cout << "Object not found" << std::endl;
                return ExitCode::NotFound;
            }
            int secondsLeft = static_cast<int>(SEARCH_TIMEOUT_SECONDS - elapsed) + 1;
            cv::putText(display, "Searching... " + std::to_string(secondsLeft) + "s",
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        }
        if (track) cv::rectangle(display, track->box, track->detected ? DETECTED_BOX_COLOR : TRACKED_BOX_COLOR, 2);

        cv::imshow(WINDOW_NAME, display);
        if (cv::waitKey(1) == 'q') return ExitCode::Ok;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<Options> options = parseArgs(argc, argv);
    if (!options) return toInt(ExitCode::Error);

    std::optional<ObjectFinder> finder = learnObject(options->imagePath, options->detector);
    if (!finder) return toInt(ExitCode::Error);

    std::optional<ObjectTracker> tracker = ObjectTracker::create(std::move(*finder), options->modelsDir);
    if (!tracker) return toInt(ExitCode::Error);

    std::optional<Camera> camera = openCamera(options->cameraIndex);
    if (!camera) return toInt(ExitCode::Error);

    ExitCode result = searchForObject(*camera, *tracker);
    cv::destroyAllWindows();
    return toInt(result);
}
