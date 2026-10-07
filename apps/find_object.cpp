// Learns an object from a photo, then looks for it in the camera feed.
//
// Usage: find_object [--detector=orb|sift] [camera_index] [image_path]
//   defaults: orb, camera 0, object.jpg
// See docs/feature-detector-decision.md for how to choose a detector.

#include "motion/camera.hpp"
#include "motion/exit_code.hpp"
#include "motion/object_finder.hpp"

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
constexpr std::string_view DETECTOR_OPTION = "--detector=";
constexpr const char* USAGE = "Usage: find_object [--detector=orb|sift] [camera_index] [image_path]";

struct Options {
    DetectorType detector = DEFAULT_DETECTOR;
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

// Shows the camera feed with a box around the object until it is found and the
// user presses 'q', or until the search times out without ever finding it.
ExitCode searchForObject(Camera& camera, const ObjectFinder& finder) {
    using Clock = std::chrono::steady_clock;
    // Started at the first real frame so camera warm-up doesn't count.
    std::optional<Clock::time_point> searchStart;
    bool everFound = false;

    std::cout << "Looking for the object for up to " << SEARCH_TIMEOUT_SECONDS
              << " seconds. Press 'q' to quit." << std::endl;

    while (true) {
        std::optional<cv::Mat> frame = camera.read();
        if (!frame) return ExitCode::Error;
        if (!searchStart) searchStart = Clock::now();

        if (std::optional<cv::Rect> box = finder.find(*frame)) {
            if (!everFound) std::cout << "Object found." << std::endl;
            everFound = true;
            cv::rectangle(*frame, *box, cv::Scalar(0, 255, 0), 2);
        } else if (!everFound) {
            double elapsed = std::chrono::duration<double>(Clock::now() - *searchStart).count();
            if (elapsed >= SEARCH_TIMEOUT_SECONDS) {
                std::cout << "Object not found" << std::endl;
                return ExitCode::NotFound;
            }
            int secondsLeft = static_cast<int>(SEARCH_TIMEOUT_SECONDS - elapsed) + 1;
            cv::putText(*frame, "Searching... " + std::to_string(secondsLeft) + "s",
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow(WINDOW_NAME, *frame);
        if (cv::waitKey(1) == 'q') return ExitCode::Ok;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<Options> options = parseArgs(argc, argv);
    if (!options) return toInt(ExitCode::Error);

    std::optional<ObjectFinder> finder = learnObject(options->imagePath, options->detector);
    if (!finder) return toInt(ExitCode::Error);

    std::optional<Camera> camera = openCamera(options->cameraIndex);
    if (!camera) return toInt(ExitCode::Error);

    ExitCode result = searchForObject(*camera, *finder);
    cv::destroyAllWindows();
    return toInt(result);
}
