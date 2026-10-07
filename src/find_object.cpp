#include "camera.hpp"
#include "exit_code.hpp"
#include "object_finder.hpp"

#include <chrono>
#include <iostream>
#include <optional>
#include <string>

// See docs/feature-detector-decision.md for why ORB is the default and how to
// switch to SIFT.
constexpr DetectorType DETECTOR = DetectorType::ORB;

constexpr double SEARCH_TIMEOUT_SECONDS = 15.0;
constexpr const char* DEFAULT_IMAGE_PATH = "object.jpg";
constexpr const char* WINDOW_NAME = "Find Object";

std::optional<ObjectFinder> learnObject(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: could not read image '" << imagePath << "'." << std::endl;
        return std::nullopt;
    }

    auto finder = ObjectFinder::fromPhoto(image, DETECTOR);
    if (!finder) {
        std::cerr << "Error: could not isolate the object or find features on it in '"
                  << imagePath << "'." << std::endl;
        return std::nullopt;
    }
    std::cout << "Learned object from '" << imagePath << "' (" << finder->featureCount()
              << " features, " << (DETECTOR == DetectorType::SIFT ? "SIFT" : "ORB") << ")." << std::endl;
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

int main(int argc, char** argv) {
    std::optional<int> cameraIndex = cameraIndexFromArgs(argc, argv);
    if (!cameraIndex) return toInt(ExitCode::Error);
    std::string imagePath = (argc > 2) ? argv[2] : DEFAULT_IMAGE_PATH;

    std::optional<ObjectFinder> finder = learnObject(imagePath);
    if (!finder) return toInt(ExitCode::Error);

    std::optional<Camera> camera = openCamera(*cameraIndex);
    if (!camera) return toInt(ExitCode::Error);

    ExitCode result = searchForObject(*camera, *finder);
    cv::destroyAllWindows();
    return toInt(result);
}
