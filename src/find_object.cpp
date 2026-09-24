#include "object_finder.hpp"

#include <chrono>
#include <iostream>
#include <string>

// See docs/feature-detector-decision.md for why ORB is the default and how to
// switch to SIFT.
constexpr DetectorType DETECTOR = DetectorType::ORB;

constexpr double SEARCH_TIMEOUT_SECONDS = 15.0;
constexpr int MAX_EMPTY_FRAMES = 100;

int main(int argc, char** argv) {
    int cameraIndex = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string imagePath = (argc > 2) ? argv[2] : "object.jpg";

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: could not read image '" << imagePath << "'." << std::endl;
        return 2;
    }

    ObjectFinder finder(DETECTOR);
    if (!finder.learn(image)) {
        std::cerr << "Error: could not isolate the object or find features on it in '"
                  << imagePath << "'." << std::endl;
        return 2;
    }
    std::cout << "Learned object from '" << imagePath << "' (" << finder.featureCount()
              << " features, " << (DETECTOR == DetectorType::SIFT ? "SIFT" : "ORB") << ")." << std::endl;

    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraIndex << "." << std::endl;
        return 2;
    }

    using Clock = std::chrono::steady_clock;
    Clock::time_point searchStart;
    bool timerStarted = false;
    bool everFound = false;
    int emptyFrames = 0;
    cv::Mat frame;

    std::cout << "Looking for the object for up to " << SEARCH_TIMEOUT_SECONDS
              << " seconds. Press 'q' to quit." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            // The camera often returns empty frames while warming up.
            if (++emptyFrames > MAX_EMPTY_FRAMES) {
                std::cerr << "Error: camera opened but returned no frames. "
                             "Check camera permission for your terminal app." << std::endl;
                return 2;
            }
            cv::waitKey(30);
            continue;
        }
        emptyFrames = 0;

        // Start the clock at the first real frame so camera warm-up doesn't count.
        if (!timerStarted) {
            searchStart = Clock::now();
            timerStarted = true;
        }

        cv::Rect box;
        bool found = finder.find(frame, box);

        if (found) {
            if (!everFound) std::cout << "Object found." << std::endl;
            everFound = true;
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        } else if (!everFound) {
            double elapsed = std::chrono::duration<double>(Clock::now() - searchStart).count();
            if (elapsed >= SEARCH_TIMEOUT_SECONDS) {
                std::cout << "Object not found" << std::endl;
                return 1;
            }
            cv::putText(frame, "Searching... " + std::to_string(static_cast<int>(SEARCH_TIMEOUT_SECONDS - elapsed) + 1) + "s",
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Find Object", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
