// Shows the camera feed with a box in the center; Enter cuts out the object in
// the box and saves it as an image.
//
// Usage: capture_object [camera_index]   (default: camera 0)

#include "motion/camera.hpp"
#include "motion/exit_code.hpp"
#include "motion/segmentation.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <optional>

using namespace motion;

namespace {

constexpr int ENTER_KEY = 13;
constexpr int MIN_FOREGROUND_PIXELS = 500;
constexpr double CENTER_ZONE_FRACTION = 0.3;
constexpr int PREVIEW_DELAY_MS = 30;
constexpr int RESULT_DISPLAY_MS = 2000;
constexpr const char* OUTPUT_PATH = "captured_object.jpg";
constexpr const char* PREVIEW_WINDOW = "Camera Preview";
constexpr const char* CAPTURE_WINDOW = "Captured Object";

// Cuts the object in `zone` out of `frame`. Prints a warning and returns
// nullopt if there is no distinct object there.
std::optional<cv::Mat> captureObject(const cv::Mat& frame, const cv::Rect& zone) {
    // GrabCut treats the zone as "probably foreground" and everything outside
    // it as background, then refines that guess using color statistics -- much
    // more robust to cluttered/uneven backgrounds than a global brightness
    // threshold.
    cv::Mat mask = isolateObject(frame, zone);
    if (mask.empty() || cv::countNonZero(mask) < MIN_FOREGROUND_PIXELS) {
        std::cerr << "\aCapture failed: no distinct object found in the center zone." << std::endl;
        return std::nullopt;
    }
    return cutOutObject(frame, mask);
}

// Shows the camera feed with the center zone. Enter captures the object in the
// zone; 'q' quits.
ExitCode captureLoop(Camera& camera) {
    std::cout << "Press Enter to capture the central object, or 'q' to quit." << std::endl;

    while (true) {
        std::optional<cv::Mat> frame = camera.read();
        if (!frame) return ExitCode::Error;

        cv::Rect zone = centeredRect(frame->size(), CENTER_ZONE_FRACTION);
        cv::Mat preview = frame->clone();
        cv::rectangle(preview, zone, cv::Scalar(255, 0, 0), 2);
        cv::imshow(PREVIEW_WINDOW, preview);

        int key = cv::waitKey(PREVIEW_DELAY_MS);
        if (key == 'q') return ExitCode::Ok;
        if (key != ENTER_KEY) continue;

        if (std::optional<cv::Mat> object = captureObject(*frame, zone)) {
            cv::imwrite(OUTPUT_PATH, *object);
            std::cout << "Success! Object captured and saved to '" << OUTPUT_PATH << "'." << std::endl;
            cv::imshow(CAPTURE_WINDOW, *object);
            cv::waitKey(RESULT_DISPLAY_MS);
            return ExitCode::Ok;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Error: too many arguments.\nUsage: capture_object [camera_index]" << std::endl;
        return toInt(ExitCode::Error);
    }
    std::optional<int> cameraIndex = (argc > 1) ? parseCameraIndex(argv[1]) : DEFAULT_CAMERA_INDEX;
    if (!cameraIndex) return toInt(ExitCode::Error);

    std::optional<Camera> camera = openCamera(*cameraIndex);
    if (!camera) return toInt(ExitCode::Error);

    ExitCode result = captureLoop(*camera);
    cv::destroyAllWindows();
    return toInt(result);
}
