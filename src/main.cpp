#include <opencv2/opencv.hpp>
#include <opencv2/geometry/2d.hpp>
#include <algorithm>
#include <iostream>
#include <vector>

constexpr int ENTER_KEY = 13;
constexpr int MIN_FOREGROUND_PIXELS = 500;
constexpr int MAX_EMPTY_FRAMES = 100;

int main(int argc, char** argv) {
    int cameraIndex = (argc > 1) ? std::atoi(argv[1]) : 0;
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraIndex << "." << std::endl;
        return -1;
    }

    cv::Mat frame;

    std::cout << "Press Enter to capture the central object, or 'q' to quit." << std::endl;

    int emptyFrames = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            // The camera often returns empty frames while warming up.
            if (++emptyFrames > MAX_EMPTY_FRAMES) {
                std::cerr << "Error: camera opened but returned no frames. "
                             "Check camera permission for your terminal app." << std::endl;
                break;
            }
            cv::waitKey(30);
            continue;
        }
        emptyFrames = 0;

        int zoneWidth = frame.cols * 0.3;
        int zoneHeight = frame.rows * 0.3;
        cv::Rect centerZone(
            (frame.cols - zoneWidth) / 2,
            (frame.rows - zoneHeight) / 2,
            zoneWidth,
            zoneHeight
        );

        cv::Mat preview = frame.clone();
        cv::rectangle(preview, centerZone, cv::Scalar(255, 0, 0), 2);
        cv::imshow("Camera Preview", preview);

        int key = cv::waitKey(30);

        if (key == 'q') {
            break;
        }
        else if (key == ENTER_KEY) {
            // GrabCut treats centerZone as "probably foreground" and everything
            // outside it as background, then refines that guess using color
            // statistics -- much more robust to cluttered/uneven backgrounds
            // than a global brightness threshold.
            cv::Mat mask, bgdModel, fgdModel;
            cv::grabCut(frame, mask, centerZone, bgdModel, fgdModel, 5, cv::GC_INIT_WITH_RECT);

            cv::Mat fgMask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);

            if (cv::countNonZero(fgMask) < MIN_FOREGROUND_PIXELS) {
                std::cerr << "\aCapture failed: no distinct object found in the center zone." << std::endl;
                continue;
            }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                std::cerr << "\aCapture failed: could not isolate an object boundary." << std::endl;
                continue;
            }

            // GrabCut can leave small speckles in the mask; the largest
            // contour is the actual object.
            const auto& largest = *std::max_element(
                contours.begin(), contours.end(),
                [](const auto& a, const auto& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });
            cv::Rect objectBox = cv::boundingRect(largest);

            // Lift the object out of the frame onto a white background, then
            // crop to just its bounding box.
            cv::Mat cutout(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
            frame.copyTo(cutout, fgMask);
            cv::Mat objectImage = cutout(objectBox).clone();

            cv::imwrite("captured_object.jpg", objectImage);
            std::cout << "Success! Object captured and saved to 'captured_object.jpg'." << std::endl;

            cv::imshow("Captured Object", objectImage);
            cv::waitKey(2000);
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
