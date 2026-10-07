#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

// The part of a photo, centered, that is assumed to contain the whole object.
constexpr double PHOTO_SEED_FRACTION = 0.84;

// A rectangle covering `fraction` of `size` in each dimension, centered.
[[nodiscard]] cv::Rect centeredRect(cv::Size size, double fraction);

// The outline of the largest white blob in a binary mask, if there is one.
[[nodiscard]] std::optional<std::vector<cv::Point>> largestContour(const cv::Mat& mask);

// Separates the object inside `seed` from its background using GrabCut, then
// drops the object's shadow (see segmentation.cpp). Returns a binary mask
// (255 = object) the same size as `image`, or an empty Mat if nothing is found.
[[nodiscard]] cv::Mat isolateObject(const cv::Mat& image, const cv::Rect& seed);

// The masked object on a white background, cropped to its bounding box.
[[nodiscard]] cv::Mat cutOutObject(const cv::Mat& image, const cv::Mat& mask);
