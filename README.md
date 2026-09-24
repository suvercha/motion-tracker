# motion-tracking

A small OpenCV program that shows a live webcam feed and, when you press Enter,
captures the object in the center of the frame and saves it as an image.

## What it does

1. Opens a camera and shows a live preview with a blue box marking the center zone
   (the middle 30% of the frame, in each dimension).
2. Hold an object inside the blue box and press **Enter**.
3. The program runs [GrabCut](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html)
   segmentation seeded with that box to separate the object from the background.
4. The object is cut out onto a white background, cropped to its bounding box, saved as
   `captured_object.jpg` in the current directory, and shown for 2 seconds. Then the
   program exits.

If no distinct object is found in the box, it prints a warning and keeps the preview
running so you can adjust and press Enter again.

Press **q** at any time to quit.

## Finding a known object: `find_object`

A second program that learns an object from a photo and finds it in the live webcam feed.

```bash
./build/find_object [camera_index] [image_path]   # defaults: camera 0, object.jpg
```

1. The object is isolated from the photo (`object.jpg` should show the object against a
   plain background, roughly centered) using GrabCut.
2. The camera opens and each frame is searched for the object using ORB feature matching.
3. When it is found, a green bounding box is drawn around it. It stays on screen until you
   press **q**.
4. If the object is not found within **15 seconds**, the program prints
   `Object not found`, exits, and returns exit code 1.

ORB is used by default because it is fast. See
[docs/feature-detector-decision.md](docs/feature-detector-decision.md) for the trade-offs
against SIFT and how to switch.

## Tests

Offline tests for `find_object` (no camera needed). They learn the object from
`object.jpg`, paste it onto cluttered backgrounds under different conditions (rotation,
perspective tilt, darkness, noise, blur, small size), and check that it is found with a
tight box and that object-free frames give no false detections.

```bash
ctest --test-dir build --output-on-failure      # runs the ORB and SIFT tests
./build/find_object_test both                   # full report for both detectors
./build/find_object_test orb path/to/photo.jpg  # one detector, or a different photo
```

The report shows hits, mean overlap with the true box, and time per frame for each case.
The isolated cutout and one annotated example per case (green = detected box, blue =
true box) are saved to `test_output/` (or `build/test_output/` under `ctest`).
The tests need `object.jpg` in the project root, and expect the photo's top-left corner
to be plain background.

## Choosing the camera

The camera is selected with an optional argument, the camera index (default `0`):

```bash
./build/motion_tracking      # camera 0
./build/motion_tracking 1    # camera 1
```

On a Mac with an iPhone nearby, Continuity Camera can take index `0`, and the built-in
camera then becomes `1`. If you see a black preview, try the other index.

## Requirements

- CMake 3.16+
- A C++17 compiler
- OpenCV 5.0 (for example `brew install opencv`)

On macOS, the terminal app you run this from needs camera permission
(System Settings > Privacy & Security > Camera). The first launch after installing
OpenCV can take a while while macOS verifies the libraries.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/motion_tracking [camera_index]
```
