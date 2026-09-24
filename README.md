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
