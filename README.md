# motion-tracking

Two small OpenCV programs that work with a webcam:

- **`capture_object`**: shows a live preview; press Enter to cut out the object in the center
  of the frame and save it as an image.
- **`find_object`**: learns an object from a photo, finds it in the live feed, and follows it,
  drawing its recent path and showing its speed in cm/s.

## Capturing an object: `capture_object`

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
./scripts/download_models.sh     # once: fetches the tracker's model files into models/
./build/find_object [--detector=orb|sift] [--models=DIR] [--object-width-cm=N] [camera_index] [image_path]
# defaults: orb, the project's models/ folder, 4.5 cm, camera 0, object.jpg
```

1. The object is isolated from the photo (`object.jpg` should show the object against a
   plain background, roughly centered) using GrabCut.
2. The camera opens and each frame is searched for the object using feature matching
   (ORB by default, or SIFT with `--detector=sift`).
3. When it is found, a bounding box is drawn around it and it is followed until you press
   **q**:
   - the box is **green** when the detector found the object in that frame, and **yellow**
     when the detector missed it (for example, motion blur) and a small neural-network
     tracker (OpenCV's `TrackerNano`, about 4 ms a frame) is following it;
   - the object's recent path (its last 64 positions) is drawn as a **red line** that tapers from
     thick at the object to thin at its tail. `TRAIL_LENGTH` in
     [include/motion/path_history.hpp](include/motion/path_history.hpp) sets how long the tail is;
   - its **speed** is shown in the top-left corner, in cm/s and px/s (`--` when it is not known).
4. If the object is not found within **15 seconds**, the program prints
   `Object not found`, exits, and returns exit code 1.

**Speed in cm/s** uses the object's real width: the apparent width of the object in the frame,
compared with `--object-width-cm` (4.5 by default, the tin in `object.jpg`), gives the real
size of a pixel, and the pixel speed is multiplied by it. It is only right for motion
**across** the view at about a constant distance; motion toward or away from the camera is not
measured. Pass `--object-width-cm=0` to show pixels per second only, and set the width to match
your object when you use another photo.

ORB is the default because it is fast. See
[docs/feature-detector-decision.md](docs/feature-detector-decision.md) for the trade-offs
against SIFT, and [docs/tracking-design.md](docs/tracking-design.md) for how the object is
followed, why `TrackerNano` was chosen, and the known limit: the tracker is trusted for at
most 1 second, so an object that stays motion-blurred for longer is lost.

The tracker's model files (about 1.8 MB) are not stored in the repository; see
[models/README.md](models/README.md).

## Tests

Offline tests for `find_object` (no camera needed). They learn the object from
`object.jpg`, paste it onto cluttered backgrounds under different conditions (rotation,
perspective tilt, darkness, noise, blur, small size), and check that it is found with a
tight box and that object-free frames give no false detections.

```bash
ctest --test-dir build --output-on-failure      # runs every test below
./build/find_object_test both                   # full report for both detectors
./build/find_object_test orb path/to/photo.jpg  # one detector, or a different photo
./build/tracking_test orb                       # following a moving, blurred object
./build/path_history_test                       # the trail and speed maths
```

- **`find_object_test`**: hits, mean overlap with the true box, error of the object's center,
  and time per frame for each case. The isolated cutout and one annotated example per case
  (green = detected box, blue = true box) are saved to `test_output/` (or
  `build/test_output/` under `ctest`).
- **`tracking_test`**: renders a short video of the object moving at a known speed, with
  motion blur, and checks the positions and speed found by the detector plus tracker against
  the truth. It also reports how often the detector alone would have found the object, and
  takes about a minute. It needs the tracker models (`scripts/download_models.sh`). For the
  blur scenarios it checks that the tracker carries the object for about a second after the
  detector loses it, with an accurate speed; following a blur that lasts longer is a known
  limit, see [docs/tracking-design.md](docs/tracking-design.md).
- **`path_history_test`**: unit tests for the trail, the speeds in px/s and cm/s, and the
  trail drawing, using made-up timestamps.

The tests need `object.jpg` in the project root, and expect the photo's top-left corner
to be plain background.

## Choosing the camera

Both programs use camera `0` by default. Pass the camera index to use another one (for
`find_object` it is the first number on the command line):

```bash
./build/capture_object      # camera 0
./build/capture_object 1    # camera 1
./build/find_object 1       # camera 1
```

On a Mac with an iPhone nearby, Continuity Camera can take index `0`, and the built-in
camera then becomes `1`. If you see a black preview, try the other index.

## Requirements

- CMake 3.16+
- A C++20 compiler
- OpenCV 5.0 (for example `brew install opencv`)

On macOS, the terminal app you run this from needs camera permission
(System Settings > Privacy & Security > Camera). The first launch after installing
OpenCV can take a while while macOS verifies the libraries.

## Build

```bash
cmake -S . -B build
cmake --build build
```

Tests are built too; turn them off with `cmake -S . -B build -DBUILD_TESTING=OFF`.

`find_object` and `tracking_test` also need the tracker's model files, downloaded once:

```bash
./scripts/download_models.sh
```

## Run

```bash
./build/capture_object [camera_index]
./build/find_object [--detector=orb|sift] [--models=DIR] [--object-width-cm=N] [camera_index] [image_path]
```

## Project layout

| Path | Contents |
|---|---|
| `apps/` | The two programs: `capture_object.cpp`, `find_object.cpp` |
| `include/motion/` | Public headers of the shared library (namespace `motion`) |
| `src/` | The shared library: camera access, object segmentation, object learning/finding |
| `tests/` | Offline tests |
| `docs/` | Design decisions and measurements |
| `models/` | The tracker's model files (downloaded, not in git) |
| `scripts/` | `download_models.sh` |
