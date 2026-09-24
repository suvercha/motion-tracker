# Decision: ORB vs SIFT for `find_object`

**Status:** ORB is the default. Switch to SIFT if ORB proves too inaccurate on the real webcam.

`find_object` learns an object from a photo (`object.jpg`), then looks for it in the
webcam feed by matching local image features between the photo and each frame. This
document records why ORB was chosen first and how to switch.

## Decision

Start with **ORB**, because it is faster. Fall back to **SIFT** if ORB misses the object
too often in real use.

## How to switch to SIFT

In [src/find_object.cpp](../src/find_object.cpp), change one line and rebuild:

```cpp
constexpr DetectorType DETECTOR = DetectorType::SIFT;   // was ORB
```

The distance metric (Hamming for ORB, L2 for SIFT) and the reference image size (320 px
for ORB, 640 px for SIFT) switch automatically with it. The detector code lives in
[src/object_finder.cpp](../src/object_finder.cpp).

## Comparison

| | ORB | SIFT |
|---|---|---|
| Speed (per 1280x720 frame, Apple Silicon) | ~13 ms | ~43 ms |
| Descriptor | Binary (256 bits), compared with Hamming distance | 128 floats, compared with L2 distance |
| Scale range | Narrow: limited by an 8-level pyramid, so the reference size matters | Wide: handles large scale differences well |
| Blur, low contrast | Noticeably weaker | Robust |
| Rotation | Handled | Handled |
| Perspective tilt | Handled (with a homography) | Handled (with a homography) |
| Extra setup | None | None (SIFT is part of the main OpenCV 5 `features` module) |
| Licensing | Free | Free (patent expired in 2020) |

**Pros of ORB:** about 3x faster, so a smoother live preview and headroom on slower
machines; less CPU heat; compact binary descriptors.

**Cons of ORB:** less reliable when the object is small in frame, blurred, or dim; more
sensitive to how large the reference image is relative to the object on screen.

**Pros of SIFT:** more accurate across scale, blur and lighting; needs less tuning.

**Cons of SIFT:** about 3x slower per frame (still roughly 20 fps on the test machine);
larger descriptors.

## Measurements behind the decision

These come from an offline test, not the live webcam. The object is cut out of
`object.jpg`, pasted onto a cluttered 1280x720 background, and then rotated, scaled,
tilted, darkened, blurred, or noised, with JPEG compression (quality 70) applied. Each
case runs 6 times. A hit means the detected box overlapped the true box with IoU > 0.5.
Both detectors had 0 false positives on 20 object-free frames.

Reproduce with `ctest --test-dir build --output-on-failure`, or run
`./build/find_object_test both` for the full table (see the README).

| Case | ORB (320 px reference, current) | SIFT (640 px reference) |
|---|---|---|
| Baseline | 6/6 | 6/6 |
| Perspective tilt (mild / strong) | 6/6, 6/6 | 6/6, 6/6 |
| Sensor noise | 6/6 | 6/6 |
| Blur (sigma 3.5) | 6/6 | 6/6 |
| Small in frame (~180 px wide) | 6/6 | 6/6 |
| Darker (x0.55 brightness) | 3/6 | 6/6 |
| Everything combined | 1/6 | 2/6 |
| Time per frame | ~13 ms | ~45 ms |

What this tells us:
- ORB was tuned once. With a 640 px reference it missed small and blurry objects
  (0/6 and 4/6); shrinking the reference to 320 px moved the learned features closer to
  the size the object appears on screen and fixed both. Lowering ORB's corner threshold
  did not help the dark case.
- ORB's remaining gap is dim lighting. SIFT closes it.
- Caveat: the test pastes pixels from the same photo that was used for learning, so it is
  easier than the real world. A real webcam differs in lighting, color, focus and
  compression, so expect lower hit rates than the table shows for both detectors.
- The dark and combined cases are known hard and are reported but do not fail the test.

## When to revisit

Switch to SIFT if, on the real webcam, ORB often fails to find the object within the
15-second window when it is clearly visible, especially in dim rooms, at a distance, or
with motion blur.

## Related decisions in the same program

- **Isolating the object from the photo:** GrabCut, followed by removal of pixels that
  match the background's hue. GrabCut alone kept the object's shadow, which made the
  bounding box too large. The hue step only applies when the photo's background is
  clearly colored (not gray, white or black).
- **Locating the object in a frame:** feature matching, then a RANSAC homography, then the
  object's outline projected into the frame. Using the outline (not the reference
  image's corners) keeps the box tight when the object is rotated.
- **Timeout:** 15 seconds, measured from the first real camera frame, covering the first
  search only. Once the object is found, the program keeps drawing the box until `q`.
