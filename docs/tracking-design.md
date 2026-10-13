# Following the object: design and measurements

**Status:** `find_object` finds the object with a detector (ORB by default) on every frame
and uses OpenCV's `TrackerNano` only when the detector misses. Nano was chosen for its cost,
not for better accuracy: all five trackers compared are about equally accurate, and Nano is
13x cheaper than the most accurate one (see [Why TrackerNano](#why-trackernano)). The design
has one known limit: a moving object that stays motion-blurred for more than about a second
is lost (see [Limits](#limits)).

## What `find_object` does once the object is found

1. **Detect every frame.** `ObjectFinder` returns the object's box, its centroid projected
   through the homography, and its apparent size. The position used for the path and speed is
   that centroid rather than the box center, because a box center moves when a non-round
   object rotates (its bounding box changes size) and the centroid does not. On the tilted
   test views the projected centroid was within 1.4 px of the truth.
2. **Fill in misses.** When the detector misses a frame, `ObjectTracker` starts a
   `TrackerNano` from the last detected box and follows the object until the detector finds
   it again. The box is green when it comes from the detector and yellow when it comes from
   the tracker. The tracker's two model files are loaded once at startup.
3. **Give up after 1 second.** A tracker cannot tell when it has lost the object, so
   tracker-only positions are trusted for at most 1 second.
4. **Keep a path.** `PathHistory` keeps one entry per camera frame, in a queue of the last
   `TRAIL_LENGTH` frames (64, about 2 seconds at 30 fps; a constant in
   `include/motion/path_history.hpp`). A frame where the object was lost is a gap in the
   queue, so the tail fades away when the object is lost. The trail is drawn as line segments
   between consecutive points that taper with age, `thickness = sqrt(TRAIL_LENGTH / (i + 1)) * 2.5`
   for the i-th segment back, and no line is drawn across a gap.
5. **Speed in pixels.** A straight-line fit over the last 0.4 s of the path, in pixels per
   second. A single-frame difference would be useless: 2 px of detection jitter at 30 fps
   reads as 60 px/s. Below 15 px/s it reads 0, and it shows `--` as soon as the object is lost.
6. **Speed in cm/s.** See below.

## Speed in cm/s

A single camera cannot measure distance by itself, so the object's real width is needed
(`--object-width-cm`, 4.5 by default for the tin in `object.jpg`). Each detection reports the
object's widest dimension in pixels: the longest distance between two points of its outline
as projected into the frame. Then:

```
cm per pixel = real width / apparent width in pixels
speed in cm/s = speed in px/s x cm per pixel
```

The cm per pixel used is the median over the same 0.4 s window as the speed. While only the
tracker is following the object, the apparent width from the last detection is used, because
a tracker's box size drifts.

For motion **across** the view at about a constant distance this needs no camera calibration:
the focal length cancels out. Motion toward or away from the camera is not measured. Lens
distortion is ignored. The speed is as accurate as the width you give it: a width that is 5%
off makes every speed 5% off.

Measured with the synthetic test (a 222 px tin, so about 0.02 cm per pixel):

- The detector's apparent size was within 0.1-0.5% of the true size in the sharp test cases
  and 1.9% with blur (`find_object_test`).
- On sharp motion the cm/s error is 0.1%, the same as the px/s error.
- On blurred frames the cm/s error is within 1 point of the px/s error, so the size estimate
  adds very little; the error there comes from the tracker (below).

## Findings

Measured with `tests/tracking_test.cpp` and some temporary benchmarks (see
[Reproducing](#reproducing)): a synthetic 30 fps video of the tin moving at 250 px/s over a
cluttered background, still for the first 0.6 s (so it is found while sharp, as when someone
holds it up to the camera). Motion blur is rendered by averaging copies of the tin across
the exposure, and each frame gets noise and JPEG compression. All timings are from an
Apple M1 (8 cores, CPU only) with the machine otherwise idle.

### The pipeline as built (detector every frame, TrackerNano on misses)

Ranges are two runs. The object is learned with GrabCut, which is not exactly repeatable, so
results move a little from run to run.

| Scenario | Object found: detector alone | With tracker | Position error: detector frames | Position error: tracker frames | Median speed error (px/s and cm/s) |
|---|---|---|---|---|---|
| Sharp, 250 px/s | 100% | 100% | 0.9 px | n/a | 0.1-0.2% |
| 12 px motion blur | 20% | 53% | 0.8 px | 11-14 px | 5-11% too low |
| 25 px motion blur | 20% | 53% | 0.9 px | 14 px | 8-13% too low |
| 50 px motion blur | 19% | 52% | 0.8-0.9 px | 22 px | 16% too low |
| Not moving | 100% | 100% | 0.9-1.0 px | n/a | reads 0 in 100% of frames |
| Leaves the scene at 1.6 s | n/a | tracked for 0.97 s after it left, then stopped | n/a | n/a | n/a |

Where the detector works, position and speed are accurate. With any motion blur the
detector finds the tin only in the frames before it starts moving (the 20%), and the tracker
carries it for 1 second after that, which is where the 53% comes from. On those tracker
frames the speed reads low, by more the heavier the blur: trackers damp the movement they
report.

(An earlier version of this table showed blurred speed errors of 4-8%. The path code then
kept returning the last speed for 0.3 s after the object was lost, which repeated one
accurate reading for several frames. The speed now disappears as soon as the object is lost,
and the errors above are measured only on frames where the tracker is really following.)

### Cost per frame

Median over 25 frames of 1280x720 video, interleaved so every method sees the same
conditions (the object's box is 213x212 px):

| | ms per frame | Compared with the ORB detector | Time to start |
|---|---|---|---|
| ORB detector | 13.2 | 1.0x | none |
| TrackerMIL | 54.8 | 4.2x | 8 ms |
| TrackerKCF | 28.7 | 2.2x | 2 ms |
| TrackerCSRT | 23.8 | 1.8x | 21 ms |
| **TrackerNano** | **4.2** | **0.3x** | 2 ms |
| TrackerVit | 3.2 | 0.2x | 0.2 ms |

The SIFT detector costs about 43 ms per frame. The two neural trackers are 3-4x cheaper than
detecting with ORB, and about 10x cheaper than SIFT. The older trackers (MIL, KCF, CSRT) cost
more than the detector.

(An earlier run on a heavily loaded machine made every tracker look slower than the
detector. That was machine load; these numbers are from an idle machine.)

### Which tracker? End to end

The same video with each tracker swapped in. Each cell is the mean position error on tracker
frames / the median speed error (always too low, except MIL at 25 px where it was 1% high).

| Tracker | 12 px blur | 25 px blur | 50 px blur |
|---|---|---|---|
| TrackerMIL | 9 px / 6% | 4 px / 2% | 17 px / 13% |
| TrackerKCF | 16 px / 9% | 17 px / 10% | 19 px / 13% |
| TrackerCSRT | 21 px / 26% | 8 px / 6% | 22 px / 12% |
| **TrackerNano** (two runs) | 11-14 px / 5-11% | 14 px / 8-13% | 22 px / 16% |
| TrackerVit | 7 px / 7% | 8 px / 12% | 11 px / 15% |

No tracker is clearly more accurate. The differences between them are about the size of the
run-to-run variation (Nano alone varies by 5 points at 12 px; two MIL runs differed by up to
9 px and 6 points). CSRT is the one clear outlier, at 26% at 12 px. "Frames followed" is
49-54% for every tracker, because it is set by the 1-second limit, not by the tracker.

### Following alone, with no detector to re-anchor it

Each tracker started from a sharp detection and then followed the object for the rest of the
video (66 frames of movement, about 2.2 s). Counts are frames within 20 px of the true
position. This does not use the path code, so the scoring change above does not affect it.

| Tracker | 25 px blur (of 66) | 50 px blur (of 66) | Object in view before it left (of 24) | Still reported a box after the object left | Confidence score |
|---|---|---|---|---|---|
| TrackerMIL | 28 | 11 | 24 | all 68 frames | none |
| TrackerKCF | 15 | 6 | 16 | none: said it was lost after 0.1 s | none |
| TrackerCSRT | 7 | 0 | 24 | all 68 frames | none |
| TrackerNano | 31 | 12 | 21 | all 68 frames | 0.8-0.9 in view, 0.8-0.9 after it left |
| TrackerVit | 31 | 24 | 24 | all 68 frames | 0.6-0.8 in view (0.3 under blur), 0.6-0.7 after it left |

Every tracker drifts when it is left alone for a couple of seconds, so none can replace the
detector. The neural trackers hold on better than the older ones under heavy blur. Only
KCF noticed that the object had gone, and it is the worst at following it.

### Does a different detector fix the blur?

No. ORB found the tin in 19-20% of frames and SIFT in 21%: in both cases only the frames
where it was still. Neither finds a 222 px object once it is smeared over 12 px or more.
SIFT also costs about 3x as much per frame.

## Why TrackerNano

- **Cheap, which keeps the frame rate up.** The detector costs about 13 ms a frame. On a
  detector miss, Nano adds 4 ms (about 17 ms in total, so the loop keeps up with a 30 fps
  camera). MIL would add 55 ms (about 68 ms in total, or about 15 fps) in exactly the frames
  where the object is moving fast.
- **Equally accurate as far as these runs can tell.** Its speed error is in the same 5-16%
  range as MIL, KCF and ViT, and better than CSRT at 12 px. It is not more accurate than MIL.
- **Small.** Two model files, about 1.8 MB together.
- **No tracker offered a usable lost signal**, so the 1-second limit is needed whichever is
  used. The neural trackers' scores do not separate "object in view" from "object gone", and
  the one tracker that noticed the object leaving (KCF) is the worst at following it.

ViT is cheaper still (3.2 ms) and has the best position accuracy, but its speed error under
heavy blur is in the same range as the others and it has no other advantage here.

**Trade-off:** Nano needs its model files at runtime. They are not in the repository (the
repository the OpenCV documentation links to shows no license), so
`scripts/download_models.sh` fetches them into `models/`. All five trackers can be chosen with
`find_object --tracker=NAME`, if you want to use another.

## Limits

- **A moving, blurred object is followed for 1 second at most.** The detector fails on it,
  and nothing can tell "blurred" from "gone". Tracker scores do not separate them (Nano's
  score was 0.8-0.9 whether the tin was in view or had left), and the detector fails in both
  cases. A longer limit would follow the background after the object leaves.
- **The speed reads low on tracker frames**, by 5-16% depending on the blur. It is accurate
  (within 0.2%) when the detector is finding the object.
- **Speed in cm/s is for sideways motion only**, and is only as accurate as the object width
  you give it.
- **These results are synthetic.** The test pastes pixels from the learning photo, so a real
  webcam will be harder. Real blur depends on exposure time, which is long in dim rooms.
- **The tracking test takes about a minute**, mostly rendering the blurred frames.

## Ideas not tried

1. **A detector that works on blurred objects.** Both feature detectors fail once the tin
   is smeared. A small neural detector (such as a nano YOLO) trained on synthetic blurred
   images of the tin would probably find it and report a confidence, which would remove the
   1-second limit. The cost is training one model per object, where today any photo works.
2. **Lucas-Kanade optical flow on the matched feature points.** Cheap and gives a count of
   surviving points, but its blur robustness has not been measured here.
3. **Real distances including depth.** Calibrating the camera (a printed checkerboard) would
   give 3D position and speed, including motion toward the camera, from the object's width.
4. **Detecting every N frames.** Now affordable (the tracker is 3x cheaper than the
   detector), but it only saves time. It does not help with blur.

## Reproducing

```bash
./scripts/download_models.sh          # the tracker models, once
cmake -S . -B build && cmake --build build
./build/tracking_test orb             # the "pipeline as built" table
ctest --test-dir build --output-on-failure
```

The "which tracker" table can be reproduced by passing a tracker to the test, for example
`./build/tracking_test orb object.jpg test_output models kcf` (`nano`, `vit`, `mil`, `kcf` or
`csrt`). The cost table and the "following alone" table came from temporary benchmark
programs that are not part of the repository.
