# Following the object: design and measurements

**Status:** `find_object` finds the object with a detector (ORB by default) on every frame
and uses OpenCV's `TrackerNano` only when the detector misses. Nano was chosen after
comparing five trackers (see [Why TrackerNano](#why-trackernano)). The design has one known
limit: a moving object that stays motion-blurred for more than about a second is lost
(see [Limits](#limits)).

## What `find_object` does once the object is found

1. **Detect every frame.** `ObjectFinder` returns the object's box and its centroid,
   projected through the homography. The position used for the path and speed is that
   centroid rather than the box center, because a box center moves when a non-round object
   rotates (its bounding box changes size) and the centroid does not. On the tilted test
   views the projected centroid was within 1.4 px of the truth.
2. **Fill in misses.** When the detector misses a frame, `ObjectTracker` starts a
   `TrackerNano` from the last detected box and follows the object until the detector finds
   it again. The box is green when it comes from the detector and yellow when it comes from
   the tracker. The tracker's two model files are loaded once at startup.
3. **Give up after 1 second.** A tracker cannot tell when it has lost the object, so
   tracker-only positions are trusted for at most 1 second.
4. **Keep a path.** `PathHistory` stores timestamped centers. The dotted trail is the last
   3 seconds, with dots every 12 px that fade with age. The path is broken wherever the
   object was lost, so the trail never joins two separate sightings.
5. **Speed.** A straight-line fit over the last 0.4 s of the path, in pixels per second.
   A single-frame difference would be useless: 2 px of detection jitter at 30 fps reads as
   60 px/s. Below 15 px/s it reads 0, and it shows `--` when the object was not seen in the
   last 0.3 s.

Speed is in pixels because a single 2D camera cannot give real distances without knowing
the object's size or distance. It also cannot see motion toward or away from the camera.

## Findings

Measured with `tests/tracking_test.cpp` and some temporary benchmarks (see
[Reproducing](#reproducing)): a synthetic 30 fps video of the tin moving at 250 px/s over a
cluttered background, still for the first 0.6 s (so it is found while sharp, as when someone
holds it up to the camera). Motion blur is rendered by averaging copies of the tin across
the exposure, and each frame gets noise and JPEG compression. All timings are from an
Apple M1 (8 cores, CPU only) with the machine otherwise idle.

### The pipeline as built (detector every frame, TrackerNano on misses)

| Scenario | Object found: detector alone | With tracker | Position error: detector frames | Position error: tracker frames | Median speed error |
|---|---|---|---|---|---|
| Sharp, 250 px/s | 100% | 100% | 0.9 px | n/a | 0.2% |
| 12 px motion blur | 20% | 53% | 0.8 px | 11 px | 3.7% |
| 25 px motion blur | 20% | 53% | 0.9 px | 14 px | 6.3% |
| 50 px motion blur | 19% | 52% | 0.8 px | 22 px | 4.9% |
| Not moving | 100% | 100% | 0.9 px | n/a | reads 0 px/s in 100% of frames |
| Leaves the scene at 1.6 s | n/a | tracked for 0.97 s after it left, then stopped | n/a | n/a | n/a |

Where the detector works, position and speed are accurate. With any motion blur the
detector finds the tin only in the frames before it starts moving (the 20%), and the tracker
carries it for 1 second after that, which is where the 53% comes from.

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
detecting with ORB, and about 10x cheaper than SIFT. That is what makes "detect
occasionally, track in between" worthwhile here; the older trackers (MIL, KCF, CSRT) cost
more than the detector, so for them it was not.

(An earlier run on a heavily loaded machine made every tracker look slower than the
detector. That was machine load; these numbers are from an idle machine.)

### Which tracker? End to end

The same video with each tracker swapped in. Tracker frames are the frames where the
detector missed and the tracker supplied the position. Each cell is mean position error /
median speed error.

| Tracker | 12 px blur | 25 px blur | 50 px blur |
|---|---|---|---|
| TrackerMIL | 10 px / 6% | 10 px / 7% | 18 px / 13% |
| TrackerKCF | 15 px / 9% | 17 px / 10% | 19 px / 11% |
| TrackerCSRT | 23 px / 26% | 6 px / 4% | 22 px / 4% |
| **TrackerNano** (two runs) | 11-14 px / 4-7% | 14 px / 4-6% | 22 px / 5-8% |
| TrackerVit | 7 px / 9% | 8 px / 12% | 11 px / 24% |

"Frames followed" was 49-54% for every tracker, because it is set by the 1-second limit.
Run-to-run variation is a few pixels and several points of speed error (two MIL runs differed
by up to 3.4 px and 5.7 points; two Nano runs by up to 2.6 px and 3.7 points), so small
differences are inside the noise.

### Following alone, with no detector to re-anchor it

Each tracker started from a sharp detection and then followed the object for the rest of the
video (66 frames of movement, about 2.2 s). Counts are frames within 20 px of the true
position.

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

- **Cheap.** 4.2 ms a frame, about 13x cheaper than MIL and 3x cheaper than detecting.
- **Accurate enough where it matters.** Its speed error is 4-8% at every blur level. That
  is better than MIL and KCF at 25 and 50 px of blur and better than ViT at every level.
  CSRT matches it at 25 and 50 px but had 26% at 12 px, and costs about 6x as much.
- **Small.** Two model files, about 1.8 MB together.
- **No tracker offered a usable lost signal**, so the 1-second limit is needed whichever is
  used. The neural trackers' scores do not separate "object in view" from "object gone", and
  the one tracker that noticed the object leaving (KCF) is the worst at following it.

ViT is cheaper still and has the best position accuracy, but its speed error under heavy
blur (12-24%) is the worst, and the speed readout is what this tracker's positions feed.

**Trade-off:** Nano needs its model files at runtime. They are not in the repository (the
repository the OpenCV documentation links to shows no license), so
`scripts/download_models.sh` fetches them into `models/`.

## Limits

- **A moving, blurred object is followed for 1 second at most.** The detector fails on it,
  and nothing can tell "blurred" from "gone". Tracker scores do not separate them (Nano's
  score was 0.8-0.9 whether the tin was in view or had left), and the detector fails in both
  cases. A longer limit would follow the background after the object leaves.
- **Speed is noisier on tracker frames.** With Nano, tracker positions are 11-22 px off,
  where detector positions are within about 1 px.
- **These results are synthetic.** The test pastes pixels from the learning photo, so a real
  webcam will be harder. Real blur depends on exposure time, which is long in dim rooms.
  Results also vary a little from run to run, because the object is learned with GrabCut,
  which is not exactly repeatable.
- **The tracking test takes about a minute**, mostly rendering the blurred frames.

## Ideas not tried

1. **A detector that works on blurred objects.** Both feature detectors fail once the tin
   is smeared. A small neural detector (such as a nano YOLO) trained on synthetic blurred
   images of the tin would probably find it and report a confidence, which would remove the
   1-second limit. The cost is training one model per object, where today any photo works.
2. **Lucas-Kanade optical flow on the matched feature points.** Cheap and gives a count of
   surviving points, but its blur robustness has not been measured here.
3. **Detecting every N frames.** Now affordable (the tracker is 3x cheaper than the
   detector), but it only saves time. It does not help with blur.

## Reproducing

```bash
./scripts/download_models.sh          # the tracker models, once
cmake -S . -B build && cmake --build build
./build/tracking_test orb             # the "pipeline as built" table
ctest --test-dir build --output-on-failure
```

The tracker comparison, the cost table and the "following alone" table came from temporary
builds that swapped the tracker for MIL, KCF, CSRT or ViT (the `opencv_tracking` module, and
the ViT model from OpenCV's model zoo); they are not part of the repository.
