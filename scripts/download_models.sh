#!/bin/sh
# Downloads the tracker model files into models/ (about 2.5 MB):
#   - the TrackerNano models (--tracker=nano, the default), from the NanoTrack models
#     repository that OpenCV's documentation links to:
#     https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
#   - the TrackerVit model (--tracker=vit), from OpenCV's model zoo (Apache 2.0):
#     https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
# The other trackers (mil, kcf, csrt) need no files. Safe to run again; existing files are kept.
set -eu

dir="$(cd "$(dirname "$0")/.." && pwd)/models"
nano_base="https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2"
vit_base="https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack"
mkdir -p "$dir"

fetch() {
    base="$1"
    name="$2"
    if [ -s "$dir/$name" ]; then
        echo "have  $name"
        return
    fi
    echo "fetch $name"
    curl -fL --silent --show-error -o "$dir/$name.part" "$base/$name"
    # A Git LFS stub or an error page is a few hundred bytes; the real models are far larger.
    if [ "$(wc -c < "$dir/$name.part")" -lt 100000 ]; then
        rm -f "$dir/$name.part"
        echo "error: $name downloaded too small to be the model" >&2
        exit 1
    fi
    mv "$dir/$name.part" "$dir/$name"
}

fetch "$nano_base" nanotrack_backbone_sim.onnx
fetch "$nano_base" nanotrack_head_sim.onnx
fetch "$vit_base" object_tracking_vittrack_2023sep.onnx
echo "done: $dir"
