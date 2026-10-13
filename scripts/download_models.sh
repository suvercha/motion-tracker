#!/bin/sh
# Downloads the TrackerNano model files into models/ (about 1.8 MB).
# Source: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
# (the link OpenCV documents for TrackerNano). Safe to run again; existing files are kept.
set -eu

dir="$(cd "$(dirname "$0")/.." && pwd)/models"
base="https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2"
mkdir -p "$dir"

for name in nanotrack_backbone_sim.onnx nanotrack_head_sim.onnx; do
    if [ -s "$dir/$name" ]; then
        echo "have  $name"
        continue
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
done
echo "done: $dir"
