# Models

`find_object` follows the object between detections with OpenCV's `TrackerNano`, which
needs two small neural-network files in this folder:

- `nanotrack_backbone_sim.onnx` (about 1.0 MB)
- `nanotrack_head_sim.onnx` (about 0.7 MB)

They are not stored in the repository. Download them once:

```bash
./scripts/download_models.sh
```

The script fetches them from the NanoTrack models repository that OpenCV's documentation
links to (https://github.com/HonglinChu/SiamTrackers). To keep the files somewhere else,
pass `--models=DIR` to `find_object`.
