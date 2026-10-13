# Models

`find_object` follows the object between detections with a tracker, chosen with
`--tracker=NAME`. Two of the trackers use small neural-network files in this folder:

| `--tracker=` | Files | Size |
|---|---|---|
| `nano` (default) | `nanotrack_backbone_sim.onnx`, `nanotrack_head_sim.onnx` | about 1.0 MB and 0.7 MB |
| `vit` | `object_tracking_vittrack_2023sep.onnx` | about 0.7 MB |
| `mil`, `kcf`, `csrt` | none | |

They are not stored in the repository. Download them once:

```bash
./scripts/download_models.sh
```

The Nano files come from the NanoTrack models repository that OpenCV's documentation links to
(https://github.com/HonglinChu/SiamTrackers), which shows no license. The ViT file comes from
OpenCV's model zoo (https://github.com/opencv/opencv_zoo, Apache 2.0). To keep the files
somewhere else, pass `--models=DIR` to `find_object`.
