# Validation Checklist

## Objective Coverage

| Requirement | Evidence artifact | Status |
|---|---|---|
| Produce actual camera-only 3D reconstruction | documented final stage: 1,508,944 points | Satisfied |
| Provide viewable result | generated local viewer after pipeline rebuild | Satisfied |
| Prove primary path did not consume `lidar01` | manifests and no-LiDAR reports state `lidar_used=false` | Satisfied |
| Main input is vehicle-mounted RGB | 1176 RGB frames from 4 vehicles x 6 cameras | Satisfied |
| Use calibration for metric placement | camera/world placement and vehicle replay use DeepAccident calibration | Satisfied |
| Static 3D reconstruction from RGB | SLAM3R camera-ray-depth calibrated point fusion | Satisfied |
| Preserve good base and add context incrementally | 4 cumulative stages, 1,508,944 final points | Satisfied |
| Calibration-constrained dynamic replay | documented four-agent replay tracks | Satisfied |
| Collision/replay diagnostic | documented closest proxy frame and clearance | Satisfied |
| Vehicle geometry not overclaimed | docs describe car-like proxies | Satisfied |
| LiDAR moved to reference role | README, method summary, artifacts docs | Satisfied |

## Verification Gates

Run lightweight checks:

```bash
python3 -m py_compile scripts/*.py research_camera_only/scripts/*.py
python3 -m unittest discover -s tests
```

Inspect generated viewer after rerunning the pipeline:

```bash
python3 scripts/serve_viewer.py --port 8132
```

```text
http://127.0.0.1:8132/outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html
```

## Residual Risks

- Generated GLB meshes, preview images, viewer files, raw dataset, and heavy
  SLAM3R outputs/checkpoints are intentionally not committed.
- DeepAccident calibration is still required for metric placement and vehicle
  replay.
- Low-texture asphalt and distant background regions are visually weaker than
  LiDAR-assisted references.
- Proxy clearance is a structured replay diagnostic, not a physical crash-force
  model.
- Broader real-world validation remains future work.
