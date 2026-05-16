# Artifacts

## Camera-Only Result Policy

The current repository tracks the implementation and documentation surface for
the final ReSceneScribe submission. It includes the successful Town04
camera-centered method, research summaries, validation scaffolding, and
reproduction scripts. Paper files, generated GLB meshes, preview images, and
viewer outputs are intentionally not committed.

| File | Purpose |
|---|---|
| `docs/camera_only_success_results_full_ko.md` | Detailed successful-result record |
| `docs/method_summary.md` | Method summary |
| `research_camera_only/README.md` | Camera-only research track overview |
| `research_camera_only/reports/` | Validation and evidence notes |
| `scripts/run_town04_camera_only_final_pipeline.py` | End-to-end pipeline entry point |

After local regeneration, open the generated viewer:

```bash
python3 scripts/serve_viewer.py --port 8132
```

```text
http://127.0.0.1:8132/outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html
```

## Camera-Only Result Metrics

| Metric | Value |
|---|---:|
| Base points | 545,306 |
| Final cumulative points | 1,508,944 |
| Added points | 963,638 |
| Stages | 4 |
| Additive layers | 3 |
| RGB views | 1176 |
| Vehicle tracks | 4 |
| Track samples per vehicle | 49 |
| LiDAR point cloud used | false |
| Legacy LiDAR assets used | false |

## Files Not Tracked

The following are intentionally excluded from git:

- raw DeepAccident data,
- `deepaccident_mini_dataset_download/`,
- `.venv-camera-only/`,
- `third_party/SLAM3R` and other third-party checkouts,
- full training logs and checkpoints,
- generated `outputs/` subtrees including GLB meshes, viewer files, replay JSON,
  diagnostics JSON, and preview images,
- `paper/` manuscripts, PDFs, and translation files,
- tracked image files such as `.png`, `.jpg`, `.jpeg`, and `.webp`,
- tracked `.glb` scene files,
- model weights such as YOLO `.pt` files.

## Supplementary Readability And Legacy Artifacts

The old Town03 LiDAR-camera viewer assets remain in git history only. They are
not part of the current tracked repository surface:

| File | Legacy role |
|---|---|
| `viewer_assets/four_vehicle_static_lidar_background.glb` | Town03 static LiDAR background |
| `viewer_assets/four_vehicle_static_lidar_background_ultra.glb` | Town03 high-density static LiDAR background |
| `town03_4dashcam_collision_3dgs_45000.ply` | Legacy LiDAR/RGB hybrid release artifact |

These files are not the main method in the current repository. When referenced,
they should be described only as supplementary readability/reference material
from the earlier study. They are not inputs to the reported Town04 point counts,
alignment statistics, replay tracks, or proxy-clearance diagnostic.
