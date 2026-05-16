# Quickstart

## Rebuild The Final Submission Result

```bash
git clone https://github.com/WoosopYi/ReSceneScribe.git
cd ReSceneScribe
```

The repository tracks code, Korean/English research summaries, and lightweight
validation structure. Paper files, generated GLB meshes, and preview images are
not committed.

The repository is aligned with the final paper:

> ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from
> Vehicle-Mounted Sensing and Multi-Agent Simulation for Evidence-Grounded
> Investigation

## What The Documented Result Contains

| Artifact | Path |
|---|---|
| Main research summary | `docs/camera_only_success_results_full_ko.md` |
| Method summary | `docs/method_summary.md` |
| Camera-only research track | `research_camera_only/README.md` |
| Evidence report | `research_camera_only/reports/evidence_report.md` |
| Pipeline implementation | `scripts/run_town04_camera_only_final_pipeline.py` |

## Rebuild The Camera-Centered Pipeline

Install baseline Python dependencies:

```bash
python3 -m venv .venv-camera-only
. .venv-camera-only/bin/activate
python -m pip install -r requirements.txt
```

Set the local DeepAccident root:

```bash
export DEEPACCIDENT_ROOT=/path/to/deepaccident_mini_dataset
```

Prepare RGB frames, masks, calibration exports, and replay metadata:

```bash
make camera-export
```

Run the SLAM3R camera-only base reconstruction:

```bash
make camera-slam3r
```

Build the incremental cumulative scene:

```bash
make camera-layers
```

The heavy reconstruction path requires CUDA and a local `third_party/SLAM3R`
checkout. After the pipeline finishes, serve the generated viewer locally:

```bash
python3 scripts/serve_viewer.py --port 8132
```

Then open:

```text
http://127.0.0.1:8132/outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html
```

## Legacy Town03 LiDAR Path

The older Town03 LiDAR-camera fusion scripts remain available under explicit
legacy make targets:

```bash
make legacy-rebuild-viewer
make legacy-rebuild-ply
```

Those outputs are supplementary readability/reference material only. They are
not the main camera-centered claim of the current repository and are not used
for the reported Town04 reconstruction or replay metrics.
