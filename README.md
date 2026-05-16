# ReSceneScribe

Static-dynamic 3D accident reconstruction from vehicle-mounted RGB evidence,
calibration-aware scene placement, and multi-agent replay simulation.

This repository tracks the implementation surface for the final ReSceneScribe
submission:

> ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from
> Vehicle-Mounted Sensing and Multi-Agent Simulation for Evidence-Grounded
> Investigation

## What Is New

The main branch has been reorganized around the successful DeepAccident Town04
dashcam/RGB reconstruction experiment. The primary reconstruction path uses:

- vehicle-mounted RGB frames,
- DeepAccident calibration for camera/world placement and replay,
- dynamic masks when available,
- SLAM3R RGB point-map prediction,
- conservative incremental point layering,
- calibrated car-like proxy vehicles for the dynamic accident replay.

The primary path does not read `lidar01` point clouds, historical LiDAR PLY/GLB
outputs, simulator map meshes, or legacy viewer assets as reconstruction
geometry.

## Paper Alignment

The final paper presents ReSceneScribe as a vehicle-mounted RGB evidence
workflow. It separates two evaluation roles:

- a physical pilot with two real vehicles and windshield-area RGB cameras,
  used to check whether vehicle-mounted visual capture can support static
  accident-scene reconstruction;
- the DeepAccident Town04 benchmark evaluation, used for calibrated metric
  alignment, static scene reconstruction, multi-agent replay, and the reported
  proxy-clearance diagnostic.

The physical pilot is a capture-feasibility study rather than a collision
dynamics or liability validation. The quantitative replay results below come
from the controlled DeepAccident benchmark because it provides synchronized
vehicle-mounted RGB observations, calibration, and vehicle metadata.

## Main Result

| Item | Value |
|---|---:|
| Dataset split | `type1_subtype2_accident` |
| Scenario | `Town04_type001_subtype0002_scenario00017` |
| Agents | `4` vehicles |
| Cameras | `6` RGB cameras per vehicle |
| RGB frames | `1176` |
| Base backend | `slam3r_camera_ray_depth_calibrated` |
| Base point count | `545,306` |
| Incremental final point count | `1,508,944` |
| Added points through layering | `963,638` |
| Base accepted streams | `10 / 24` |
| Base alignment RMSE | `1.4469 m` |
| Vehicle replay samples | `49` per vehicle |
| Closest proxy diagnostic | frame `49`, clearance `-0.6842 m` |
| LiDAR point cloud used in primary path | `false` |

## Rebuild Or Inspect The Camera-Only Result

This repository does not track generated GLB meshes, preview images, or paper
files. The successful Town04 result is documented in `docs/` and
`research_camera_only/`; the heavy visual artifacts are regenerated locally from
the pipeline.

The GitHub Pages entry point is a lightweight project landing page:

```text
https://woosopyi.github.io/ReSceneScribe/
```

After regenerating the outputs, serve the local viewer:

```bash
python3 scripts/serve_viewer.py --port 8132
```

Then open:

```text
http://127.0.0.1:8132/outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html
```

The generated viewer shows the cumulative camera-only point scene, additive
layers, four calibrated vehicle tracks, and car-like proxy vehicles. A
supplementary readability view may be regenerated locally to make sparse road
or urban context easier to inspect, but it is qualitative only and is not used
for the reported reconstruction counts or replay diagnostic. Raw DeepAccident
data, generated GLB/image artifacts, and heavy model checkpoints are not
included in git.

## Repository Layout

```text
.
├── README.md
├── QUICKSTART.md
├── Makefile
├── docs/
│   ├── method_summary.md
│   ├── dataset.md
│   ├── artifacts.md
│   ├── camera_only_success_results_full_ko.md
│   └── research_success_results_ko.md
├── research_camera_only/
│   ├── README.md
│   ├── pipeline_design.md
│   ├── reports/
│   └── schemas/
├── scripts/
│   ├── run_multicam_world_reconstruction.py
│   ├── run_slam3r_deepaccident_reconstruction.py
│   ├── build_slam3r_incremental_layers.py
│   └── run_town04_camera_only_final_pipeline.py
└── viewer/
    └── vendor/three/
```

## Rebuild From DeepAccident

The rebuild path expects a local DeepAccident-compatible copy. Raw data are not
redistributed in this repository.

```bash
python3 -m venv .venv-camera-only
. .venv-camera-only/bin/activate
python -m pip install -r requirements.txt
export DEEPACCIDENT_ROOT=/path/to/deepaccident_mini_dataset
```

Prepare the known-pose RGB/mask export:

```bash
python scripts/run_multicam_world_reconstruction.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --category type1_subtype2_accident \
  --scenario Town04_type001_subtype0002_scenario00017 \
  --out outputs/town04_type1_subtype2_multicam_export \
  --frame-start 1 \
  --frame-end 49 \
  --frame-step 1
```

Run the SLAM3R camera-only base reconstruction:

```bash
python scripts/run_slam3r_deepaccident_reconstruction.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --category type1_subtype2_accident \
  --scenario Town04_type001_subtype0002_scenario00017 \
  --mask-export outputs/town04_type1_subtype2_multicam_export \
  --out outputs/town04_type1_subtype2_slam3r_reconstruction \
  --slam3r-root third_party/SLAM3R
```

Build the incremental layers:

```bash
python scripts/build_slam3r_incremental_layers.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --source outputs/town04_type1_subtype2_slam3r_reconstruction \
  --mask-export outputs/town04_type1_subtype2_multicam_export \
  --out outputs/town04_type1_subtype2_slam3r_incremental_layers \
  --slam3r-root third_party/SLAM3R
```

SLAM3R execution requires a CUDA-capable environment and an installed SLAM3R
checkout. Generated GLB meshes, preview images, and viewer outputs are kept out
of git and can be rebuilt with the commands above.

## Supplementary Readability Material

Earlier ReSceneScribe artifacts reconstructed a Town03 scene with DeepAccident
LiDAR-camera fusion and exported a hybrid Gaussian/RGB PLY. Those files remain
legacy material and are not tracked in the current camera-centered repository
surface. When referenced, they should be described only as supplementary
readability material for road or urban context, not as the primary source of the
reported Town04 reconstruction or replay metrics.

## Citation

If you use this repository, cite this software repository and the DeepAccident
benchmark:

```bibtex
@inproceedings{lee2026rescenescribe,
  title     = {ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from Vehicle-Mounted Sensing and Multi-Agent Simulation for Evidence-Grounded Investigation},
  author    = {Lee, Woosup and Kim, Yongmin},
  year      = {2026}
}
```

## AI-Assisted Drafting Declaration

The final submission declares that generative AI assistance was used only to
support visual drafting. The authors reviewed and finalized the figure and take
responsibility for the scientific content of the work.

## Data And License Note

Raw DeepAccident data, heavy model checkpoints, local virtual environments, and
large training outputs are not redistributed here. Follow the DeepAccident,
SLAM3R, and any model-specific terms for data access and citation. A source-code
license has not been selected in this repository yet; contact the authors before
reuse beyond review, reproduction, or evaluation.
