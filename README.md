# ReSceneScribe

Static-dynamic 3D accident reconstruction from vehicle-mounted cameras and
multi-agent simulation for evidence-grounded investigation.

This repository accompanies the conference manuscript:

> ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from
> Vehicle-Mounted Cameras and Multi-Agent Simulation for Evidence-Grounded
> Investigation

## What is in this repository

ReSceneScribe packages the successful DeepAccident Town03 four-vehicle
collision reconstruction outputs:

- an interactive Three.js replay of the four-vehicle accident process,
- scripts that rebuild the replay from a local DeepAccident mini dataset,
- scripts that rebuild the clean hybrid Gaussian/RGB PLY,
- paper-ready methodology notes and the final PDF artifact,
- checksums and verification tools for the large reconstruction assets.

The repository intentionally does not include the raw DeepAccident dataset.
Users who want to rebuild the assets must download or mount their own
DeepAccident-compatible dataset copy.

## Live / local viewer

After cloning, the replay can be opened through any local HTTP server:

```bash
python3 -m http.server 8132
```

Then open:

```text
http://127.0.0.1:8132/viewer/index.html?quality=ultra&trail
```

The viewer included in git contains:

- 56 synchronized replay frames,
- 4 vehicle trajectories,
- a 1.05M-point regular static LiDAR GLB,
- a 2.2M-point ultra static LiDAR GLB,
- synchronized front dashcam panels for all four vehicle agents.

## Key scenario facts

| Item | Value |
|---|---:|
| Scenario | `Town03_type001_subtype0001_scenario00024` |
| Weather | `MidRainSunset` |
| Available frames | `001`-`056` |
| Collision overlap frames | `054`-`056` |
| Selected collision-state frame | `056` |
| Final OBB gap | `-0.447944 m` |
| Final center distance | `4.229926 m` |
| Viewer regular / ultra background | `1,050,000` / `2,200,000` points |
| Dynamic vehicle points removed from viewer background | `312,543` |
| Final hybrid PLY vertices | `1,177,009` |

## Repository layout

```text
.
├── README.md
├── QUICKSTART.md
├── Makefile
├── requirements.txt
├── artifacts/
│   └── manifest.json
├── docs/
│   ├── artifacts.md
│   ├── dataset.md
│   ├── method_summary.md
│   └── research_success_results_ko.md
├── paper/
│   ├── ReSceneScribe_IEEE.tex
│   └── final_pdf/
├── previews/
│   └── four_vehicle_topdown_plan.png
├── scripts/
│   ├── build_four_vehicle_collision_viewer.py
│   ├── build_town03_clean_hybrid_gaussian_ply.py
│   ├── prepare_town03_3dgs_dataset.py
│   ├── prepare_town03_3dgs_masked_static_dataset.py
│   ├── serve_viewer.py
│   └── verify_ply.py
├── viewer/
├── viewer_assets/
└── viewer_frames/
```

## Rebuild from DeepAccident

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the dataset root:

```bash
export DEEPACCIDENT_ROOT=/path/to/deepaccident_mini_dataset
```

Rebuild the web replay:

```bash
python scripts/build_four_vehicle_collision_viewer.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --output-root .
```

Rebuild the collision-state hybrid Gaussian/RGB PLY:

```bash
python scripts/build_town03_clean_hybrid_gaussian_ply.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --out outputs/town03_4dashcam_collision_3dgs_45000.ply \
  --frame-start 1 \
  --frame-end 56 \
  --static-limit 1100000 \
  --vehicle-limit-each 120000 \
  --static-voxel 0.050 \
  --vehicle-voxel 0.025 \
  --stats outputs/town03_4dashcam_collision_3dgs_45000.stats.json
```

## Large artifact handling

The final PLY is about 281.74 MiB, which is larger than GitHub's normal
single-file git limit. It is published as a GitHub Release asset rather than
tracked directly in git.

See [docs/artifacts.md](docs/artifacts.md) for checksums, expected file names,
and verification commands.

## Paper materials

The paper source is in [paper/ReSceneScribe_IEEE.tex](paper/ReSceneScribe_IEEE.tex).
The generated final PDF artifact is archived under [paper/final_pdf](paper/final_pdf).

## Citation

If you use this repository, cite the accompanying ReSceneScribe manuscript and
the DeepAccident benchmark:

```bibtex
@inproceedings{lee2026rescenescribe,
  title     = {ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from Vehicle-Mounted Cameras and Multi-Agent Simulation for Evidence-Grounded Investigation},
  author    = {Lee, Woosup and Kim, Yongmin},
  year      = {2026}
}
```

## Data and license note

Raw DeepAccident data are not redistributed here. Follow the DeepAccident
dataset terms for data access and citation. A source-code license has not been
selected in this repository yet; contact the authors before reuse beyond
review, reproduction, or evaluation of the accompanying manuscript.
