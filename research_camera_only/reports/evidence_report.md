# Evidence Report

Report date: 2026-05-16.

Status: current camera-only Town04 reconstruction completed and documented.

## Current Evidence

The successful no-LiDAR evidence is documented under:

- `docs/camera_only_success_results_full_ko.md`
- `docs/method_summary.md`
- `research_camera_only/reports/completion_audit.md`

The generated local viewer path after rerunning the pipeline is:

```text
outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html
```

## Evidence Boundary

Inputs used:

- DeepAccident RGB camera frames,
- DeepAccident calibration files,
- camera intrinsic/extrinsic and camera/world poses,
- dynamic masks where available,
- SLAM3R RGB point-map predictions and confidence maps,
- vehicle dimensions and calibration poses for replay.

Inputs not used by the primary reconstruction path:

- `lidar01` point clouds,
- historical LiDAR-derived PLY/GLB outputs,
- simulator map mesh,
- proxy meshes as background geometry.

## Reconstruction Results

| Metric | Value |
|---|---:|
| Scenario | `Town04_type001_subtype0002_scenario00017` |
| Agents | 4 |
| Cameras per agent | 6 |
| RGB views | 1176 |
| Base points | 545,306 |
| Final incremental points | 1,508,944 |
| Added points | 963,638 |
| Base accepted streams | 10 / 24 |
| Base alignment RMSE | 1.4469 m |
| Vehicle replay samples | 49 per vehicle |
| Closest proxy frame | 49 |
| Proxy clearance XY | -0.6842 m |
| LiDAR point cloud used | false |

## Accident Evidence Contract

Accident claims should include:

- vehicle IDs,
- frame index or interval,
- camera/video source,
- mask source and QA status when masks are used,
- static reconstruction source and quality,
- calibrated pose or estimated pose with uncertainty,
- scale-alignment source and residual,
- OBB overlap/gap or minimum distance,
- validation status and known limitations.

## Interpretation

The current result supports a camera-first accident reconstruction workflow with
calibrated replay. It should not be described as a perfect reconstruction of all
vehicle surfaces or as a physical crash-force model. Vehicle geometry is a
dimension-aware proxy for replay readability.

LiDAR-assisted renderings remain useful as optional communication references for
road and city context, but they are not the main reconstruction sensor in the
current repository.
