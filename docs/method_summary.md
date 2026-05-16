# Method Summary

ReSceneScribe is now organized around a camera-first accident reconstruction
path. The method produces two linked outputs:

1. A static RGB-derived 3D accident environment.
2. A calibration-constrained replay with dimension-aware vehicle proxies.

The previous Town03 LiDAR-camera fusion result is retained only as legacy
readability/reference material.

## Paper Scope

The final paper frames ReSceneScribe as a static-dynamic accident reconstruction
workflow from vehicle-mounted RGB evidence and calibration-aware multi-agent
simulation. The repository therefore documents the successful camera-centered
path and keeps generated papers, images, GLB files, and raw datasets outside
git.

The paper separates two experimental roles:

- a physical pilot with two real vehicles and windshield-area RGB cameras,
  used as a capture-feasibility check for vehicle-mounted visual evidence;
- the DeepAccident Town04 benchmark, used for metric reconstruction, calibrated
  vehicle replay, and the reported quantitative diagnostic.

The physical pilot is not used to claim collision dynamics, accident liability,
or crash-force estimation. Those interpretations are outside the evidence
boundary of this repository.

## Camera-Only Static Reconstruction

The primary input is synchronized vehicle-mounted RGB video. In the successful
Town04 experiment, the pipeline uses:

- 4 vehicle agents,
- 6 RGB cameras per vehicle,
- 49 frames per camera stream,
- 1176 total RGB views,
- DeepAccident calibration for camera/world placement.

SLAM3R predicts RGB point maps and confidence maps from camera streams. The
accepted points are placed into the shared DeepAccident world through calibrated
camera poses and stream scale alignment. Dynamic masks are used when available;
unmasked frames are admitted only under stricter confidence and alignment
filters.

The primary path does not read:

- `lidar01` point clouds,
- historical LiDAR-derived PLY/GLB assets,
- simulator map meshes,
- proxy meshes as background geometry.

## Metric Alignment

DeepAccident calibration is used as a metric prior. For a camera ray associated
with an RGB point-map prediction, the implementation places the point with the
calibrated camera-to-world transform:

```text
p_world = camera_to_world @ camera_ray_depth_from_rgb_prediction
```

The calibration files contain matrices with names such as `lidar_to_Camera_*`.
In this camera-only path those matrices are used to recover camera extrinsics;
LiDAR point geometry itself is not loaded.

## Incremental Layering

The base camera-only reconstruction is preserved as an immutable good result.
Additional layers reuse saved SLAM3R RGB point maps and calibration, but a new
point is accepted only if its 0.08 m voxel is not already occupied in the
cumulative scene.

Successful cumulative stages:

| Stage | New points | Cumulative points |
|---|---:|---:|
| `00_base` | 545,306 | 545,306 |
| `01_01_masked_dense` | 541,877 | 1,087,183 |
| `02_02_all_aligned_frames` | 303,076 | 1,390,259 |
| `03_03_strict_extra_streams` | 118,685 | 1,508,944 |

This strategy expands urban and road context without damaging the reliable
base layer.

## Vehicle Replay

Dynamic vehicles are not claimed as fully reconstructed video-derived surfaces.
They are rendered as car-like calibrated proxies driven by frame-wise
`ego_to_world` poses and vehicle dimensions from the dataset context.

The proxy model contains body, hood, trunk, cabin, windows, bumpers, wheels,
lights, and trim. Its purpose is to make the accident process readable while
preserving pose, scale, and direction.

## Diagnostics

The diagnostic layer reports structured replay context:

```text
closest frame: 49
closest pair: ego_vehicle_behind / other_vehicle
center distance XY: 4.1956 m
proxy clearance XY: -0.6842 m
```

A negative proxy clearance means the simplified vehicle footprints overlap in
the calibrated replay. It is an interpretation diagnostic, not a physical
crash-force model or legal proof of impact.

## Supplementary Readability View

A supplementary readability view can be regenerated locally to make sparse road,
lane, or urban context easier to inspect. This view is qualitative
communication support only. It is not used for the reported point counts,
alignment statistics, closest-frame diagnostic, or proxy-clearance metric.

Legacy LiDAR-assisted renderings from the earlier Town03 study may be discussed
only in this supplementary readability role. The deployment claim and primary
Town04 reconstruction path remain grounded in vehicle-mounted RGB videos and
DeepAccident calibration.
