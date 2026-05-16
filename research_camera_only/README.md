# Camera-Only ReSceneScribe Research Track

This folder documents the completed camera-centered research track for the final
ReSceneScribe submission:

> ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from
> Vehicle-Mounted Sensing and Multi-Agent Simulation for Evidence-Grounded
> Investigation

The current main claim is that vehicle-mounted RGB observations, combined with
calibration, can produce an interpretable static accident-scene reconstruction
and a calibration-constrained multi-agent replay. Dynamic vehicles are
represented as dimension-aware proxies for evidence-grounded simulation and
diagnostics.

## Completed Result

The successful execution is the DeepAccident Town04 type1-subtype2 case:

| Item | Value |
|---|---:|
| Scenario | `Town04_type001_subtype0002_scenario00017` |
| Vehicle agents | 4 |
| RGB cameras per agent | 6 |
| RGB frames | 1176 |
| Base camera-only points | 545,306 |
| Final incremental points | 1,508,944 |
| Vehicle replay samples | 49 per vehicle |
| Primary LiDAR use | false |

Generated local output:

```text
outputs/town04_type1_subtype2_slam3r_incremental_layers/
```

After rebuilding, open the generated viewer:

```bash
python3 scripts/serve_viewer.py --port 8132
```

```text
http://127.0.0.1:8132/outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html
```

## Evidence Boundary

Used by the main path:

- DeepAccident RGB camera frames,
- DeepAccident calibration for camera/world placement and replay,
- dynamic masks when available,
- SLAM3R RGB point-map prediction and confidence maps,
- vehicle dimensions and frame-wise calibration poses.

Not used by the main path:

- `lidar01` point clouds,
- legacy LiDAR PLY/GLB assets,
- simulator map meshes,
- proxy meshes as background geometry.

Vehicles are represented by car-like calibrated proxies. This preserves pose,
size, direction, and replay readability without overclaiming full video-derived
vehicle surface reconstruction.

## Paper Evaluation Roles

The paper includes a physical pilot with two real vehicles and windshield-area
RGB cameras. That pilot checks capture feasibility for vehicle-mounted visual
evidence, but it is not used as collision-dynamics or liability validation.

The quantitative reconstruction and replay results in this repository come from
the controlled DeepAccident Town04 benchmark because it provides synchronized
RGB observations, calibration, vehicle dimensions, and frame-wise poses.

## Artifact Map

| Artifact | Purpose |
|---|---|
| `lidar_claim_audit.md` | Audit of old LiDAR-dependent repo/paper claims. |
| `technology_decision_record.md` | Component decision record for RGB reconstruction backends and fallbacks. |
| `pipeline_design.md` | End-to-end camera-only pipeline and evidence contracts. |
| `experiment_protocol.md` | Dataset, masking, reconstruction, scale, replay, and failure protocol. |
| `implementation_skeleton.md` | Practical scripts, integration points, and expected output tree. |
| `schemas/` | JSON schema contracts for masks, cameras, scale, tracks, diagnostics, and evidence bundles. |
| `reports/evidence_report.md` | Evidence history and final output pointers. |
| `reports/completion_audit.md` | Current completion audit for the camera-only result. |
| `reports/validation_checklist.md` | Verification gates and residual risks. |

## Supplementary Readability Material

The historical `viewer_assets/` and Town03 hybrid PLY are LiDAR-derived. They
can be discussed only as supplementary readability/reference visualizations for
road or city context. They are not the current camera-only evidence layer and
are not used for the reported Town04 metrics.
