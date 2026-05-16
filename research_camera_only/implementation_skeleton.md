# Implementation Skeleton

This document defines the execution contract for the camera-only reconstruction
track. The current scripts provide smoke-test support, but a skeleton alone is
not completion. Completion requires an actual 3D artifact under
`outputs/camera_only_reconstruction/`.

## Included Scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_rgb_demo.py` | Build a small RGB-only demo set from a user-provided `viewer_frames/` folder or a DeepAccident-style RGB camera folder. Can create placeholder masks for schema smoke tests. |
| `scripts/validate_research_artifacts.py` | Verify durable docs/schemas exist and optionally validate a mask manifest without external dependencies. |
| `scripts/compute_obb_diagnostics.py` | Compute frame-wise 2D ground-plane OBB gaps/min distances from proxy vehicle tracks. |

## Required Reconstruction Integrations

Wrappers required for completion, depending on installed backends:

```text
scripts/run_grounded_sam2_masks.py
scripts/run_sam2_video_masks.py
scripts/run_vggt_reconstruction.py
scripts/run_colmap_check.py
scripts/run_megasam_fallback.py
scripts/run_mast3r_slam_fallback.py
scripts/run_depth_anything_support.py
scripts/build_camera_only_viewer.py
```

The wrappers should write the output tree described in `pipeline_design.md` and
must never require `lidar01` for the real-world method.

If a wrapper cannot run because of missing dependencies, GPU limits, checkpoint
access, or license constraints, the failed command and fallback decision must be
recorded in `outputs/camera_only_reconstruction/reports/quality_report.md`.

## Minimal Demo Contract

`prepare_rgb_demo.py` writes:

```text
demo_outputs/rgb_demo/
├── rgb_frames/<agent>/frame_0000.jpg
├── masks/<agent>/frame_0000.png      # placeholder only when requested
└── mask_manifest.json
```

The manifest declares `source.lidar_used=false`. Placeholder masks are marked
with `mask_status=placeholder_empty` and `qa_status=not_evidence`.

This contract is only a smoke test. It proves that RGB-only paths and manifests
exist, not that 3D reconstruction succeeded.

## Reconstruction Backend Contract

VGGT or fallback backends must write the equivalent of:

```text
outputs/camera_only_reconstruction/reconstruction/cameras.json
outputs/camera_only_reconstruction/reconstruction/points.ply
outputs/camera_only_reconstruction/reconstruction/depth/
outputs/camera_only_reconstruction/reconstruction/reprojection_report.json
outputs/camera_only_reconstruction/reconstruction/scale_alignment.json
outputs/camera_only_reconstruction/scene.glb
```

`reprojection_report.json` must include:

- backend name and version/commit,
- input manifest path,
- registered frame count,
- failed frame list,
- mean/median reprojection error when available,
- confidence/quality summary,
- known failure notes.

`reports/no_lidar_audit.md` must include:

- command used for the reconstruction stage,
- input directories,
- explicit statement that `lidar01` was not consumed,
- explicit statement that legacy `viewer_assets/*.glb` and historical PLY files
  were not used as geometry input,
- any automated search or wrapper guard used to enforce the rule.

`reports/quality_report.md` must include:

- backend attempts and versions,
- output artifact paths and sizes,
- registered frame count or equivalent coverage metric,
- point count, mesh count, or backend-native density metric,
- visual inspection notes,
- reprojection/consistency metrics when available,
- failure reasons and fallback decisions.

## Replay Contract

Vehicle replay should write:

```text
replay/vehicle_dimensions.json
replay/vehicle_tracks.json
replay/collision_diagnostics.json
```

`compute_obb_diagnostics.py` accepts a combined track JSON with vehicles,
dimensions, and frame-wise `[x, y, yaw_rad]` poses. It computes 2D OBB gaps in
the ground plane and writes collision diagnostics suitable for the evidence
report. This deliberately avoids claiming full dynamic surface reconstruction.
