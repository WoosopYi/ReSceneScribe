# Camera-Only Pipeline Design

## Target Output

The camera-only pipeline must produce:

- masked static RGB frames,
- an actual static 3D background reconstruction from RGB,
- camera parameters and reprojection/registration diagnostics,
- metric scale alignment report,
- calibrated vehicle proxy tracks,
- OBB overlap or minimum-distance collision diagnostics,
- a viewable 3D export or viewer,
- evidence-grounded reports with frame IDs, vehicle IDs, masks, pose, geometry,
  and uncertainty.

LiDAR is not an input to the real-world method. DeepAccident calibration can be
used only as a prototype pose prior or validation reference.

Plans, placeholder masks, and schemas are not enough. The current
completed artifact is packaged under
`outputs/town04_type1_subtype2_slam3r_incremental_layers/`; future real-world
cases can reuse the generic `outputs/camera_only_reconstruction/` template.

## Data Flow

```text
RGB videos / RGB frame folders
  -> frame selection and synchronization
  -> Grounded SAM 2 / SAM 2 dynamic-object masks
  -> masked static frames
  -> VGGT static reconstruction
  -> optional COLMAP bundle adjustment / model analysis
  -> scale alignment from calibration, vehicle dimensions, lanes, markings, or survey priors
  -> vehicle proxy replay from dimensions, OBBs, frame-wise poses, and masks
  -> collision diagnostics and evidence report
  -> optional mesh/splat/viewer export
```

## Expected Output Tree

```text
outputs/camera_only_reconstruction/
├── manifest.json
├── rgb_frames/<vehicle_or_video>/frame_0000.jpg
├── masks/<vehicle_or_video>/frame_0000.png
├── masked_frames/<vehicle_or_video>/frame_0000.jpg
├── mask_manifest.json
├── reconstruction/
│   ├── cameras.json
│   ├── points.ply
│   ├── depth/
│   ├── reprojection_report.json
│   └── scale_alignment.json
├── calibration/
│   ├── intrinsics.json
│   ├── extrinsics.json
│   ├── scale_constraints.json
│   └── scale_alignment_report.md
├── replay/
│   ├── vehicle_tracks.json
│   ├── vehicle_dimensions.json
│   └── collision_diagnostics.json
├── scene.glb
├── viewer/
└── reports/
    ├── evidence_report.md
    ├── limitations.md
    ├── no_lidar_audit.md
    └── quality_report.md
```

`points.ply` or `scene.glb` may be replaced by a backend-native 3D output only
when the report explains how to view it and why conversion was not possible.

## Non-Negotiable Completion Gates

- `manifest.json` must declare `lidar_used=false`.
- No reconstruction stage may read `lidar01`.
- Legacy `viewer_assets/*.glb` and historical LiDAR PLY outputs must not be used
  as geometry input.
- At least one reconstruction backend must be executed or fail with captured
  command output.
- Completion requires a no-LiDAR audit and a quality report. A smoke-test
  manifest with placeholder masks is not completion.

## Quality Loop

Repeat the reconstruction stage until the best achievable result is documented:

1. select or resample RGB frames for parallax, texture, and static context,
2. generate or improve dynamic-object masks,
3. run the primary backend,
4. inspect registration count, point density/coverage, reprojection or
   consistency metrics, and visual preview,
5. adjust frames, masks, backend, or preprocessing when quality fails,
6. record every failed attempt and fallback decision in `quality_report.md`.

## Phase Design

### Phase 1: Claim Audit

Keep:

- static/dynamic decomposition,
- frame-wise vehicle IDs and frame indices,
- calibrated proxy replay,
- OBB overlap/minimum-distance diagnostics,
- evidence-grounded reporting.

Revise/remove:

- LiDAR fusion as metric geometry source,
- LiDAR-derived vehicle surfaces as dynamic geometry proof,
- hybrid Gaussian/RGB PLY as accident evidence,
- LiDAR point counts as headline metrics.

### Phase 2: Technology Selection

Primary path:

- Grounded SAM 2 or SAM 2 for masks.
- VGGT for cameras, depth, point maps, tracks, and COLMAP export.
- COLMAP checks when image geometry allows.

Fallback path:

- MegaSaM or MASt3R-SLAM for video pose/depth instability.
- MASt3R or DUSt3R for difficult matching.
- Depth Anything V2 for monocular depth support only.

### Phase 3: RGB Frame Set and Mask-Out Protocol

Frame selection rules:

- sample 8 to 40 frames per vehicle/video for the first smoke test,
- prefer frames with road markings, buildings, signs, lane lines, or textured
  static context,
- include pre-contact, contact, and post-contact frames if available,
- reject frames dominated by blur, glare, night/rain artifacts, or full-frame
  dynamic occlusion unless used as a documented failure case.

Mask QA gates:

- every dynamic road user must have a mask entry,
- mask coverage must be plausible by class and frame,
- masked static frames must preserve road markings and static landmarks,
- mask failures must be listed in `reports/evidence_report.md`.

Placeholder masks can validate paths only. They cannot be used for a completed
reconstruction claim.

### Phase 3b: Reconstruction Execution

Execution rules:

- run VGGT first when installable and compatible with the available hardware,
- run COLMAP when enough camera motion and texture exist,
- try DUSt3R/MASt3R/MegaSaM/MASt3R-SLAM/Depth Anything based fallbacks when the
  primary path is unavailable or low quality,
- export a 3D artifact and preview after each serious attempt,
- fail the completion gate if no actual artifact exists.

### Phase 4: Replay and Collision Schemas

Vehicle replay uses:

- `vehicle_id`,
- dimensions in meters,
- frame-wise pose in the shared metric frame,
- OBB corners or dimensions plus pose,
- optional 2D masks and keypoints,
- source camera/video IDs,
- uncertainty fields.

Collision diagnostics use:

- pair of vehicle IDs,
- frame interval,
- separating-axis OBB gap or minimum distance,
- overlap boolean,
- scale residual and pose uncertainty,
- evidence paths for masks, tracks, and reconstruction.

### Phase 5: Research Narrative

The research summary should state that static scene geometry is reconstructed
from masked RGB videos, dynamic vehicles are replayed as calibrated proxies, and
collision claims come from structured evidence. It should not claim complete
dynamic 3D Gaussian reconstruction from ordinary black-box video.

## Evidence Layer

Evidence allowed for accident claims:

- frame index and timestamp,
- vehicle ID and camera/source ID,
- segmentation mask path and mask status,
- camera intrinsics/extrinsics or estimated camera pose,
- static reconstruction confidence/reprojection error,
- metric scale constraints and residuals,
- vehicle dimensions and pose source,
- OBB gap or minimum distance,
- uncertainty and validation status.

Evidence not sufficient by itself:

- visually plausible novel-view rendering,
- a 3D Gaussian scene without metric diagnostics,
- monocular depth without scale alignment,
- LiDAR-only reconstruction when describing the new real-world method.
