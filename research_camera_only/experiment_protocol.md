# Experiment Protocol

## Objective

Demonstrate that ReSceneScribe can produce an actual, viewable camera-only 3D
reconstruction result:

1. reconstruct the static environment from vehicle-mounted RGB frames after
   dynamic-object masking,
2. align reconstruction scale with explicit priors,
3. replay vehicles as calibrated proxies,
4. report collision diagnostics using frame-indexed OBB evidence.

The completed Town04 execution is packaged under
`outputs/town04_type1_subtype2_slam3r_incremental_layers/`. Earlier generic
paths such as `outputs/camera_only_reconstruction/` remain valid templates for
future real-world cases, but the current repository evidence is the Town04
SLAM3R incremental output.

## Inputs

Required real-world inputs:

- RGB videos or extracted RGB frames from one or more vehicle-mounted cameras,
- camera intrinsics or a documented plan for estimating them,
- approximate camera mounting pose when available,
- vehicle dimensions,
- at least one scale prior such as lane width, road marking spacing, surveyed
  distance, calibrated camera baseline, or known vehicle dimensions.

Prototype-only inputs:

- DeepAccident RGB frames and calibration may be used to validate schemas or
  compare estimated poses.
- DeepAccident LiDAR must not be used as an input to the new method.

Hard input exclusions:

- no `lidar01` reads during reconstruction,
- no reuse of `viewer_assets/*.glb`,
- no reuse of historical LiDAR-derived PLY/GLB files.

## Step 1: RGB Frame Set

Prepare a small smoke set first:

```bash
python3 research_camera_only/scripts/prepare_rgb_demo.py \
  --source viewer_frames \
  --out research_camera_only/demo_outputs/rgb_demo \
  --agents ego_vehicle other_vehicle \
  --frame-start 0 \
  --frame-end 12 \
  --copy \
  --make-empty-masks
```

The generated masks are placeholders only. Replace them with Grounded SAM 2 or
SAM 2 outputs before reconstruction evaluation.

## Step 2: Dynamic-Object Masking

Primary prompts:

```text
car. truck. bus. van. motorcycle. bicycle. pedestrian. person.
```

Required mask manifest fields:

- `case_id`,
- `source.kind`,
- `source.lidar_used=false`,
- `frames[].agent`,
- `frames[].frame_index`,
- `frames[].rgb_path`,
- `frames[].mask_path`,
- `frames[].mask_status`,
- `frames[].dynamic_classes`,
- `frames[].qa_status`.

Mask QA:

- visually inspect every Nth frame plus all collision-near frames,
- record missed objects and over-masked static objects,
- fail the reconstruction run if a moving vehicle remains unmasked in static
  reconstruction frames.

## Step 3: Static Reconstruction

Primary:

- run VGGT on masked static frames,
- export cameras, points, depth, and COLMAP-compatible files when possible,
- record registered frames, confidence, and runtime/GPU facts.

Checks:

- if COLMAP can register enough frames, run bundle adjustment or model analysis,
- record reprojection error and failed frames,
- export a static point cloud or mesh for inspection,
- optionally export a 3DGS/splat representation for visualization only.

Required exports:

- `outputs/camera_only_reconstruction/reconstruction/points.ply` or a documented
  backend-native equivalent,
- `outputs/camera_only_reconstruction/scene.glb` or a viewer/preview path,
- `outputs/camera_only_reconstruction/reports/no_lidar_audit.md`,
- `outputs/camera_only_reconstruction/reports/quality_report.md`.

Fallbacks:

- MegaSaM when video pose/depth is unstable or dynamic content remains high,
- MASt3R-SLAM for monocular video sequences with useful motion,
- MASt3R/DUSt3R for difficult keyframe matching,
- Depth Anything V2 to support qualitative depth or scale hypotheses.

Quality loop:

- if frame registration is low, resample frames for more parallax/static texture,
- if moving objects contaminate geometry, improve masks and rerun,
- if point density/coverage is poor, try a fallback backend or depth support,
- if scale is unstable, add or revise scale priors,
- stop only when the best achievable result is documented with evidence.

## Step 4: Scale and Calibration

Scale constraints must be explicit:

- source type: lane width, road marking, known vehicle size, surveyed distance,
  camera calibration, or prototype DeepAccident calibration,
- measurement value and units,
- frame(s) where measured,
- residual after alignment,
- uncertainty bounds.

The scale report fails if:

- no metric prior is declared,
- the alignment residual is missing,
- collision diagnostics are reported in meters without a scale source.

## Step 5: Vehicle Replay

Vehicle replay stores:

- vehicle ID,
- dimensions in meters,
- pose per frame in the shared metric frame,
- pose source,
- source frame and camera,
- OBB corners or pose plus dimensions,
- uncertainty.

Acceptable pose sources:

- real-world calibration or multi-view tracking,
- manual/assisted annotation with calibration constraints,
- DeepAccident calibration for prototype validation only.

Do not claim reconstructed vehicle surface completeness unless directly
validated. Use procedural/CAD proxies by default.

## Step 6: Collision Diagnostics

For every collision statement, report:

- vehicle pair,
- frame interval,
- best/closest frame,
- OBB gap or minimum distance in meters,
- overlap boolean,
- scale residual,
- pose uncertainty,
- mask/reconstruction evidence paths,
- validation status.

## Failure Recording

Record failures explicitly:

- low parallax,
- motion blur,
- rolling shutter,
- night/rain glare,
- synchronization uncertainty,
- occlusion,
- segmentation miss/over-mask,
- reconstruction drift,
- insufficient scale prior,
- OBB proxy too coarse for deformation/contact claims.

Failure records must include the failed command, relevant log excerpt, output
directory, why the result was rejected, and the next fallback attempted.
