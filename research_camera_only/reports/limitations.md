# Camera-Only Limitations

## Geometry and Scale

- Monocular or weak-baseline video has scale ambiguity unless a metric prior is
  supplied.
- Low parallax from straight driving or stationary cameras can prevent stable
  camera/structure estimation.
- Textureless asphalt, sky, glare, rain, and night scenes reduce feature and
  reconstruction quality.
- Rolling shutter and motion blur can bias pose and OBB placement.
- COLMAP checks may fail on short, low-texture, or highly dynamic clips.

## Segmentation

- Grounded SAM 2/SAM 2 can miss unusual vehicles or small/distant road users.
- Masks can over-remove road markings, curbs, signs, or static parked objects.
- ID tracking can switch during occlusion or heavy overlap.
- Placeholder/empty masks are only smoke-test artifacts and must never be used
  as reconstruction evidence.

## Dynamic Replay

- Procedural/CAD vehicle proxies are geometric approximations, not deformation
  models.
- OBB overlap is a contact proxy. It does not prove deformation, impact force,
  occupant injury, or exact contact surface.
- Pose uncertainty near collision can dominate the final OBB gap/min-distance
  result.
- Synchronization errors across vehicles/videos can shift the inferred collision
  frame.

## Model and License Constraints

- VGGT checkpoint choice matters: commercial use requires the commercial
  checkpoint and compliance with its license.
- MASt3R, DUSt3R, and MASt3R-SLAM are non-commercial/share-alike options.
- MegaSaM and SAM 2/Grounded SAM 2 are more permissive, but their full dependency
  and model terms still need to be checked at install time.
- Heavy model runs require GPUs; this repo currently provides a scaffold and
  local RGB demo, not installed model checkpoints.

## Reporting Constraints

- 3DGS/splat exports are visualization artifacts unless backed by camera,
  reprojection, scale, and OBB diagnostics.
- A visually plausible static reconstruction is not enough for an accident
  statement.
- Every claim must state frame interval, vehicle IDs, evidence paths, pose/scale
  source, diagnostic value, and uncertainty.
