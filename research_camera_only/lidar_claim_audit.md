# LiDAR Claim Audit

Audit date: 2026-05-13.

Scope: `goal.md`, `README.md`, `docs/`, historical manuscript notes,
`scripts/`, and `viewer/`. Paper files are no longer tracked in the repository;
this audit is retained only as a record of claims that were revised during the
camera-only update.

## Summary

The existing repository proves a useful static/dynamic accident replay pattern,
but its metric geometry is produced from DeepAccident LiDAR streams and labels.
The camera-only upgrade should keep the decomposition, frame-indexed replay,
calibrated vehicle proxies, and OBB diagnostics, while revising or removing
claims that LiDAR fusion, LiDAR-observed vehicle surfaces, or a hybrid
Gaussian/RGB PLY are the primary accident evidence layer.

## Keep / Revise / Remove Table

| Source | Current claim or dependency | Decision | Camera-only replacement |
|---|---|---|---|
| Historical manuscript abstract | Combined camera pilot with DeepAccident dynamic validation and reported a hybrid Gaussian/RGB reconstruction with 1,177,009 vertices. | Revise | State masked RGB static reconstruction plus calibrated proxy replay. Keep frame 54-56/OBB diagnostics only as prototype validation evidence. |
| Historical manuscript dataset section | DeepAccident was justified because it provides synchronized camera, LiDAR, calibration, and labels. | Revise | DeepAccident may prototype pose priors and validation, but the real-world method must require RGB videos plus calibration/scale priors, not LiDAR. |
| Historical manuscript metric alignment | Metric alignment mapped each LiDAR point into world coordinates, then colorized it through cameras. | Remove from main method | Replace with camera pose/depth estimation from VGGT or fallback SLAM/SfM, followed by COLMAP-format export and reprojection checks. |
| Historical manuscript static fusion | Static background was fused from world-space points after dynamic boxes were removed. The point source was LiDAR. | Revise | Static background is reconstructed from RGB frames after segmentation-mask removal of dynamic objects. Dynamic object labels are masks, not LiDAR boxes. |
| Historical manuscript vehicle geometry | Target vehicle geometry was reconstructed by collecting LiDAR points inside vehicle OBBs and motion-compensating them into a final pose. | Revise | Replace reconstructed vehicle surfaces with calibrated vehicle proxies: known dimensions, OBBs, frame-wise poses, optional CAD/procedural mesh, and mask-derived 2D constraints. |
| Historical manuscript collision diagnostics | Collision diagnostics used frame-wise vehicle poses and OBB separating-axis gap. | Keep | This is still the correct metric evidence layer if pose, scale, and uncertainty are reported. |
| Historical manuscript experiment setup | DeepAccident experiment listed multi-view RGB cameras, LiDAR, calibration, and 3D labels. | Revise | Use DeepAccident RGB/calibration only for prototype and validation references. Do not use `lidar01` as camera-only method input. |
| Historical manuscript replay background | Temporal replay background was reconstructed by fusing all four LiDAR streams. | Revise | Replay background should come from masked RGB static reconstruction. Vehicle motion remains calibration-constrained proxy animation. |
| Historical manuscript collision-state asset | Collision-state 3D asset started from 8,616,764 LiDAR points and exported a hybrid Gaussian/RGB PLY. | Remove from main claim | Keep only as prior demo/background. New method exports camera-only static geometry and optional splat/mesh visualization, with metric evidence in structured reports. |
| Historical manuscript quantitative table | Quantitative result table included source LiDAR points, removed vehicle points, and final PLY vertices. | Revise | Replace with frame count, mask coverage, registered camera count, reprojection error, scale residual, OBB min distance/gap, collision frame, and uncertainty. |
| Historical manuscript limitations | Limitations mentioned hybrid PLY and LiDAR visibility. | Revise | Limitations must focus on monocular scale ambiguity, low parallax, motion blur, rolling shutter, rain/night, segmentation error, pose drift, and OBB proxy limits. |
| `README.md:41`-`47` | Included viewer contains regular/ultra static LiDAR GLBs. | Revise | Keep as historical packaged demo; add camera-only research path and state LiDAR GLBs are not the new real-world method. |
| `README.md:120`-`132` | Rebuild command creates collision-state hybrid Gaussian/RGB PLY from DeepAccident. | Revise | Move to legacy section. Add new commands for RGB frame preparation, mask generation, reconstruction backend execution, and validation. |
| `docs/method_summary.md:7`-`28` | Static points are fused from DeepAccident LiDAR and vehicle geometry is rebuilt from target-vehicle LiDAR points. | Revise | Document camera-only static reconstruction and proxy vehicles; keep static/dynamic separation and OBB diagnostics. |
| `docs/dataset.md:14`-`29` | Expected dataset includes `lidar01`, calibration, labels, and six cameras. | Revise | New protocol should require RGB frames/videos and calibration/scale priors. DeepAccident LiDAR remains optional validation reference only. |
| `docs/artifacts.md:6`-`18` | Packaged viewer assets are static LiDAR GLBs. | Revise | Mark as legacy demo assets. Camera-only outputs should be under `research_camera_only/outputs/`. |
| `docs/artifacts.md:19`-`55` | Large final PLY is a LiDAR-derived collision-state asset. | Remove from new evidence layer | Optional visualization/export only; not metric accident proof. |
| `docs/research_success_results_ko.md:369`-`383` | Workflow loads LiDAR, labels, calibration, and RGB cameras to produce PLY. | Revise | Convert paper-ready summary to RGB frames -> masks -> static reconstruction -> scale alignment -> proxy replay. |
| `scripts/build_four_vehicle_collision_viewer.py:388`-`476` | `build_static_lidar_scene` fuses LiDAR points, removes dynamic vehicle boxes, and exports GLBs. | Keep as legacy only | New viewer builder should load camera-only static reconstruction and `replay/vehicle_tracks.json`. |
| `scripts/build_town03_clean_hybrid_gaussian_ply.py:1`-`360` | Builds LiDAR/RGB hybrid PLY and vehicle surfaces from LiDAR observations. | Legacy only | Do not reuse as camera-only method except for output-field inspiration. |
| `scripts/prepare_town03_3dgs_masked_static_dataset.py:1`-`405` | Masks images using projected 3D labels and initializes static points from LiDAR. | Revise | Replace label projection with Grounded SAM 2/SAM 2 masks; replace LiDAR initialization with VGGT/MegaSaM/MASt3R/COLMAP reconstruction. |
| `viewer/index.html:165`-`173` | UI states background is static LiDAR from all DeepAccident streams. | Revise | New camera-only viewer label should state masked RGB static reconstruction and calibrated proxy replay. |

## Conversion Rules

- Keep frame-indexed diagnostics and vehicle IDs.
- Keep OBB overlap/minimum distance, but attach uncertainty and scale residuals.
- Keep procedural vehicle proxies because they are more defensible than claiming
  full dynamic surface reconstruction from ordinary dashcam video.
- Replace LiDAR point counts with camera-only metrics: usable frame count, mask
  coverage, registered camera count, reprojection error, scale residual, track
  continuity, OBB distance/gap, and collision-frame confidence.
- Treat 3DGS as visualization/export only. Do not use it as the metric evidence
  layer.
