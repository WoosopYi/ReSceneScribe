# Multi-camera Image-based Reconstruction Notes

## Sources checked

- Street Gaussians: dynamic urban driving scenes are modeled by separating
  static background and foreground actors instead of forcing all pixels into one
  static Gaussian field. Repo: https://github.com/zju3dv/street_gaussians
- DrivingGaussian: surrounding autonomous-driving reconstruction uses a
  composite representation for large static background plus multiple dynamic
  objects. Repo: https://github.com/VDIGPKU/DrivingGaussian
- SplatAD: autonomous-driving Gaussian rendering extends `gsplat` for camera and
  LiDAR rendering, but its official repo is mostly renderer extensions and the
  full model/data pipeline is not the camera-only fit for this project.
  Repo: https://github.com/carlinds/splatad
- VGGT: the official repo added `demo_colmap.py` to export VGGT predictions as
  COLMAP, with optional bundle adjustment, and says these COLMAP files can feed
  `gsplat` or other NeRF/Gaussian splatting libraries.
  Repo: https://github.com/facebookresearch/vggt
- Nerfstudio Splatfacto: Gaussian splatting works much better when initialized
  from existing SfM/COLMAP geometry; non-COLMAP datasets otherwise initialize
  randomly.
  Docs: https://docs.nerf.studio/nerfology/methods/splat.html
- NVIDIA NuRec / 3DGUT: the 3DGUT workflow consumes COLMAP sparse points and
  camera poses, then trains a dense photorealistic Gaussian scene.
  Docs: https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html
- COLMAP rig support: calibrated multi-camera systems should be modeled as
  fixed sensor rigs with synchronized frames when the rig constraints are
  available.
  Docs: https://colmap.github.io/legacy/3.12/rigs.html

## Resulting decision

The previous local approach trained 3DGS from DeepAccident transforms plus a
dense VGGT point cloud aligned by camera centers. That path is weak because the
Gaussian initializer is not a real SfM/COLMAP model with track constraints, and
the static/dynamic split is not strong enough for driving scenes.

The added backend uses:

1. Dynamic-object masked RGB images for geometry estimation.
2. VGGT `demo_colmap.py --use_ba` to create a COLMAP sparse model.
3. Sim(3) alignment from VGGT/COLMAP camera centers back to DeepAccident world
   camera centers.
4. Full RGB images for downstream 3DGS photometric training while retaining the
   COLMAP sparse model from the masked geometry stage.

This is still camera-image based. It does not consume `lidar01` point geometry.
Calibration is used as pose/scale metadata and as the world-frame alignment
target.

## Implemented files

- `scripts/run_multicamera_vggt_colmap_backend.py`
- Smoke output:
  `outputs/town04_type1_subtype2_vggt_colmap_ba_smoke_front/`
- Smoke Nerfstudio output:
  `outputs/nerfstudio_town04_vggt_colmap_ba_smoke/town04_type1_subtype2_vggt_colmap_ba_smoke_front/splatfacto/iter10/`

## Smoke validation

The smoke run used `ego_vehicle` with `Camera_Front`, `Camera_FrontLeft`, and
`Camera_FrontRight`, 12 total images. It completed VGGT + track bundle
adjustment and produced a COLMAP sparse model.

Key evidence:

- VGGT/COLMAP BA converged with final cost about `0.376 px`.
- COLMAP registered images: `12`.
- COLMAP points: `2671`.
- DeepAccident world alignment: median camera-center error about `0.729 m`;
  RMSE about `2.625 m`.
- Nerfstudio Splatfacto accepted the COLMAP output and completed a 10-step
  training smoke run using `images_fullrgb` plus `scene/sparse`.

An all-agent diagnostic export was also run with 48 images from all 4 agents and
6 cameras using VGGT feed-forward COLMAP mode without BA:

- Output:
  `outputs/town04_type1_subtype2_vggt_colmap_feedforward_all48/`
- Registered images: `48`.
- World alignment RMSE: about `8.009 m`.
- COLMAP `points3D.bin` was empty in this mode, so this output is useful as a
  camera-layout diagnostic, not as the final 3DGS initializer.

## Known limitation

The smoke run is not a final-quality reconstruction. It proves the corrected
backend works. Full-scene quality still requires shard selection that respects
actual overlap: per-agent/per-camera-group COLMAP shards should be built first,
then fused into DeepAccident world coordinates. A naive all-agent/all-camera BA
sample failed because too many selected images had insufficient shared visual
features.
