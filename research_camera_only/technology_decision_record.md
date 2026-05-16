# Technology Decision Record

Status: accepted and executed for the Town04 camera-only SLAM3R result.

This record selected the candidate technologies for the camera-only upgrade.
The completed repository result uses SLAM3R RGB point-map prediction,
DeepAccident calibration, dynamic masks where available, and calibrated vehicle
proxies. Generated visual outputs are written under
`outputs/town04_type1_subtype2_slam3r_incremental_layers/` during local runs
and are not tracked in git.

Date: 2026-05-13.

## Decision

Use Grounded SAM 2 or SAM 2 for dynamic-object masking, then use VGGT as the
primary camera-only static reconstruction backend. Represent dynamic vehicles as
calibrated proxies with known dimensions, OBBs, frame-wise poses, masks, and
uncertainty. Use MegaSaM, MASt3R-SLAM, MASt3R, DUSt3R, COLMAP, and Depth
Anything V2 as fallbacks depending on failure mode.

## Rationale

The goal is not black-box dynamic 3D scene generation. Accident statements must
be traceable to frames, masks, geometry, calibration, scale constraints, poses,
and OBB diagnostics. The selected stack separates visual reconstruction from
metric accident evidence:

- Segmentation masks remove moving objects from static reconstruction and remain
  auditable evidence.
- VGGT can infer cameras, depth maps, point maps, and tracks from one to many
  views, and can export COLMAP-style files for downstream checks.
- Classical COLMAP checks provide interpretable SfM/MVS diagnostics when the
  footage has enough parallax and texture.
- Dynamic vehicles are modeled as calibrated objects rather than hallucinated
  dynamic geometry.

## Candidate Evaluation

| Component | Role | Inputs | Outputs | License notes | GPU / install | Maturity | Failure modes |
|---|---|---|---|---|---|---|---|
| Grounded SAM 2 | Primary open-vocabulary dynamic-object masking | RGB video or frames, text prompts such as `car. truck. bus. pedestrian. motorcycle.` | Per-object masks, tracking visualization, IDs if tracking is stable | Grounded-SAM-2 repo is Apache-2.0; check linked detector/model terms for chosen grounding backend | Python stack with SAM 2 and grounding model; GPU expected for practical video | Active repo with video demos and custom prompt workflow | Misses unusual vehicles, ID switches, over-masks reflections/shadows, API-token dependence if using hosted DINO-X/Grounding DINO 1.5 variants |
| SAM 2 | Promptable video/image segmentation and tracking fallback | RGB video/frames plus box/click/mask prompts | Object masks through a video | Apache-2.0 for checkpoints, demo, and training code | Python >=3.10, torch >=2.5.1, torchvision >=0.20.1; CUDA extension improves post-processing | Strong Meta FAIR release with video support | Needs prompts/detections, may drift through occlusion, transparent/night/rain cases need manual QA |
| VGGT | Primary static reconstruction from masked RGB | Masked RGB frames or videos sampled to frames | Camera intrinsics/extrinsics, depth maps, point maps, 3D tracks, COLMAP export | VGGT license updated July 29, 2025 for commercial-friendly code; only `VGGT-1B-Commercial` checkpoint permits commercial use; original checkpoint remains non-commercial | Python/PyTorch; H100 reference memory ranges from about 1.9 GB for 1 frame to 40.6 GB for 200 frames | CVPR 2025, active Meta/Oxford repo, COLMAP export path | Low parallax, dynamic leftovers, motion blur, rolling shutter, textureless roads, scale ambiguity, checkpoint access/licensing |
| MegaSaM | Fallback for casual/dynamic video pose and depth | Monocular RGB video frames, mono-depth precompute | Camera tracking and consistent video depth | Software Apache-2.0; other materials CC-BY | Multi-stage Python pipeline, GPU expected; scripts for in-the-wild DAVIS-style video | CVPR 2025 project, useful for difficult video motion | Complex install, depends on mono-depth quality, may smooth/regularize collision-critical details |
| MASt3R-SLAM | Fallback dense monocular SLAM for video | RGB video stream, optional known calibration | Poses and dense geometry | CC BY-NC-SA 4.0, non-commercial/share-alike | CUDA-heavy; PyTorch 2.5.1 with CUDA 11.8/12.x; several third-party installs | CVPR 2025, real-time dense SLAM claim, strong for in-the-wild video | Non-commercial license, build complexity, loop closure or drift failures, moving vehicles can contaminate map |
| MASt3R | Fallback matching/reconstruction for hard pairs or sparse keyframes | Image pairs or image collections | 2D-2D matches, 3D pointmaps, SfM support | CC BY-NC-SA 4.0, non-commercial/share-alike | Conda/PyTorch CUDA; optional ASMK and CUDA RoPE kernels | ECCV 2024 official implementation | Pair selection scale, non-commercial license, dynamic objects and low texture can still break matching |
| DUSt3R | Fallback two-view/multi-view geometric reconstruction | Image pairs/sets | Focals, poses, 3D point maps, confidence masks | CC BY-NC-SA 4.0, non-commercial/share-alike | Conda/PyTorch CUDA; optional CUDA RoPE kernels | CVPR 2024 official implementation | Non-commercial license, global alignment may struggle on long video, scale ambiguity |
| COLMAP | Independent SfM/MVS and bundle-adjustment check | RGB frames, masks, optional camera priors | Sparse/dense reconstruction, cameras, points, reprojection/model diagnostics | COLMAP itself is new BSD; dependencies may affect binary licensing | Mature C++/CUDA optional; can run CPU but slower | Longstanding, interpretable, widely used | Fails with low parallax, motion blur, low texture, dynamic occluders, rolling shutter |
| Depth Anything V2 | Monocular depth support when geometry is underconstrained | RGB image/frame | Relative depth; metric variants depending on model/task | Small model Apache-2.0; Base/Large/Giant models CC-BY-NC-4.0 | Python/PyTorch; lightweight to very large model variants | NeurIPS 2024, broad community support | Relative scale, temporal inconsistency, poor metric reliability without scale priors |

## Fallback Policy

1. Run Grounded SAM 2/SAM 2 QA first. If moving objects remain in the static
   frames, reconstruction metrics are not evidence.
2. Try VGGT on masked keyframes. Export cameras/points/depth and record
   registered frame count, confidence, and optional COLMAP bundle adjustment.
3. If VGGT fails because of video motion or low parallax, try MegaSaM or
   MASt3R-SLAM for pose/depth.
4. If matching is the bottleneck, try MASt3R/DUSt3R on curated keyframe pairs.
5. Use COLMAP as a consistency check whenever feature geometry is sufficient.
6. Use Depth Anything V2 only as support for depth priors or qualitative
   geometry hints, not as standalone metric collision evidence.

## References

- VGGT official repository and license: https://github.com/facebookresearch/vggt
- Grounded SAM 2 official repository: https://github.com/IDEA-Research/Grounded-SAM-2
- SAM 2 official repository: https://github.com/facebookresearch/sam2
- MegaSaM official repository: https://github.com/mega-sam/mega-sam
- MASt3R official repository: https://github.com/naver/mast3r
- DUSt3R official repository: https://github.com/naver/dust3r
- MASt3R-SLAM official repository/project page: https://github.com/rmurai0610/MASt3R-SLAM and https://edexheim.github.io/mast3r-slam/
- COLMAP documentation/repository: https://github.com/colmap/colmap
- Depth Anything V2 official repository: https://github.com/DepthAnything/Depth-Anything-V2
