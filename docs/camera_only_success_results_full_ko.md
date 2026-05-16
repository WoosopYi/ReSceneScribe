# Camera-Only 사고 현장 3D Reconstruction 성공 결과 상세 정리

작성일: 2026-05-13
작업 위치: `ReSceneScribe` repository root
대상 시나리오: DeepAccident mini/local `type1_subtype2_accident` / `Town04_type001_subtype0002_scenario00017`
정리 기준: **성공한 중간 산출물과 최종 산출물만 포함한다. 실패하거나 폐기한 시도는 제외한다.**
최종 논문 정합성 기준:
`ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from Vehicle-Mounted Sensing and Multi-Agent Simulation for Evidence-Grounded Investigation`

## 1. 최종 결론

이번 작업의 핵심 성공 결과는 다음과 같다.

- DeepAccident Town04 type1-subtype2 사고 시나리오에서 **4대 차량, 차량당 6개 RGB 카메라, 총 1176개 RGB frame**을 사용했다.
- `lidar01` point cloud, 기존 LiDAR-derived PLY/GLB, simulator map mesh를 primary reconstruction path에서 읽지 않았다.
- DeepAccident calibration은 metric camera/world placement와 vehicle replay를 위해 사용했다.
- SLAM3R 기반 camera-only base reconstruction을 성공적으로 만들었고, base scene은 **545,306 points**를 export했다.
- 사용자가 요청한 “현재 좋은 결과를 base로 잡고 조금씩 쌓아가는 방식”을 incremental layering으로 구현했고, 최종 cumulative scene은 **1,508,944 points**까지 확장했다.
- 차량은 영상으로 표면을 복원했다고 주장하지 않고, calibration pose와 실제 vehicle dimensions를 기반으로 한 **car-like 3D proxy**로 replay했다.
- 연구 문서는 camera-only reconstruction을 main contribution으로 재구성했다. 현재 GitHub 커밋에는 논문 파일을 포함하지 않는다.

논문에서 physical pilot은 실제 차량 2대와 windshield-area RGB camera를 사용한
capture-feasibility 확인 단계로 설명한다. 정량 reconstruction/replay 수치와
diagnostic은 calibration과 vehicle metadata가 제공되는 DeepAccident Town04
benchmark 결과로 해석한다.

## 2. 연구 방향과 증거 경계

이번 연구에서 가장 중요한 framing은 다음이다.

> 사고 현장 3D reconstruction과 accident replay를 일반 차량에서 얻을 수 있는 vehicle-mounted RGB videos 중심으로 구성한다. Calibration은 metric alignment와 replay를 위한 prior로 사용하고, LiDAR는 primary reconstruction sensor로 사용하지 않는다.

사용한 것:

- DeepAccident RGB camera frames
- DeepAccident calibration files
- Camera intrinsic/extrinsic 및 camera/world pose
- Vehicle replay를 위한 frame-wise `ego_to_world` calibration pose
- Dynamic mask export가 존재하는 frame의 mask
- SLAM3R RGB point-map prediction과 confidence map
- Vehicle dimensions 및 label/context 정보

사용하지 않은 것:

- `lidar01` point cloud
- Historical LiDAR-derived PLY/GLB outputs
- Simulator map mesh
- Proxy meshes나 viewer asset을 background geometry로 재사용하는 경로

주의할 점:

- DeepAccident calibration key 이름에 `lidar_to_Camera_*`가 포함될 수 있으나, 이번 작업에서는 이를 camera extrinsic chain으로만 사용했다.
- LiDAR point geometry 자체는 읽지 않았다.
- 차량 geometry는 reconstructed surface가 아니라 replay용 calibrated proxy다.

## 3. 입력 데이터 구성

| 항목 | 값 |
|---|---|
| Dataset root | local DeepAccident-compatible dataset root |
| Category | `type1_subtype2_accident` |
| Scenario | `Town04_type001_subtype0002_scenario00017` |
| Vehicle agents | `ego_vehicle`, `ego_vehicle_behind`, `other_vehicle`, `other_vehicle_behind` |
| Agent count | 4 |
| Cameras per agent | 6 |
| Camera names | `Camera_Front`, `Camera_FrontLeft`, `Camera_FrontRight`, `Camera_Back`, `Camera_BackLeft`, `Camera_BackRight` |
| Selected frames | 1 to 49 |
| Frames per camera stream | 49 |
| Total RGB views | 4 agents x 6 cameras x 49 frames = 1176 |
| Calibration use | camera/world placement, scale alignment, vehicle replay |
| LiDAR point cloud use | false |

## 4. 성공 산출물 전체 목록

| 단계 | 산출물 | 상태 | 핵심 값 |
|---|---|---|---|
| 1 | Multi-camera known-pose export | `export_only` 성공 | 1176 RGB views, masks ok, no LiDAR |
| 2 | Camera-calibrated visual background proxy | `ok` | 1,400,000 exported points |
| 3 | Readable camera-only road/city viewer | `ok` | ground texture coverage 0.579, buildings 7, trees 24 |
| 4 | SLAM3R camera-only base reconstruction | `ok` | 545,306 points |
| 5 | Incremental SLAM3R layering | `ok` | 1,508,944 final points |
| 6 | Calibration-constrained vehicle replay | `ok` | 4 vehicle tracks, 49 samples each |
| 7 | Proxy collision/replay diagnostics | `ok` | closest frame 49, clearance -0.6842 m |
| 8 | Car-like vehicle viewer model | `ok` | body/cabin/windows/wheels/lights, local +X forward |
| 9 | Repository research documentation | `ok` | camera-only narrative |

## 5. 단계 1: Multi-Camera Known-Pose Export 성공

초기 성공 단계는 RGB camera frames, masks, calibration pose를 한 world coordinate system으로 묶는 export였다. 이 단계는 dense geometry를 직접 만들기보다, 이후 reconstruction backend가 사용할 수 있는 정렬된 입력 package를 만드는 목적이었다.

출력 경로:

- `outputs/town04_type1_subtype2_multicam_export/manifest.json`
- `outputs/town04_type1_subtype2_multicam_export/nerfstudio/transforms.json`
- `outputs/town04_type1_subtype2_multicam_export/nerfstudio/images/`
- `outputs/town04_type1_subtype2_multicam_export/nerfstudio/masks/`
- `outputs/town04_type1_subtype2_multicam_export/replay/agent_tracks.json`
- `outputs/town04_type1_subtype2_multicam_export/replay/accident_diagnostics.json`
- `outputs/town04_type1_subtype2_multicam_export/reports/no_lidar_audit.md`
- `outputs/town04_type1_subtype2_multicam_export/reports/quality_report.md`
- `outputs/town04_type1_subtype2_multicam_export/reports/nerfstudio_status.md`

성공 내용:

| 항목 | 값 |
|---|---:|
| Status | `export_only` |
| RGB views | 1176 |
| Nerfstudio frames | 1176 |
| Frame range | 1-49 |
| Mask backend | `ultralytics` / `ok` |
| Object context records | 6,992 |
| LiDAR geometry used | false |
| `lidar01` read | false |
| Historical LiDAR PLY/GLB read | false |

이 단계의 의미:

- 4대 차량의 6개 camera rig가 하나의 calibration-derived world frame 안에 들어갔다.
- Dynamic mask가 export되어 이후 static reconstruction에서 moving object suppression에 사용할 수 있게 되었다.
- Nerfstudio/instant-ngp 등 다른 renderer로 넘길 수 있는 known-pose dataset도 생성되었다.
- 이 단계는 geometry point count가 0인 것이 정상이다. 목적은 reconstruction 자체가 아니라 정렬된 RGB/calibration/mask handoff였기 때문이다.

## 6. 단계 2: Camera-Calibrated Visual Background Proxy 성공

SLAM3R base 이전에, camera rays와 calibration만으로 도시/도로의 가독성을 높이는 background proxy를 만들었다. 이 산출물은 metric dense MVS reconstruction이라기보다는, RGB camera evidence와 calibration으로 만든 stable visual background proxy다.

출력 경로:

- `outputs/town04_type1_subtype2_camera_background/summary.json`
- `outputs/town04_type1_subtype2_camera_background/reconstruction/points_world.ply`
- `outputs/town04_type1_subtype2_camera_background/scene.glb`
- `outputs/town04_type1_subtype2_camera_background/reports/preview_point_cloud.png`
- `outputs/town04_type1_subtype2_camera_background/reports/topdown_preview.png`
- `outputs/town04_type1_subtype2_camera_background/viewer/index.html`

성공 내용:

| 항목 | 값 |
|---|---:|
| Status | `ok` |
| Backend | `camera_calibrated_ground_and_background_proxy` |
| Input views | 312 |
| Ground candidate points | 2,943,758 |
| Shell candidate points | 1,731,565 |
| Points before voxel | 4,675,323 |
| Exported point count | 1,400,000 |
| Object context records | 1,855 |
| Elapsed seconds | 18.733 |
| LiDAR used | false |
| Calibration used | true |

World bounds:

| 항목 | 값 |
|---|---|
| BBox min | `[163.0076, -352.0524, 0.0200]` |
| BBox max | `[399.5519, -115.5398, 34.9989]` |
| World origin | `[255.2169, -252.5392, 4.9240]` |

이 단계의 의미:

- Ground는 image rays를 fixed world ground plane에 intersect해서 구성했다.
- Distant background는 non-sky image rays를 bounded shell에 project해서 만들었다.
- 도시/도로 배경의 시각적 맥락을 빠르게 확인하는 데 성공했다.
- 이 산출물은 main metric reconstruction으로 주장하기보다는, camera-only visual context를 확인한 중간 성공 결과로 두는 것이 적절하다.

## 7. 단계 3: Readable Camera-Only Road/City Viewer 성공

Raw point cloud viewer가 도로/도시 맥락을 충분히 읽기 어렵다는 문제가 있었기 때문에, road-frame 기반의 readable viewer를 추가로 만들었다. 이 결과는 사용자가 “배경이 제대로 안 만들어진다”라고 지적한 문제를 시각적으로 개선하기 위한 성공 산출물이다.

출력 경로:

- `outputs/town04_type1_subtype2_readable/manifest.json`
- `outputs/town04_type1_subtype2_readable/scene.glb`
- `outputs/town04_type1_subtype2_readable/viewer/index.html`
- `outputs/town04_type1_subtype2_readable/reports/ground_texture.png`
- `outputs/town04_type1_subtype2_readable/reports/ground_texture_projected_rgb_only.png`
- `outputs/town04_type1_subtype2_readable/reports/background_mask_preview.png`
- `outputs/town04_type1_subtype2_readable/reports/readable_topdown_preview.png`
- `outputs/town04_type1_subtype2_readable/reports/viewer_dense_background_front.png`
- `outputs/town04_type1_subtype2_readable/reports/viewer_dense_background_top.png`
- `outputs/town04_type1_subtype2_readable/reports/no_lidar_audit.md`
- `outputs/town04_type1_subtype2_readable/reconstruction/vertical_points_road_frame.ply`

성공 내용:

| 항목 | 값 |
|---|---:|
| Status | `ok` |
| Ground texture coverage | 0.579 |
| RGB projection views used | 312 |
| RGB projection samples used | 1,328,838 |
| Vertical points exported for audit | 220,000 |
| Buildings exported | 7 |
| Trees/vegetation proxies exported | 24 |
| BEV background prior | `ok` |
| LiDAR geometry used | false |
| `lidar01` point files read | false |
| Legacy LiDAR assets read | false |

Road frame:

| 축 | World vector |
|---|---|
| Origin | `[259.4314, -245.1090, 0.0344]` |
| Right | `[-0.999995, -0.003089, 0.000395]` |
| Up | `[0.000393, 0.000791, 0.9999996]` |
| Forward | `[0.003089, -0.999995, 0.000790]` |

Viewer 검증:

| 항목 | 값 |
|---|---:|
| Check file | `outputs/town04_type1_subtype2_readable/reports/viewer_dense_background_check.json` |
| Status | `single-plane RGB mesh loaded` |
| Canvas | 1440 x 900 |
| Top-view non-background pixels | 534,882 / 1,296,000 |
| Top-view non-background ratio | 0.4127 |
| Browser errors | none |
| Closest proxy shown | `ego_vehicle_behind / other_vehicle`, frame 49, clearance -0.68 m |

이 단계의 의미:

- Road plane은 calibration-derived vehicle trajectories에서 fit했다.
- Viewer frame은 `+Z`가 `ego_vehicle` motion direction을 따르도록 수정했다.
- Masked RGB camera frames를 road plane에 project해 ground texture를 만들었다.
- RGB/BEV color masks를 building/tree proxy mesh로 변환해 도로 주변 맥락을 읽기 쉽게 만들었다.
- VGGT dense vertical points는 audit용 PLY로 export했지만, 기본 `scene.glb`에는 넣지 않았다.
- 이 결과도 LiDAR geometry를 사용하지 않았다.

## 8. 단계 4: SLAM3R Camera-Only Base Reconstruction 성공

최종 camera-only reconstruction의 primary evidence로 사용한 가장 중요한 base 결과다. SLAM3R을 DeepAccident RGB streams에 적용하고, calibration camera centers로 world placement를 수행했다.

출력 경로:

- `outputs/town04_type1_subtype2_slam3r_reconstruction/summary.json`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/manifest.json`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reconstruction/points_world.ply`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reconstruction/points.ply`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/scene.glb`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reports/preview_point_cloud.png`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reports/quality_report.md`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reports/no_lidar_audit.md`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reconstruction/alignment.json`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reconstruction/stream_runs.json`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/viewer/index.html`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/replay/agent_tracks.json`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/replay/accident_diagnostics.json`

성공 내용:

| 항목 | 값 |
|---|---:|
| Status | `ok` |
| Backend | `slam3r_camera_ray_depth_calibrated` |
| Fusion mode | `camera-ray-depth` |
| SLAM3R commit | `f531d841ab743217a4464344119a350eb0556d17` |
| Stream count | 24 |
| Accepted streams | 10 |
| Rejected streams | 14 |
| Input RGB frames | 1176 |
| Points before voxel/sample | 900,000 |
| Exported point count | 545,306 |
| Alignment camera centers | 490 |
| Alignment RMSE | 1.4469 m |
| Alignment median error | 0.7856 m |
| Alignment max error | 6.8423 m |
| Object context records | 6,992 |
| Elapsed seconds | 72.567 |
| LiDAR used | false |
| Legacy LiDAR assets used | false |

World bounds:

| 항목 | 값 |
|---|---|
| BBox min | `[236.2262, -297.9653, -2.9975]` |
| BBox max | `[330.4998, -197.2098, 21.1524]` |

품질 필터:

| 필터 | 값 |
|---|---:|
| Stream RMSE max | 2.5 m |
| Stream median error max | 1.8 m |
| Stream max error max | 7.0 m |
| Frame alignment error max | 4.0 m |
| World z range | `[-3.0, 25.0]` |

이 단계의 의미:

- 24개 RGB stream을 모두 처리 대상으로 두고, alignment 품질 기준을 통과한 10개 stream만 base fusion에 사용했다.
- 14개 stream은 base scene에 억지로 넣지 않았다. 이 선택이 배경 폭발/잘못된 geometry 확산을 줄이는 데 중요했다.
- SLAM3R은 monocular RGB 기반 point-map/depth prediction을 제공하고, DeepAccident calibration이 최종 camera/world placement를 제공했다.
- `lidar01` point cloud는 읽지 않았고, no-LiDAR audit이 pass 상태다.

## 9. 단계 5: Incremental Layering 성공

사용자가 요청한 “지금 좋은 결과를 base로 잡고 조금씩 쌓는 방식”을 구현한 결과다. Base layer를 보존하고, 이후 layer는 기존 cumulative scene의 voxel을 침범하지 않는 새 points만 추가했다.

출력 경로:

- `outputs/town04_type1_subtype2_slam3r_incremental_layers/summary.json`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/manifest.json`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/reconstruction/points_world.ply`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/reconstruction/points.ply`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/scene.glb`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/reports/preview_point_cloud.png`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/reports/topdown_preview.png`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/reports/incremental_layers_report.md`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/stages/`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/layers/`

성공 내용:

| 항목 | 값 |
|---|---:|
| Status | `ok` |
| Backend | `slam3r_incremental_camera_ray_depth_layers` |
| Source | `outputs/town04_type1_subtype2_slam3r_reconstruction` |
| Base point count | 545,306 |
| Added point count | 963,638 |
| Final point count | 1,508,944 |
| Stage count | 4 |
| Layer count | 3 |
| Object context records | 6,992 |
| Elapsed seconds | 185.857 |
| Calibration used | true |
| RGB used | true |
| LiDAR used | false |
| Legacy LiDAR assets used | false |

World bounds:

| 항목 | 값 |
|---|---|
| World origin | `[279.9382, -242.8801, -0.3784]` |
| BBox min | `[232.1345, -307.0313, -2.9975]` |
| BBox max | `[334.6944, -196.8775, 24.9948]` |

누적 stage 결과:

| Stage | 설명 | New points | Cumulative points |
|---|---|---:|---:|
| `00_base` | 현재 좋은 no-LiDAR SLAM3R 결과를 immutable base로 사용 | 545,306 | 545,306 |
| `01_01_masked_dense` | base와 같은 accepted streams, masked frames 중심, 더 dense한 static background | 541,877 | 1,087,183 |
| `02_02_all_aligned_frames` | accepted streams와 추가 aligned frames, mask가 있으면 사용하고 없으면 high-confidence RGB depth만 사용 | 303,076 | 1,390,259 |
| `03_03_strict_extra_streams` | stream rejection을 일부 완화하되 per-frame alignment, masks, confidence를 엄격히 유지 | 118,685 | 1,508,944 |

Layer별 세부 결과:

| Layer | Candidate points | Duplicate voxel points | New points | Cumulative | Accepted streams | RMSE / Median / Max |
|---|---:|---:|---:|---:|---:|---|
| `01_masked_dense` | 844,000 | 302,123 | 541,877 | 1,087,183 | 9 | 1.3233 / 0.7334 / 6.4580 m |
| `02_all_aligned_frames` | 604,662 | 301,586 | 303,076 | 1,390,259 | 9 | 1.3233 / 0.7334 / 6.4580 m |
| `03_strict_extra_streams` | 465,224 | 346,539 | 118,685 | 1,508,944 | 13 | 1.9832 / 0.8766 / 11.8520 m |

Layer acceptance rule:

- Base layer는 그대로 보존한다.
- 추가 layer는 saved SLAM3R RGB point maps와 DeepAccident calibration을 재사용한다.
- 새 point는 cumulative scene에서 `0.08 m` voxel이 아직 점유되지 않은 경우에만 추가한다.
- 기본 run에는 point cap을 두지 않았다.
- `lidar01` point cloud는 읽지 않았다.

이 단계의 의미:

- 기존 좋은 base를 망가뜨리지 않고 주변 도시/도로 context를 확장하는 데 성공했다.
- 최종 point count가 base 대비 약 2.77배 증가했다.
- Layer 3에서는 stream-level rejection을 일부 완화했지만, per-frame alignment와 mask/confidence 조건을 더 엄격히 유지해 폭발적인 잘못된 geometry를 억제했다.
- 이 결과가 main quantitative result에 사용되었다.

## 10. 단계 6: Vehicle Replay 성공

차량은 reconstructed dynamic surface가 아니라, calibration pose와 vehicle dimensions를 이용한 procedural proxy로 replay했다. 이 방식이 연구 설명에서 가장 방어 가능한 표현이다.

출력 경로:

- `outputs/town04_type1_subtype2_slam3r_incremental_layers/replay/agent_tracks.json`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/replay/accident_diagnostics.json`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html`

Coordinate system:

- Viewer coordinate: `viewer xyz = world x,z,y minus reconstruction origin`
- Reconstruction world origin: `[279.9382, -242.8801, -0.3784]`

Vehicle dimensions and motion:

| Vehicle | Length / Width / Height | Samples | Frames | First world position | Last world position | Net motion | Cumulative motion | Avg speed | Max speed |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| `ego_vehicle` | 5.2375 / 1.9298 / 1.6384 m | 49 | 1-49 | `[258.5709, -216.3001, 0.0193]` | `[258.7170, -263.5906, 0.0202]` | 47.291 m | 47.291 m | 19.302 m/s | 23.163 m/s |
| `ego_vehicle_behind` | 4.1812 / 1.9941 / 1.3853 m | 49 | 1-49 | `[258.5387, -205.8781, 0.0083]` | `[258.6304, -243.1701, 0.0097]` | 37.292 m | 37.293 m | 15.221 m/s | 22.622 m/s |
| `other_vehicle` | 4.6110 / 2.2417 / 1.6673 m | 49 | 1-49 | `[295.6063, -249.9874, 0.0701]` | `[260.2324, -247.0478, 0.1074]` | 35.496 m | 35.907 m | 14.656 m/s | 22.747 m/s |
| `other_vehicle_behind` | 4.1928 / 1.8162 / 1.4738 m | 49 | 1-49 | `[304.7236, -250.0500, -0.0009]` | `[279.9965, -249.8797, -0.0008]` | 24.728 m | 24.728 m | 10.093 m/s | 18.490 m/s |

이 단계의 의미:

- 각 차량은 49개 frame-wise calibration pose sample을 갖는다.
- 차량 크기는 label/context에서 얻은 dimensions를 사용했다.
- 차량 replay는 collision story를 설명하기 위한 calibrated visualization이며, 차량 표면을 camera-only reconstruction으로 완전 복원했다는 주장은 하지 않는다.

## 11. 단계 7: Collision / Replay Diagnostics 성공

사고 replay diagnostic은 proxy vehicle footprints의 center distance와 clearance를 계산하는 방식으로 구현했다.

출력 경로:

- `outputs/town04_type1_subtype2_slam3r_incremental_layers/replay/accident_diagnostics.json`

성공 내용:

| 항목 | 값 |
|---|---|
| Method | `calibration_pose_center_distance_with_proxy_vehicle_radii` |
| Proxy only | true |
| Pair records | 294 |
| Closest frame | 49 |
| Closest pair | `ego_vehicle_behind`, `other_vehicle` |
| Center distance 3D | 4.1967 m |
| Center distance XY | 4.1956 m |
| Proxy clearance XY | -0.6842 m |

Sequence summary:

| Phase | Frame | Proxy clearance XY |
|---|---:|---:|
| pre-closest | 47 | 2.2348 m |
| closest | 49 | -0.6842 m |
| post-closest | 49 | -0.6842 m |

해석:

- Frame 49에서 `ego_vehicle_behind`와 `other_vehicle` proxy footprint가 겹치는 것으로 계산된다.
- Negative clearance는 proxy footprint overlap을 의미한다.
- 이 값은 scene reconstruction과 replay 해석을 위한 structured geometric context다.
- 법적 crash proof, physical crash-force proof, exact impact surface evidence로 해석하지 않는다.

## 12. 단계 8: Car-Like Vehicle Model 및 방향 수정 성공

초기 차량 표현은 속이 빈 box처럼 보여서 사고 replay 가독성이 낮았다. 이를 실제 차량처럼 보이는 3D proxy로 교체하고, 차량 방향이 옆으로 서 보이는 문제를 수정했다.

반영 위치:

- `scripts/build_slam3r_incremental_layers.py`
- `scripts/run_multicam_world_reconstruction.py`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html`

차량 proxy 구성 요소:

- body
- hood
- trunk
- cabin
- roof
- windshield
- rear window
- side windows
- side trim
- front/rear bumpers
- wheels
- rims
- grille
- headlights
- tail lights

방향 수정:

- Vehicle local forward axis를 `+X` 기준으로 맞췄다.
- Frame-wise movement direction과 차량 앞 방향이 일치하도록 yaw 적용을 정리했다.

Heading verification:

| Vehicle | Mean heading error | Median heading error | Max heading error |
|---|---:|---:|---:|
| `ego_vehicle` | 0.003 deg | 0.003 deg | 0.013 deg |
| `ego_vehicle_behind` | 0.035 deg | 0.002 deg | 1.235 deg |
| `other_vehicle` | 0.138 deg | 0.014 deg | 2.364 deg |
| `other_vehicle_behind` | 0.002 deg | 0.001 deg | 0.008 deg |

이 단계의 의미:

- 차량이 더 이상 빈 박스처럼 보이지 않는다.
- 차량이 옆으로 서 있는 orientation 문제를 해결했다.
- Car-like proxy는 사고 차량의 위치, 방향, 크기, 움직임을 사람이 이해하기 위한 visual carrier다.

## 13. 단계 9: No-LiDAR Audit 성공

No-LiDAR audit은 여러 산출물에서 pass/ok로 확인되었다.

대표 audit 파일:

- `outputs/town04_type1_subtype2_multicam_export/reports/no_lidar_audit.md`
- `outputs/town04_type1_subtype2_slam3r_reconstruction/reports/no_lidar_audit.md`
- `outputs/town04_type1_subtype2_readable/reports/no_lidar_audit.md`
- `outputs/town04_type1_subtype2_slam3r_incremental_layers/manifest.json`

확인된 내용:

| 항목 | 결과 |
|---|---|
| RGB camera frames used | true |
| DeepAccident calibration used | true |
| Dynamic masks used when available | true |
| `lidar01` point clouds read | false |
| Historical LiDAR PLY/GLB read | false |
| Simulator map mesh read | false |
| Proxy meshes used as background geometry | false |
| Calibration matrices with `lidar_to_*` names used as camera extrinsics only | true |

## 14. 단계 10: 연구 문서화 성공

최종 reconstruction 결과를 바탕으로 기존 LiDAR 중심 설명을 camera-only 중심 연구 문서로 수정했다. 현재 GitHub 커밋에는 논문 TEX/PDF/번역 파일을 포함하지 않는다.

추적되는 문서 경로:

- `README.md`
- `QUICKSTART.md`
- `docs/method_summary.md`
- `docs/dataset.md`
- `docs/artifacts.md`
- `docs/research_success_results_ko.md`
- `docs/camera_only_success_results_full_ko.md`
- `research_camera_only/reports/completion_audit.md`
- `research_camera_only/reports/evidence_report.md`
- `research_camera_only/reports/validation_checklist.md`

문서 수정 방향:

- Main claim을 LiDAR-camera fusion에서 camera-only RGB reconstruction으로 전환했다.
- DeepAccident calibration 사용을 명시했다.
- `lidar01` point cloud를 primary method에서 사용하지 않았음을 명시했다.
- SLAM3R 설명을 RGB point-map/ray-distance + calibration placement로 정리했다.
- Dynamic object 처리 표현을 `mask-assisted / when available`로 조정했다.
- LiDAR-assisted rendering은 primary sensor가 아니라 supplementary readability/reference visualization으로 재배치했다.
- Frame 49 proxy clearance는 replay diagnostic이지 physical crash-force proof가 아님을 명시했다.

문서 산출물 검증:

| 항목 | 결과 |
|---|---|
| Markdown research summaries | present |
| No tracked paper files | true |
| No tracked GLB/image files | true |
| Camera-only main claim | documented |

## 15. 최종 연구 설명에 넣을 수 있는 핵심 수치

| 항목 | 값 |
|---|---:|
| Scenario | Town04 type1-subtype2 scenario00017 |
| Agents | 4 |
| Cameras | 6 per vehicle |
| RGB frames | 1176 |
| Base reconstruction points | 545,306 |
| Incremental final points | 1,508,944 |
| Added points through layering | 963,638 |
| Base accepted streams | 10 / 24 |
| Base alignment RMSE | 1.4469 m |
| Base alignment median | 0.7856 m |
| Object context records | 6,992 |
| Vehicle replay samples | 49 per vehicle |
| Diagnostic pair records | 294 |
| Closest diagnostic frame | 49 |
| Closest diagnostic pair | `ego_vehicle_behind` / `other_vehicle` |
| Proxy clearance at closest frame | -0.6842 m |
| LiDAR point cloud used | false |

## 16. 최종 산출물 파일 크기

아래는 로컬 실험에서 생성된 성공 산출물의 크기 기록이다. GitHub 커밋에는 GLB와
image binary를 포함하지 않는다.

| 파일 | 크기 |
|---|---:|
| `outputs/town04_type1_subtype2_slam3r_reconstruction/reconstruction/points_world.ply` | 8,179,770 bytes |
| `outputs/town04_type1_subtype2_slam3r_reconstruction/scene.glb` | 8,725,844 bytes |
| `outputs/town04_type1_subtype2_slam3r_reconstruction/viewer/index.html` | 12,223 bytes |
| `outputs/town04_type1_subtype2_slam3r_incremental_layers/reconstruction/points_world.ply` | 22,634,341 bytes |
| `outputs/town04_type1_subtype2_slam3r_incremental_layers/scene.glb` | 24,144,060 bytes |
| `outputs/town04_type1_subtype2_slam3r_incremental_layers/viewer/index.html` | 16,094 bytes |
| `outputs/town04_type1_subtype2_readable/scene.glb` | 21,357,336 bytes |
| `outputs/town04_type1_subtype2_readable/viewer/index.html` | 7,750 bytes |

Preview assets generated locally but not tracked:

| 파일 | 크기 |
|---|---:|
| `outputs/town04_type1_subtype2_slam3r_reconstruction/reports/preview_point_cloud.png` | 396,559 bytes |
| `outputs/town04_type1_subtype2_slam3r_incremental_layers/reports/preview_point_cloud.png` | 472,285 bytes |
| `outputs/town04_type1_subtype2_slam3r_incremental_layers/reports/topdown_preview.png` | 704,120 bytes |
| `outputs/town04_type1_subtype2_readable/reports/readable_topdown_preview.png` | 501,168 bytes |

## 17. 성공 결과의 해석 범위

이번 성공 결과를 표현할 때 지켜야 할 경계는 다음이다.

정확한 표현:

- “The primary reconstruction path uses RGB frames and DeepAccident calibration.”
- “LiDAR point clouds are not read by the primary reconstruction path.”
- “SLAM3R RGB point-map predictions are placed in the shared world with calibration.”
- “Vehicles are represented as dimension-aware calibrated proxies.”
- “Proxy clearance is a replay diagnostic, not physical crash-force proof.”
- “Supplementary readability visualizations can be used to make sparse road or urban context easier to inspect, but they are not used for reported metrics.”

피해야 할 표현:

- “The city background is reconstructed from LiDAR.”
- “Dashcam images only colorize LiDAR point clouds.”
- “Vehicle surfaces are fully reconstructed from ordinary video.”
- “Frame 49 proves the physical crash force or exact legal impact.”
- “The camera-only output is a complete photorealistic reconstruction.”

기술적 caveat:

- SLAM3R은 monocular RGB reconstruction이므로 metric scale과 world placement에 calibration이 필요하다.
- SLAM3R input resizing/cropping 때문에 wide-angle border content 일부는 reconstruction에 충분히 반영되지 않을 수 있다.
- Dynamic masks가 있는 frame에서는 moving objects를 더 잘 억제하지만, mask가 없는 frame에는 transient geometry가 남을 수 있다.
- Textureless asphalt, distant facade, rain-affected region은 pure RGB reconstruction에서 sparse하게 보일 수 있다.
- Vehicle proxy는 trajectory와 clearance 해석을 돕는 표현이며, crash deformation/contact force/exact impact surface를 모델링하지 않는다.

## 18. 발표/저장소 설명에서 사용할 수 있는 최종 메시지

한국어:

> ReSceneScribe는 일반 차량에서 확보 가능한 vehicle-mounted RGB 영상과 calibration만으로 사고 현장의 static 3D context와 dynamic vehicle replay를 구성할 수 있음을 보였다. DeepAccident Town04 4대 차량 사고 시나리오에서 primary path는 `lidar01` point cloud를 사용하지 않고 1176개 RGB frame을 처리했으며, 545,306-point base scene과 1,508,944-point incremental scene을 생성했다. 차량은 calibration pose 기반 car-like proxy로 replay되어 사고 과정과 closest proxy interaction을 frame-indexed diagnostic으로 설명한다.

영어:

> ReSceneScribe demonstrates that accident-scene 3D reconstruction and replay can be organized around ordinary vehicle-mounted RGB videos and calibration. In the DeepAccident Town04 four-vehicle accident scenario, the primary path processes 1176 RGB frames without reading `lidar01` point clouds, producing a 545,306-point base scene and a 1,508,944-point incremental scene. Vehicles are replayed as calibration-driven car-like proxies, enabling frame-indexed diagnostics for interpreting the accident process.
