# ReSceneScribe Camera-Only 연구 성공 결과 요약

작성일: 2026-05-16
상세 문서: `docs/camera_only_success_results_full_ko.md`

## 핵심 결론

현재 GitHub 저장소의 메인 연구는 기존 Town03 LiDAR-camera fusion 결과가
아니라, DeepAccident Town04 type1-subtype2 사고 장면을 사용한
대시캠/RGB 중심 3D reconstruction 결과다.

최종 논문 제목과 맞춘 저장소 방향은 다음이다.

> ReSceneScribe: Static-Dynamic 3D Accident Reconstruction from
> Vehicle-Mounted Sensing and Multi-Agent Simulation for Evidence-Grounded
> Investigation

논문에서는 두 가지 실험 역할을 분리한다. 실제 차량 2대와 windshield-area RGB
camera를 사용한 physical pilot은 vehicle-mounted visual evidence의 capture
feasibility를 확인하는 단계이고, 정량 reconstruction/replay 결과는 calibration과
vehicle metadata가 제공되는 DeepAccident Town04 benchmark에서 보고한다.

주 방법은 다음 입력만을 primary reconstruction path로 사용한다.

- 4대 차량의 vehicle-mounted RGB camera frames
- 차량당 6개 camera stream
- 총 1176개 RGB frame
- DeepAccident calibration
- dynamic mask가 존재하는 경우 mask
- SLAM3R RGB point-map prediction과 confidence map
- vehicle replay를 위한 frame-wise calibration pose와 차량 치수

주 방법에서 사용하지 않은 것은 다음이다.

- `lidar01` point cloud
- 기존 LiDAR-derived PLY/GLB
- simulator map mesh
- legacy viewer asset을 background geometry로 재사용하는 경로

## 사용 시나리오

| 항목 | 값 |
|---|---|
| Dataset | DeepAccident mini/local |
| Category | `type1_subtype2_accident` |
| Scenario | `Town04_type001_subtype0002_scenario00017` |
| Agents | `ego_vehicle`, `ego_vehicle_behind`, `other_vehicle`, `other_vehicle_behind` |
| Cameras | `Camera_Front`, `Camera_FrontLeft`, `Camera_FrontRight`, `Camera_Back`, `Camera_BackLeft`, `Camera_BackRight` |
| Selected frames | 1-49 |
| Total RGB views | 1176 |
| Calibration use | camera/world placement, scale alignment, vehicle replay |
| LiDAR use | false |

## 성공 산출물

| 단계 | 산출물 | 핵심 결과 |
|---|---|---:|
| Multi-camera known-pose export | RGB/mask/calibration package | 1176 RGB views |
| Camera-calibrated background proxy | visual background check | 1,400,000 points |
| Readable road/city viewer | road-frame context | ground texture coverage 0.579 |
| SLAM3R camera-only base | primary base reconstruction | 545,306 points |
| Incremental layering | final cumulative scene | 1,508,944 points |
| Vehicle replay | calibrated proxy replay | 4 tracks, 49 samples each |
| Collision diagnostic | proxy clearance | frame 49, -0.6842 m |
| Repository documentation | camera-only research summary | documented |

## Base Reconstruction

Base reconstruction은 `slam3r_camera_ray_depth_calibrated` backend를 사용했다.
24개 RGB stream 중 alignment 품질 기준을 통과한 10개 stream만 base fusion에
사용했고, 나머지 stream은 억지로 넣지 않았다. 이 선택이 잘못된 geometry 확산을
줄이는 데 중요했다.

| 항목 | 값 |
|---|---:|
| Input RGB frames | 1176 |
| Accepted streams | 10 / 24 |
| Exported points | 545,306 |
| Alignment camera centers | 490 |
| Alignment RMSE | 1.4469 m |
| Alignment median error | 0.7856 m |
| Object context records | 6,992 |
| LiDAR used | false |

## Incremental Layering

사용자가 요청한 "좋은 결과를 base로 잡고 조금씩 쌓는 방식"은 cumulative voxel
acceptance로 구현했다. 기존 cumulative scene의 0.08 m voxel을 침범하지 않는
point만 추가했다.

| Stage | 설명 | New points | Cumulative |
|---|---|---:|---:|
| `00_base` | 안정적인 no-LiDAR SLAM3R base | 545,306 | 545,306 |
| `01_01_masked_dense` | masked dense layer | 541,877 | 1,087,183 |
| `02_02_all_aligned_frames` | 추가 aligned frames | 303,076 | 1,390,259 |
| `03_03_strict_extra_streams` | strict extra streams | 118,685 | 1,508,944 |

## 차량 표현

차량은 영상으로 완전한 surface geometry를 복원했다고 주장하지 않는다. 대신
DeepAccident calibration pose와 vehicle dimensions를 기반으로 한 car-like 3D
proxy로 replay한다. 현재 proxy는 body, cabin, windows, bumpers, wheels,
headlights, tail lights 등을 포함한다.

이 표현은 사고 차량의 위치, 방향, 크기, 움직임을 설명하기 위한 시각적 carrier다.
법적 crash-force proof 또는 완전한 차량 표면 복원으로 해석하면 안 된다.

## 저장소 방향

저장소의 main claim은 "LiDAR로 도시 배경을 만들고 RGB로 색을 입혔다"가 아니다.
현재 방향은 다음이다.

> 일반 차량에서 확보 가능한 대시캠/RGB 영상과 calibration을 기반으로 사고 현장의
> 정적 3D 맥락을 만들고, 차량 움직임은 calibrated car-like proxy로 replay한다.

기존 Town03 LiDAR 결과는 버리지 않고, 도시/도로 맥락을 더 읽기 쉽게 보여주는
supplementary readability/reference visualization으로만 둔다. 이 보조 view는
정량 point count, alignment statistic, replay track, proxy clearance 계산에는
사용하지 않는다.
