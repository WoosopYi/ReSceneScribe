# Dataset Setup

ReSceneScribe rebuild scripts expect a local DeepAccident-style dataset root.
Raw DeepAccident data are not redistributed in this repository.

Set:

```bash
export DEEPACCIDENT_ROOT=/path/to/deepaccident_mini_dataset
```

## Main Camera-Only Scenario

The current repository centers on:

| Item | Value |
|---|---|
| Category | `type1_subtype2_accident` |
| Scenario | `Town04_type001_subtype0002_scenario00017` |
| Agents | `ego_vehicle`, `ego_vehicle_behind`, `other_vehicle`, `other_vehicle_behind` |
| Cameras per agent | `6` |
| Selected frames | `001`-`049` |
| RGB views used | `1176` |
| Calibration use | camera/world placement, scale alignment, vehicle replay |
| LiDAR point cloud use | `false` |

Expected camera/calibration layout:

```text
type1_subtype2_accident/
├── meta/Town04_type001_subtype0002_scenario00017.txt
├── ego_vehicle/
│   ├── Camera_Front/Town04_type001_subtype0002_scenario00017/*.jpg
│   ├── Camera_FrontLeft/Town04_type001_subtype0002_scenario00017/*.jpg
│   ├── Camera_FrontRight/Town04_type001_subtype0002_scenario00017/*.jpg
│   ├── Camera_Back/Town04_type001_subtype0002_scenario00017/*.jpg
│   ├── Camera_BackLeft/Town04_type001_subtype0002_scenario00017/*.jpg
│   ├── Camera_BackRight/Town04_type001_subtype0002_scenario00017/*.jpg
│   ├── calib/Town04_type001_subtype0002_scenario00017/*.pkl
│   └── label/Town04_type001_subtype0002_scenario00017/*.txt
├── ego_vehicle_behind/
├── other_vehicle/
└── other_vehicle_behind/
```

`label` files are used for vehicle dimensions and object context. They are not
used as LiDAR geometry.

## Calibration Notes

DeepAccident calibration files may include matrix names such as
`lidar_to_Camera_Front`. In the camera-only path these matrices are used to
recover camera extrinsics and place camera-derived RGB geometry in the world.
The `lidar01/*.npz` point cloud files are not read by the primary reconstruction
scripts.

## Main Rebuild Commands

```bash
python scripts/run_multicam_world_reconstruction.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --category type1_subtype2_accident \
  --scenario Town04_type001_subtype0002_scenario00017 \
  --out outputs/town04_type1_subtype2_multicam_export \
  --frame-start 1 \
  --frame-end 49 \
  --frame-step 1
```

```bash
python scripts/run_slam3r_deepaccident_reconstruction.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --category type1_subtype2_accident \
  --scenario Town04_type001_subtype0002_scenario00017 \
  --mask-export outputs/town04_type1_subtype2_multicam_export \
  --out outputs/town04_type1_subtype2_slam3r_reconstruction \
  --slam3r-root third_party/SLAM3R
```

```bash
python scripts/build_slam3r_incremental_layers.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --source outputs/town04_type1_subtype2_slam3r_reconstruction \
  --mask-export outputs/town04_type1_subtype2_multicam_export \
  --out outputs/town04_type1_subtype2_slam3r_incremental_layers \
  --slam3r-root third_party/SLAM3R
```

## Legacy Scenario

The previous repository release used `type1_subtype1_accident` /
`Town03_type001_subtype0001_scenario00024` and LiDAR-camera fusion. That path
is kept under legacy scripts for comparison only.
