# Dataset Setup

ReSceneScribe rebuild scripts expect a DeepAccident-style local dataset root.
The raw dataset is not redistributed in this repository.

Set:

```bash
export DEEPACCIDENT_ROOT=/path/to/deepaccident_mini_dataset
```

Expected scenario:

```text
type1_subtype1_accident/
├── meta/Town03_type001_subtype0001_scenario00024.txt
├── ego_vehicle/
│   ├── Camera_Front/Town03_type001_subtype0001_scenario00024/*.jpg
│   ├── Camera_FrontLeft/Town03_type001_subtype0001_scenario00024/*.jpg
│   ├── Camera_FrontRight/Town03_type001_subtype0001_scenario00024/*.jpg
│   ├── Camera_BackLeft/Town03_type001_subtype0001_scenario00024/*.jpg
│   ├── Camera_BackRight/Town03_type001_subtype0001_scenario00024/*.jpg
│   ├── Camera_Back/Town03_type001_subtype0001_scenario00024/*.jpg
│   ├── lidar01/Town03_type001_subtype0001_scenario00024/*.npz
│   ├── calib/Town03_type001_subtype0001_scenario00024/*.pkl
│   └── label/Town03_type001_subtype0001_scenario00024/*.txt
├── ego_vehicle_behind/
├── other_vehicle/
└── other_vehicle_behind/
```

The packaged experiment uses:

| Item | Value |
|---|---|
| Category | `type1_subtype1_accident` |
| Scenario | `Town03_type001_subtype0001_scenario00024` |
| Weather | `MidRainSunset` |
| Road type | four-way junction |
| Metadata colliding agents | `ego`, `other` |
| Available frames | `001`-`056` |

Each of the four agents is expected to have 56 files per sensor stream in the
mini subset used by this work.

## Override scenario arguments

All main scripts accept `--dataset`, `--category`, and `--scenario`:

```bash
python scripts/build_four_vehicle_collision_viewer.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --category type1_subtype1_accident \
  --scenario Town03_type001_subtype0001_scenario00024 \
  --output-root .
```

```bash
python scripts/build_town03_clean_hybrid_gaussian_ply.py \
  --dataset "$DEEPACCIDENT_ROOT" \
  --category type1_subtype1_accident \
  --scenario Town03_type001_subtype0001_scenario00024 \
  --out outputs/town03_4dashcam_collision_3dgs_45000.ply
```
