# Quickstart

## Open the packaged replay

```bash
git clone https://github.com/WoosopYi/ReSceneScribe.git
cd ReSceneScribe
python3 scripts/serve_viewer.py --port 8132
```

Open:

```text
http://127.0.0.1:8132/viewer/index.html?quality=ultra&trail
```

## Download the large PLY artifact

The PLY is distributed as a release asset:

```bash
mkdir -p outputs
gh release download v0.1-artifacts \
  --repo WoosopYi/ReSceneScribe \
  --pattern 'town03_4dashcam_collision_3dgs_45000.ply' \
  --dir outputs
```

Verify it:

```bash
python3 scripts/verify_ply.py outputs/town03_4dashcam_collision_3dgs_45000.ply
sha256sum outputs/town03_4dashcam_collision_3dgs_45000.ply
```

Expected SHA-256:

```text
cb0edd5028bc8d81ebcc8a557b39c4a4008ed74226beb486cd8dbd5a1bd1bc55
```

## Rebuild from a local DeepAccident copy

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export DEEPACCIDENT_ROOT=/path/to/deepaccident_mini_dataset
make rebuild-viewer
make rebuild-ply
```

The rebuild commands assume the local dataset contains:

```text
type1_subtype1_accident/
├── ego_vehicle/
├── ego_vehicle_behind/
├── other_vehicle/
├── other_vehicle_behind/
└── meta/
```
