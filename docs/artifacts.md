# Artifacts

## Packaged viewer assets

The interactive replay assets are tracked in git so a clone can open the viewer
without rebuilding from DeepAccident.

| File | Size | SHA-256 |
|---|---:|---|
| `viewer_assets/four_vehicle_static_lidar_background.glb` | 16,800,960 bytes | `f8f30a25769c661791be800f9d44aaeb35a46d53e3c52283763aa1e2abc770db` |
| `viewer_assets/four_vehicle_static_lidar_background_ultra.glb` | 35,200,960 bytes | `4b68c5715028212f181468bd9fc962dadd9fc8ff281dfe7204e6d2b1ad9f1df8` |

The viewer also includes 224 resized front dashcam JPEG frames:

```text
4 agents x 56 frames = 224 frames
```

## Large PLY release asset

The final collision-state PLY is not tracked in git because it is larger than
GitHub's regular file-size limit.

| File | Size | SHA-256 |
|---|---:|---|
| `town03_4dashcam_collision_3dgs_45000.ply` | 295,430,851 bytes / 281.74 MiB | `cb0edd5028bc8d81ebcc8a557b39c4a4008ed74226beb486cd8dbd5a1bd1bc55` |

Download after the release is available:

```bash
mkdir -p outputs
gh release download v0.1-artifacts \
  --repo WoosopYi/ReSceneScribe \
  --pattern 'town03_4dashcam_collision_3dgs_45000.ply' \
  --dir outputs
```

Verify:

```bash
python3 scripts/verify_ply.py outputs/town03_4dashcam_collision_3dgs_45000.ply --hash
```

Expected structural values:

| Metric | Value |
|---|---:|
| Vertices | 1,177,009 |
| Vertex properties | 65 |
| Static background points | 1,100,000 |
| Vehicle points | 77,009 |
| RGB fields | yes |
| 3DGS-compatible fields | yes |

Segment order:

| Segment | Index range | Count |
|---|---:|---:|
| Static background | `0`-`1,099,999` | 1,100,000 |
| `ego_vehicle` | `1,100,000`-`1,128,978` | 28,979 |
| `ego_vehicle_behind` | `1,128,979`-`1,137,002` | 8,024 |
| `other_vehicle` | `1,137,003`-`1,165,891` | 28,889 |
| `other_vehicle_behind` | `1,165,892`-`1,177,008` | 11,117 |
