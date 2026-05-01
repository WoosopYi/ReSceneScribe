# Method Summary

ReSceneScribe separates accident reconstruction into two linked outputs:

1. A temporal web replay for human interpretation.
2. A collision-state hybrid Gaussian/RGB PLY for reusable 3D inspection.

## Static-dynamic decomposition

The static road environment and dynamic vehicles are handled separately. Static
background points are fused from all four DeepAccident vehicle agents after
removing points inside moving-object and target-vehicle boxes. This prevents
moving vehicles from becoming repeated ghost geometry in the road background.

## Shared metric alignment

Each LiDAR point is transformed into the CARLA world coordinate using:

```text
p_world = ego_to_world @ lidar_to_ego @ p_lidar_homogeneous
```

For the browser replay, the CARLA world is converted to the Three.js coordinate
system:

```text
viewer X = CARLA X
viewer Y = CARLA Z / up
viewer Z = CARLA Y
```

## Vehicle-local motion compensation

Vehicle geometry is reconstructed by collecting points inside each target
vehicle oriented bounding box. These points are transformed into the
vehicle-local frame, accumulated over frames and observing agents,
voxel-downsampled, and then transformed back to the final collision pose.

This avoids accumulating a moving vehicle directly in world coordinates, which
would create a trajectory-shaped vehicle trail.

## Collision diagnostics

The collision pair is validated with oriented bounding box overlap on the
ground plane. In the packaged scenario:

```text
overlap source frames: 054, 055, 056
selected collision-state frame: 056
final OBB gap: -0.447944 m
final center distance: 4.229926 m
```

A negative OBB gap means the two selected vehicle boxes overlap in the shared
metric world and is used as a geometric collision/contact proxy.

## Hybrid Gaussian/RGB PLY

The final PLY keeps standard RGB point-cloud fields and adds fields commonly
used by 3D Gaussian Splatting viewers:

```text
x, y, z,
nx, ny, nz,
f_dc_0, f_dc_1, f_dc_2,
f_rest_0 ... f_rest_44,
opacity,
scale_0, scale_1, scale_2,
rot_0, rot_1, rot_2, rot_3,
red, green, blue
```

The file is a deterministic LiDAR-camera fusion output with
Gaussian-compatible fields. It is not a fully optimized vanilla 3DGS training
checkpoint.
