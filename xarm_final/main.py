#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import yaml
from pathlib import Path

# directories (run this script from the same folder that has color/, depth/, camera/)
color_dir = Path('color')
depth_dir = Path('depth')
cam_yaml  = Path('camera/intrinsics.yaml')
out_dir   = Path('ply')
out_dir.mkdir(exist_ok=True)

# load intrinsics with FullLoader (to handle safe_dump output)
with open(cam_yaml, 'r') as f:
    intr = yaml.load(f, Loader=yaml.FullLoader)

pinhole = o3d.camera.PinholeCameraIntrinsic(
    intr['width'], intr['height'],
    intr['fx'], intr['fy'],
    intr['cx'], intr['cy']
)

for depth_file in sorted(depth_dir.iterdir()):
    stamp = depth_file.stem
    color_file = color_dir / f"{stamp}.jpg"

    # read images
    depth = o3d.io.read_image(str(depth_file))
    color = o3d.io.read_image(str(color_file))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1000.0,   # adjust if necessary
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    # back-project into a point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)

    # transform from Open3D camera coords to usual ROS frame
    pcd.transform([[1, 0, 0, 0],
                   [0,-1, 0, 0],
                   [0, 0,-1, 0],
                   [0, 0, 0, 1]])

    out_path = out_dir / f"{stamp}.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Wrote {out_path}")
