#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import open3d as o3d
import cv2
import numpy as np

class RGBDToPlyManual(Node):
    def __init__(self):
        super().__init__('rgbd_to_ply_manual')

        self.bridge = CvBridge()
        self.color_frame = None
        self.depth_frame = None
        self.saved = False

        # Hard-coded intrinsics for 640×576 NFOV unbinned
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width= 640,
            height=576,
            fx=616.36529541,
            fy=616.36529541,
            cx=320.0,
            cy=288.0
        )

        # subs to the two Image topics
        self.create_subscription(
            Image, '/rgb/image_raw', self.color_cb, 10)
        self.create_subscription(
            Image, '/depth_to_rgb/image_raw', self.depth_cb, 10)

    def color_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except CvBridgeError:
            pass
        self.try_save()

    def depth_cb(self, msg: Image):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        except CvBridgeError:
            pass
        self.try_save()

    def try_save(self):
        if self.saved:
            return
        if self.color_frame is None or self.depth_frame is None:
            return

        # Build Open3D images
        color_o3d = o3d.geometry.Image(self.color_frame)
        depth_o3d = o3d.geometry.Image(self.depth_frame)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0, depth_trunc=3.0,
            convert_rgb_to_intensity=False)

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.intrinsics)

        # Save out to a PLY
        o3d.io.write_point_cloud("output.ply", pcd)
        self.get_logger().info("✅ Saved output.ply")
        self.saved = True

        # exit after saving
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RGBDToPlyManual()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
