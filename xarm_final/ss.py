#!/usr/bin/env python3
import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2   # ← moved to module level

class DataSaver(Node):
    def __init__(self):
        super().__init__('data_saver')
        self.bridge = CvBridge()

        # prepare folders
        for d in ['color', 'depth', 'camera']:
            os.makedirs(d, exist_ok=True)

        # subscribe to RGB, Depth, and CameraInfo
        rgb_sub   = Subscriber(self, Image,       '/rgb/image_raw')
        depth_sub = Subscriber(self, Image,       '/depth_to_rgb/image_raw')
        self.create_subscription(
            CameraInfo, '/rgb/camera_info',
            self.caminfo_cb, 10)

        self.intrinsics = None
        ats = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.02)
        ats.registerCallback(self.cb)

    def caminfo_cb(self, msg: CameraInfo):
        if self.intrinsics is None:
            # build a plain-Python dict (no numpy scalars)
            self.intrinsics = {
                'width':  int(msg.width),
                'height': int(msg.height),
                'fx':     float(msg.k[0]),
                'fy':     float(msg.k[4]),
                'cx':     float(msg.k[2]),
                'cy':     float(msg.k[5]),
                'distortion_model': msg.distortion_model,
                'D':     [float(d) for d in msg.d]
            }
            # use safe_dump to avoid non-standard tags
            with open(os.path.join('camera','intrinsics.yaml'), 'w') as f:
                yaml.safe_dump(self.intrinsics, f)

    def cb(self, rgb_msg, depth_msg):
        if self.intrinsics is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return

        ts = rgb_msg.header.stamp.sec * 1e9 + rgb_msg.header.stamp.nanosec
        fname = f"{int(ts)}.png"

        # color
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        cv2.imwrite(
            os.path.join('color', fname.replace('.png','.jpg')),
            rgb
        )

        # depth
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        cv2.imwrite(
            os.path.join('depth', fname),
            depth
        )

        self.get_logger().info(f"Saved frame {fname}")

def main():
    rclpy.init()
    node = DataSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
