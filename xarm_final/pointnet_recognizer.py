#!/usr/bin/env python3
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from ament_index_python.packages import get_package_share_directory
from tensorflow.keras.models import load_model

class PointNetRecognizer(Node):
    def __init__(self):
        super().__init__('pointnet_recognizer')

        # 1) locate model.h5 in the package share directory
        pkg_share = get_package_share_directory('xarm_final')
        model_path = os.path.join(pkg_share, 'my_model.h5')
        self.get_logger().info(f'Loading PointNet model from: {model_path}')
        self.model = load_model(model_path)
        self.get_logger().info('Model loaded successfully.')

        # 2) subscribe to your point-cloud stream
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',   # ← change to your PC2 topic if different
            self.pc_callback,
            1
        )

        # 3) open the default webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open webcam!')
            rclpy.shutdown()

        # store the latest prediction
        self.latest_label = '—'

        # 4) timer to update the display at 10 Hz
        self.create_timer(0.1, self.timer_callback)

    def pc_callback(self, msg: PointCloud2):
        # convert incoming PointCloud2 → list of (x,y,z)
        pts = []
        for p in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
            if len(pts) >= 1024:
                break

        # pad / truncate to exactly 1024 points
        if len(pts) < 1024:
            pts.extend([[0.0,0.0,0.0]] * (1024 - len(pts)))
        pts = np.array(pts, dtype=np.float32)       # shape (1024,3)
        pts = np.expand_dims(pts, axis=0)           # shape (1,1024,3)

        # normalize if needed (uncomment if your model expects it)
        # pts = (pts - np.mean(pts, axis=1, keepdims=True)) / np.std(pts, axis=1, keepdims=True)

        # run inference
        preds = self.model.predict(pts)             # shape (1, num_classes)
        class_id = int(np.argmax(preds, axis=1)[0])

        # map to your labels
        label_map = {
            0: 'chair',
            1: 'table',
            2: 'sofa',
            # … add your other classes …
        }
        self.latest_label = label_map.get(class_id, f'class_{class_id}')

    def timer_callback(self):
        # grab a frame from webcam
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Webcam frame not received')
            return

        # overlay the latest prediction
        cv2.putText(
            frame,
            f'Prediction: {self.latest_label}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # show it
        cv2.imshow('Webcam + PointNet', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        # cleanup on shutdown
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PointNetRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    