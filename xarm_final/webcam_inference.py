#!/usr/bin/env python3
import os
# Disable GPU to avoid any CUDA/PTX mismatches
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import Regularizer

# Register your custom regularizer so Keras can deserialize it
@tf.keras.utils.register_keras_serializable(package='Custom', name='OrthogonalRegularizer')
class OrthogonalRegularizer(Regularizer):
    def __init__(self, coeff=1e-4):
        self.coeff = coeff

    def __call__(self, x):
        x_flat = tf.reshape(x, (x.shape[0], -1))
        ident   = tf.eye(x_flat.shape[0])
        wwt     = tf.matmul(x_flat, x_flat, transpose_b=True)
        return self.coeff * tf.reduce_sum(tf.square(wwt - ident))

    def get_config(self):
        return {'coeff': self.coeff}


class WebcamKerasInferenceNode(Node):
    def __init__(self):
        super().__init__('webcam_inference')

        # 1) Locate the model in share/xarm_final/models
        share_dir  = get_package_share_directory('xarm_final')
        model_path = os.path.join(share_dir, 'models', 'my_model.h5')
        if not os.path.isfile(model_path):
            self.get_logger().error(f"Missing model file: {model_path}")
            rclpy.shutdown()
            return

        # 2) Load the full Keras model (architecture + weights + custom reg)
        self.model = load_model(model_path)

        # 3) Open the webcam (Linux): try index 0, then 1
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().warn("Failed to open camera 0, trying camera 1…")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam on index 0 or 1")
            rclpy.shutdown()
            return

        # 4) Schedule at ~15 Hz
        self.timer = self.create_timer(1.0 / 15.0, self.timer_callback)
        self.get_logger().info("Webcam inference node started (CPU-only).")

    def timer_callback(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Preprocess to model’s input
        _, H, W, _ = self.model.input_shape
        img = cv2.resize(frame, (W, H)).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        # Run inference
        preds = self.model.predict(img, verbose=0)[0]
        idx, prob = int(np.argmax(preds)), float(np.max(preds))

        # Overlay result
        label = f"Class {idx}: {prob*100:.1f}%"
        cv2.putText(frame, label, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Keras Inference (CPU)", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamKerasInferenceNode()
    if rclpy.ok():
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
