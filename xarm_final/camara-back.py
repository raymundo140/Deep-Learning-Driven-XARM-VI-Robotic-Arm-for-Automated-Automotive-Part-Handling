#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class BlackObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('black_object_detector')
        self.subscription = self.create_subscription(
            Image,
            '/rgb/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Black object detector node started.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # === Black Mask ===
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        # Clean up black mask
        kernel = np.ones((3, 3), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        # === White Mask ===
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 80, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_found = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                # Create mask of this black object
                obj_mask = np.zeros_like(black_mask)
                cv2.drawContours(obj_mask, [cnt], -1, 255, -1)

                # Extract white inside this object only
                white_inside = cv2.bitwise_and(white_mask, white_mask, mask=obj_mask)
                white_pixels = cv2.countNonZero(white_inside)

                if white_pixels > 0:
                    object_found = True
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
    
                        # Draw result
                        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(frame, "✔️ Black + White Object", (cx - 50, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    break

        if not object_found:
            self.get_logger().info("No black object with white inside found.")

        # === Show all views ===
        cv2.imshow("📷 RGB Detection", frame)
        cv2.imshow("🖤 Black Mask", black_mask)
        cv2.imshow("⚪ White Mask", white_mask)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = BlackObjectDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
