#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class ObjectFollowerXarm(Node):
    def __init__(self):
        super().__init__('object_follower_xarm')

        # ─── Tunable parameters ──────────────────────────────
        self.H_THRESH = (0.4, 0.6)   # left/centre/right bounds
        self.V_THRESH = (0.4, 0.6)   # top/centre/bottom bounds
        self.STEP     = 0.02         # radians per tick
        self.J1_LIMIT = (-1.57, 1.57)
        self.J3_LIMIT = (-0.8, 0.8)
        self.J2_LIMIT = (-1.2, 1.2)

        # ─── Publishers & Subscribers ────────────────────────
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/xarm6_traj_controller/joint_trajectory',
            10)

        # grab initial joint positions
        self.current_positions = [0.0]*6
        self.received_js = False
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            10)

        # subscribe to Kinect / webcam color stream
        self.bridge = CvBridge()
        self.cx = None
        self.cy = None
        self.obj_detected = False
        self.create_subscription(
            Image,
            '/rgb/image_raw',   # adjust to your camera topic
            self.image_cb,
            10)

        # Prepare windows
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Black Mask", cv2.WINDOW_NORMAL)

        # Timer to drive the arm
        self.create_timer(0.1, self.track_zone)

        # joint state for motion
        self.j1 = self.j2 = self.j3 = 0.0
        self.joint_names = [
            'joint1','joint2','joint3',
            'joint4','joint5','joint6']

        self.get_logger().info("🔄 Object‐follower node started — waiting for joint_states…")

    def joint_state_cb(self, msg: JointState):
        if not self.received_js and len(msg.position) >= 6:
            self.current_positions = list(msg.position[:6])
            # seed j1, j2, j3 so robot holds this pose
            self.j1, self.j2, self.j3 = (
                self.current_positions[0],
                self.current_positions[1],
                self.current_positions[2]
            )
            self.received_js = True
            self.get_logger().info(
                f"✅ Initial joints: {[f'{p:.3f}' for p in self.current_positions]}")

    def image_cb(self, msg: Image):
        # convert to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # build black mask
        black = cv2.inRange(hsv, (0,0,0), (180,255,60))
        kernel = np.ones((3,3),np.uint8)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN,  kernel)
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        # find all black contours
        contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.obj_detected = False

        if contours:
            # pick the largest
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 1000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx_pix = int(M['m10']/M['m00'])
                    cy_pix = int(M['m01']/M['m00'])
                    h, w = frame.shape[:2]
                    self.cx = cx_pix / w
                    self.cy = cy_pix / h
                    self.obj_detected = True

                    # draw feedback
                    cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
                    cv2.drawMarker(frame, (cx_pix, cy_pix),
                                   (0,0,255), cv2.MARKER_CROSS, 20, 2)

        # show views
        cv2.imshow("RGB", frame)
        cv2.imshow("Black Mask", black)
        cv2.waitKey(1)

    def track_zone(self):
        if not self.received_js:
            return

        if not self.obj_detected:
            # hold exactly at initial pose
            self.send_joint_command(self.j1, self.j2, self.j3)
            return

        x, y = self.cx, self.cy
        left, right = self.H_THRESH
        top, bottom = self.V_THRESH

        # ─ Horizontal → joint1 ─────────────────────
        if   x <  left:
            self.j1 = np.clip(self.j1 - self.STEP, *self.J1_LIMIT)
        elif x > right:
            self.j1 = np.clip(self.j1 + self.STEP, *self.J1_LIMIT)

        # ─ Vertical → joint3 until limit, then joint2 ─
        if y < top:
            if   self.j3 < self.J3_LIMIT[1]:
                self.j3 = np.clip(self.j3 + self.STEP, *self.J3_LIMIT)
            else:
                self.j2 = np.clip(self.j2 + self.STEP, *self.J2_LIMIT)
        elif y > bottom:
            if   self.j3 > self.J3_LIMIT[0]:
                self.j3 = np.clip(self.j3 - self.STEP, *self.J3_LIMIT)
            else:
                self.j2 = np.clip(self.j2 - self.STEP, *self.J2_LIMIT)

        self.send_joint_command(self.j1, self.j2, self.j3)

    def send_joint_command(self, j1, j2, j3):
        traj = JointTrajectory(joint_names=self.joint_names)
        pts  = self.current_positions.copy()
        pts[0], pts[1], pts[2] = j1, j2, j3

        p = JointTrajectoryPoint()
        p.positions = pts
        p.time_from_start.sec = 1
        traj.points.append(p)

        self.traj_pub.publish(traj)
        self.get_logger().info(f"Sent → {[f'{v:.2f}' for v in pts]}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollowerXarm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔌 Shutting down…")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
