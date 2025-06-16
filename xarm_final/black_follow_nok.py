#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class WebcamObjectFollowerXarm(Node):
    def __init__(self):
        super().__init__('webcam_object_follower_xarm')

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

        # ─── Webcam setup ────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open webcam!")
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Black Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)

        # ─── Joint state for motion ──────────────────────────
        self.j1 = self.j2 = self.j3 = 0.0
        self.joint_names = [
            'joint1','joint2','joint3',
            'joint4','joint5','joint6']

        # ─── Timer to process image + drive arm ─────────────
        self.create_timer(0.1, self.track_zone)
        self.get_logger().info("🔄 Webcam-object-follower started — waiting for joint_states…")

    def joint_state_cb(self, msg: JointState):
        if not self.received_js and len(msg.position) >= 6:
            self.current_positions = list(msg.position[:6])
            self.j1, self.j2, self.j3 = (
                self.current_positions[0],
                self.current_positions[1],
                self.current_positions[2]
            )
            self.received_js = True
            self.get_logger().info(
                f"✅ Initial joints: {[f'{p:.3f}' for p in self.current_positions]}")

    def track_zone(self):
        if not self.received_js:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Webcam frame dropped.")
            return

        # flip so it feels natural
        frame = cv2.flip(frame, 1)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # — Black mask —
        black = cv2.inRange(hsv, (0,0,0), (180,255,60))
        kernel = np.ones((3,3),np.uint8)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN,  kernel)
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        # — White mask —
        white = cv2.inRange(hsv, (0,0,160), (180,80,255))

        # find black contours
        contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj_found = False

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            mask_region = np.zeros_like(black)
            cv2.drawContours(mask_region, [cnt], -1, 255, -1)
            inside_white = cv2.bitwise_and(white, white, mask=mask_region)
            wc, _ = cv2.findContours(inside_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(wc) < 2:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx_pix = int(M['m10']/M['m00'])
            cy_pix = int(M['m01']/M['m00'])
            h, w = frame.shape[:2]
            cx = cx_pix / w
            cy = cy_pix / h
            obj_found = True

            # draw feedback
            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
            cv2.drawMarker(frame, (cx_pix, cy_pix), (0,0,255), cv2.MARKER_CROSS, 20, 2)
            break

        # show all views
        cv2.imshow("RGB", frame)
        cv2.imshow("Black Mask", black)
        cv2.imshow("White Mask", white)
        cv2.waitKey(1)

        # ── Drive logic ────────────────────────────────────
        if not obj_found:
            # hold home pose
            self.send_joint_command(self.j1, self.j2, self.j3)
            return

        # left/centre/right → joint1
        l, r = self.H_THRESH
        if   cx < l:
            self.j1 = np.clip(self.j1 - self.STEP, *self.J1_LIMIT)
        elif cx > r:
            self.j1 = np.clip(self.j1 + self.STEP, *self.J1_LIMIT)

        # up/centre/down → joint3 until limit, then joint2
        t, b = self.V_THRESH
        if cy < t:
            if self.j3 < self.J3_LIMIT[1]:
                self.j3 = np.clip(self.j3 + self.STEP, *self.J3_LIMIT)
            else:
                self.j2 = np.clip(self.j2 + self.STEP, *self.J2_LIMIT)
        elif cy > b:
            if self.j3 > self.J3_LIMIT[0]:
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
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamObjectFollowerXarm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔌 Shutting down…")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
