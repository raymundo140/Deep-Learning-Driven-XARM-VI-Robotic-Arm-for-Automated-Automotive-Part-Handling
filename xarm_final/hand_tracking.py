#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class HandTrackerXarm(Node):
    def __init__(self):
        super().__init__('hand_tracker_xarm')

        # --- PARAMETERS you can tweak ---
        self.H_THRESH = (0.4, 0.6)       # left/right bounds (x)
        self.V_THRESH = (0.4, 0.6)       # top/bottom bounds (y)
        self.STEP     = 0.02             # rad per tick
        self.J1_LIMIT = (-1.57, 1.57)
        self.J3_LIMIT = (-0.8, 0.8)
        self.J2_LIMIT = (-1.2, 1.2)      # adjust as needed

        # publisher for commanding the arm
        self.pub = self.create_publisher(
            JointTrajectory,
            '/xarm6_traj_controller/joint_trajectory',
            10)

        # subscribe to joint_states to grab the initial “hold” pose
        self.current_positions = [0.0]*6
        self.received_js = False
        self.create_subscription(
            JointState,
            '/joint_states',   # ← adjust if yours is different
            self.joint_state_cb,
            10)

        # set up camera + Mediapipe
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera!")
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

        self.hands  = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils

        # joint1, joint3, joint2 are driven
        self.joint1 = 0.0
        self.joint3 = 0.0
        self.joint2 = 0.0

        self.joint_names = [
            'joint1','joint2','joint3',
            'joint4','joint5','joint6']

        self.create_timer(0.1, self.track_zone)
        self.get_logger().info("🖐️ Node started – waiting for joint_states…")

    def joint_state_cb(self, msg: JointState):
        if not self.received_js and len(msg.position) >= 6:
            self.current_positions = list(msg.position[:6])
            # seed joints so robot holds this pose
            self.joint1, self.joint2, self.joint3 = (
                self.current_positions[0],
                self.current_positions[1],
                self.current_positions[2]
            )
            self.received_js = True
            self.get_logger().info(
                f"✅ Initial joints: {['{:.3f}'.format(p) for p in self.current_positions]}")

    def track_zone(self):
        if not self.received_js:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No camera frame.")
            return

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self.hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark[0]
            x, y = lm.x, lm.y
            self.drawer.draw_landmarks(
                frame,
                res.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS
            )

            # HORIZONTAL control → joint1
            left, right = self.H_THRESH
            if   x <  left:
                self.joint1 = np.clip(self.joint1 - self.STEP, *self.J1_LIMIT)
            elif x > right:
                self.joint1 = np.clip(self.joint1 + self.STEP, *self.J1_LIMIT)

            # VERTICAL control → joint3 until its limits, then joint2
            top, bottom = self.V_THRESH
            if y < top:
                # up gesture
                if self.joint3 <  self.J3_LIMIT[1]:
                    self.joint3 = np.clip(self.joint3 + self.STEP, *self.J3_LIMIT)
                else:
                    self.joint2 = np.clip(self.joint2 + self.STEP, *self.J2_LIMIT)

            elif y > bottom:
                # down gesture
                if self.joint3 >  self.J3_LIMIT[0]:
                    self.joint3 = np.clip(self.joint3 - self.STEP, *self.J3_LIMIT)
                else:
                    self.joint2 = np.clip(self.joint2 - self.STEP, *self.J2_LIMIT)

        # dispatch joint1, joint2, joint3 (others hold home pose)
        self.send_joint_command()

        cv2.imshow("Hand Tracking", frame)
        cv2.waitKey(1)

    def send_joint_command(self):
        traj = JointTrajectory(joint_names=self.joint_names)
        pts  = self.current_positions.copy()

        pts[0] = self.joint1
        pts[1] = self.joint2
        pts[2] = self.joint3

        p = JointTrajectoryPoint()
        p.positions = pts
        p.time_from_start.sec = 1
        traj.points.append(p)

        self.pub.publish(traj)
        self.get_logger().info(
            f"Sent → {['{:.2f}'.format(v) for v in pts]}")

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerXarm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔌 Shutting down…")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
