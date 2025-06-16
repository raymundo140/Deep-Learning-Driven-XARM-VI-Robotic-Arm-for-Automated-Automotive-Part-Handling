#!/usr/bin/env python3
import logging
import cv2
import numpy as np
from ultralytics import YOLO

# ——————————————————————————————————————————————
# 1) Suppress logspam from Ultralytics / root
# ——————————————————————————————————————————————
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.FATAL)

# ——————————————————————————————————————————————
# 2) Load your trained YOLOv8 model
# ——————————————————————————————————————————————
model = YOLO('/home/goma/ros2_ws/src/xarm_final/xarm_final/model-xarm.pt')
CONF_THRESH = 0.3

# ——————————————————————————————————————————————
# 3) OpenCV capture
# ——————————————————————————————————————————————
cap = cv2.VideoCapture(0)                 # index 0 → default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit(1)

# ——————————————————————————————————————————————
# 4) Main loop: grab frame, run YOLO, draw, show
# ——————————————————————————————————————————————
cv2.namedWindow("YOLO Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Webcam", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inference
    results = model(frame, conf=CONF_THRESH)

    # draw
    annotated = frame.copy()
    for r in results:
        for box in (r.boxes or []):
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            label = model.names[cls].upper()
            color = (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

    # show
    cv2.imshow("YOLO Webcam", annotated)

    # exit on ESC or 'q'
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

# ——————————————————————————————————————————————
# 5) Cleanup
# ——————————————————————————————————————————————
cap.release()
cv2.destroyAllWindows()
