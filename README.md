# 🤖 Deep Learning-Driven XARM VI Robotic Arm for Automated Automotive Part Handling

This project integrates **deep learning**, **RGB-D sensing**, and **robotic manipulation** using the **XARM VI** robotic arm and an **Azure Kinect DK** camera. It features real-time object detection, 3D point cloud classification with **PointNet**, and vision-based tracking using ROS 2.

## 📌 Features

- 3D scanning and dataset generation using Kinect and STL files.
- Custom dataset of `.ply` files (550 training, 50 testing).
- Trained PointNet model with 96% accuracy on object vs. non-object classification.
- Real-time object detection via ROS 2 using live Kinect point clouds.
- Object tracking using color segmentation and servoing via ROS 2 joint trajectories.
- Modular and reproducible code, fully documented and ready for extension.

---

## 🧠 Deep Learning Model

The PointNet model was trained using Google Colab and includes:

- Orthogonal regularizer for transformation invariance.
- Centered and normalized `.ply` inputs.
- Trained with categorical crossentropy.
- Output: binary classification (object / not object).

The dataset, Colab notebook, and trained `.h5` model are available in [Appendix A of the report](https://drive.google.com/drive/folders/13EHdk1zJ5d9cUkL5tZSNeCi64MtjU-G6?usp=sharing).

---

## 📷 Object Tracking

Using OpenCV and HSV color filtering, the robot visually tracks a black object with white markers by computing the centroid of the detected contour. The centroid's position in the frame is used to generate joint commands for the XARM VI to follow it in real time.

---

## 🗂 Repository Structure

```bash
xarm_final/
├── model.py                  # ROS 2 node for PointNet-based object detection
├── hand_tracking.py          # ROS 2 node for vision-based object tracking
├── models/
│   └── final_model.tflite    # Exported lightweight model (optional)
├── my_model.h5               # Full Keras model
├── *.py                      # Supporting scripts
