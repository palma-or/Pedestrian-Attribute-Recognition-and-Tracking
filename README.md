# 👁️ Advanced Vision System for Dynamic Pedestrian Attribute Recognition and Tracking

An advanced computer vision pipeline designed for real-time pedestrian detection, tracking, and attribute recognition (PAR). The system tracks individuals across predefined virtual lines for crowd analytics and uses a multi-task deep learning network to extract specific pedestrian attributes such as gender, bag presence, and hat presence.

---

## 📖 Project Overview

In environments such as shopping malls or public spaces, analyzing customer behavior and pedestrian flow is crucial for security and space optimization. This system leverages state-of-the-art computer vision models to perform a complete analysis pipeline:
1. Identifying people in a video scene.
2. Tracking their movements over time and across specific virtual boundaries.
3. Dynamically recognizing visual attributes (gender, accessories) for each tracked individual.

---

## 🏗️ System Architecture

The pipeline consists of three core AI modules, optimized for real-time performance and high accuracy.

### 1. Object Detection: YOLOv8n
We use the Nano version of **YOLOv8** to achieve an optimal balance between accuracy and computational efficiency, making it suitable for real-time processing.
* **Backbone:** CSPDarknet53 (reduces gradient redundancy and improves feature extraction).
* **Neck:** Spatial Pyramid Pooling - Fast (SPPF) to aggregate multi-scale features.
* **Head:** Decoupled detection head, separating classification and bounding box regression.
* **Optimized Inference Parameters:** 
  * `confidence` increased to 0.4 (reducing false positives).
  * `IoU` set to 0.6 for aggressive Non-Maximum Suppression (NMS).
  * `classes=0` (strictly limited to the "person" class).
  * `half=False` to maintain detection quality over slight speed gains.

### 2. Pedestrian Tracking: HybridSORT
To ensure continuous tracking and handle occlusions, the system implements **HybridSORT**, integrating classic tracking algorithms (Kalman Filter, Hungarian Algorithm) with deep learning-based visual descriptors.
* **Re-Identification Model:** We integrated the **OSNet** model because it is specifically trained and optimized for human re-identification, providing the best trade-off between real-time inference and accuracy.
* **Key Parameter Tuning:**
  * `max_age=50`: Allows the system to hold tracks through temporary occlusions.
  * `det_thresh=0.3`: Ensures only high-confidence detections enter the tracking pipeline.
  * `iou_threshold=0.15`: Makes spatial association more rigorous to prevent identity switches.
  * `longterm_reid_weight=1.5`: Gives strong priority to the OSNet visual features for robust re-identification after occlusions.

### 3. Pedestrian Attribute Recognition (PAR): Multi-Task Network
Instead of running separate networks for each attribute, we implemented a custom Multi-Task Network.
* **VILT + VQA Dataset Enhancement:** The training dataset (over 100,000 images) initially suffered from missing labels. We utilized a Vision-and-Language Transformer (VILT) paired with Visual Question Answering (VQA) to dynamically reconstruct missing data (e.g., asking the AI "Is the person male or female?").
* **Architecture:** 
  * **Backbone:** A modified **ResNet34** (pretrained on ImageNet, final FC layer removed) serving as a universal feature extractor that outputs a 512-dimensional vector.
  * **Classification Heads:** Three independent, fully connected layers branching from the backbone to predict **Gender** (Male/Female), **Bag** (Yes/No), and **Hat** (Yes/No).
* **Preprocessing:** Bounding boxes are cropped, resized, and processed with contrast and saturation adjustments to highlight clothing and accessories before being fed to ResNet34.

---

## 📐 Camera Calibration & Line Projection

To monitor pedestrian flow across specific areas, the system projects 3D virtual lines onto the 2D video feed using camera calibration parameters.
* A `configuration.txt` file supplies internal parameters (focal length, sensor resolution) and external characteristics (x, y, z translation, plus pitch, roll, yaw rotations).
* OpenCV's `projectPoints` and `Rodrigues` functions are utilized with the custom Camera Matrix ($K$) and Rotation Matrices ($R_x, R_y, R_z$) to map real-world boundaries onto the video pixels accurately.
* **Line Crossing Logic:** To prevent false positives from mere bounding box overlaps, the algorithm tracks a specific coordinate point slightly above the base of the bounding box to confirm a definitive line crossing.

---

## ⚙️ Integration and Pipeline Flow

1. **Frame Capture:** Read frame from the video source.
2. **Detection:** YOLOv8n extracts bounding boxes of people.
3. **Tracking:** HybridSORT (with OSNet) assigns a unique ID to each bounding box and maintains the trajectory.
4. **Line Crossing Evaluation:** The system updates trajectory records if the tracked entity crosses the projected 3D boundaries.
5. **Attribute Extraction:** The frame is cropped around the bounding box and passed to the PAR Multi-Task ResNet34 to update gender, bag, and hat predictions.
6. **Result Logging:** The system keeps track of the maximum confidence values for attributes over the tracking lifespan to ignore temporary fluctuations.

---

## 🚀 How to Run

Ensure you have your configuration file and input video ready in the project directory.

Run the main pipeline using the following command:

```bash
python main.py --video video.mp4 --configuration configuration.txt --results results.txt
--video: Path to the input video file.

--configuration: Path to the text file containing the camera matrix, rotations, and line coordinates.

--results: Destination path where the final tracking and attribute recognition logs will be saved.

```
**Authors (Group 10):** Giovanni Lamb, Egidio Zottarelli, Fabrizio Saturnino, Palma Orlando  
**Institution:** University of Salerno, Department of Information Engineering, Electrical Engineering and Applied Mathematics  