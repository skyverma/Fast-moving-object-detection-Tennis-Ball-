# Fast-Moving Object Detection: Tennis Ball Detection

This project focuses on detecting fast-moving objects, specifically tennis balls, in video footage using YOLO (You Only Look Once) models. The project includes training the model on a custom dataset and using trained weights to detect tennis balls in custom video inputs.

---

## Project Overview

- **Objective**: Detect tennis balls in a tennis match using YOLO.
- **Tools**: YOLOv8 and YOLOv11 models.
- **Dataset**: Custom training dataset provided in the `Data set for training` folder.
- **Output**: Bounding boxes on detected tennis balls in video footage.

---

## Contents

1. [Dataset](#dataset)
2. [Model Training](#model-training)
   - Automatic Hyperparameter Tuning
   - Manual Hyperparameter Selection
3. [Pre-trained Weights](#pre-trained-weights)
4. [Model Evaluation](#model-evaluation)
5. [Running the Detection](#running-the-detection)
6. [Comparison of YOLOv8 and YOLOv11](#comparison-of-yolov8-and-yolov11)
7. [Results](#results)

---

## Dataset

The training dataset is available in the folder named `Data set for training`. It contains:
- Images of tennis matches with tennis balls labeled.
- Corresponding annotation files for YOLO training.

---

## Model Training

### Using YOLOv8
1. **Training Configuration**:
   - Use the `All YOLO V8` notebook for training.
   - Include both:
     - **Automatic hyperparameter tuning**.
     - **Specific hyperparameter combinations** to compare performance.

2. **Training Steps**:
   - Load the dataset from the `Data set` folder.
   - Configure the training parameters in the notebook.
   - Train the YOLOv8 model on the dataset.
   - Save the trained weights.

### Using YOLOv11
Follow a similar process with YOLOv11 using the appropriate files provided.

---

## Pre-trained Weights

If you do not wish to train the model, pre-trained weights are available in the `weights` folder:
- YOLOv8 weights
- YOLOv11 weights

---

## Running the Detection

1. Open the file `All YOLO V8`.
2. Provide the required paths:
   - **Video Path**: Path to the video file where you want to detect the tennis ball.
   - **Weights Path**: Path to the trained YOLO weights (YOLOv8 or YOLOv11).
3. Run the file. 
   - The output will display the tennis ball detections on the video.

### YOLOv11 Detection
- Follow a similar process using the corresponding YOLOv11 files to run detection.

---

## Comparison of YOLOv8 and YOLOv11

- Evaluate the performance of YOLOv8 and YOLOv11 on the same dataset and video.
- Compare metrics such as:
  - Detection speed
  - Accuracy
  - Precision and recall
  - FPS (Frames Per Second)

---

## Results

- Output videos with detected tennis balls are saved in the output folder.
- Logs for training and detection performance are in the notebooks for analysis.

---

## How to Use

1. Clone the repository and navigate to the project directory.
2. Install dependencies: 
   Use Ultralytics that have all YOLO models
   then follow the above steps

