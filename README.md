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
- Use the link to access training data 
https://drive.google.com/drive/folders/16qoSbac4ezep6fEctXtZOhk5GbCDc_e4
---

## Model Training

### Using YOLOv8
1. **Training Configuration**:
   - Use the `All YOLO V8` notebook for training.
   - Include both:
     - **Automatic hyperparameter tuning**.
     - **Specific hyperparameter combinations** to compare performance.
       
      **Training Configuration**:
      ○ Model: YOLO v8 & 11 (manual & automatic tuning)
      ○ Parameters: epochs=100, imgsz=640, batch=8, lr=1e-3,
      optimizer='Adam', augment=True, hsv_h=0.015, hsv_s=0.7,
      hsv_v=0.4, fliplr=0.5, mosaic=1.0, cache=True
2. **Training Steps**:
   - Load the dataset from the `Data set` folder.
   - Configure the training parameters in the notebook(Trainig using the https://github.com/skyverma/Fast-moving-object-detection-Tennis-Ball-/blob/main/Training_Experiments_Weights/experiment-training-yolov8-all-combination.ipynb)
   - Train the YOLOv8 model on the dataset.
   - Save the trained weights.

### Using YOLOv11
Follow a similar process with YOLOv11 using the appropriate files provided.

---

## Pre-trained Weights

If you do not wish to train the model, pre-trained weights are available in the `Trainded weights` folder:(https://drive.google.com/drive/folders/16qoSbac4ezep6fEctXtZOhk5GbCDc_e4)
- YOLOv8 weights
- YOLOv11 weights

---

## Running the Detection

1. Open the file `All YOLO V8`.(https://github.com/skyverma/Fast-moving-object-detection-Tennis-Ball-/blob/main/Final_files_to_get_all_results/all-yolo-v8-results.ipynb)
2. Provide the required paths:
   - **Video Path**: Path to the video file where you want to detect the tennis ball.
   - **Weights Path**: Path to the trained YOLO weights (YOLOv8 or YOLOv11).
   - Use pretrained weights visit folder Trained weight (https://drive.google.com/drive/folders/16qoSbac4ezep6fEctXtZOhk5GbCDc_e4)
3. Run the file. 
   - The output will display the tennis ball detections on the video.

### YOLOv11 Detection
- Follow a similar process using the corresponding YOLOv11 files to run detection.

---

## Comparison of YOLOv8 and YOLOv11

- Evaluate the performance of YOLOv8 and YOLOv11 on the same dataset and video.
- Compare metrics such as:
  - Detection Ratio
  - No of total frames detect the ball
  - Compare by Visualisation
 

---

## Results

- Output videos with detected tennis balls are saved in the output folder.
- Go to this link and check the folder Result output videos folder to check the results(https://drive.google.com/drive/folders/16qoSbac4ezep6fEctXtZOhk5GbCDc_e4)
- Logs for training and detection performance are in the notebooks for analysis.

---

## How to Use

1. Clone the repository and navigate to the project directory.
2. Install dependencies: 
   Use Ultralytics that have all YOLO models
   then follow the above steps
   The input directory looks like..........
![image](https://github.com/user-attachments/assets/b728516e-5624-4b4d-9fb4-962d72e2583f)

## To get the final results:
We made lot of experiments to get  the files go to this link and get all experments over training and testing  (https://drive.google.com/drive/folders/16qoSbac4ezep6fEctXtZOhk5GbCDc_e4) 
