# Emotion Recognition Using Mediapipe

## Overview
This project implements a real-time emotion recognition system using the Mediapipe library. The system detects facial landmarks, calculates blendshape scores, and analyzes them to classify emotions such as "Smile," "Sad," "Surprised," and "Sleepy." The tool provides both numerical and graphical outputs for visualizing the detected emotions.

## Features
- **Real-Time Emotion Detection**: Classifies facial expressions using predefined thresholds for blendshape scores.
- **Facial Landmark Visualization**: Displays facial landmarks and connections using Mediapipe’s face mesh.
- **Dynamic Graphs**:
  - **Real-Time Blendshape Scores**: A bar graph displaying current blendshape scores.
  - **Time-Series Graph**: A line graph showing changes in tracked blendshape scores over time.
- **FPS Display**: Tracks the performance of the real-time detection system.

## Requirements
The following libraries and tools are required:
- Python 3.7 or later
- OpenCV
- Mediapipe
- Matplotlib
- NumPy

Install dependencies using the following command:
```bash
pip install opencv-python mediapipe matplotlib numpy
```

## How It Works
### Mediapipe Model
The system leverages Mediapipe’s `FaceLandmarker` to detect 3D facial landmarks and compute blendshape scores. These scores represent the intensity of specific facial movements, normalized between 0 and 1.

### Emotion Classification
Emotions are classified based on thresholds applied to blendshape scores:
- **Smile**: High scores for `mouthSmileLeft` and `mouthSmileRight`, low `mouthOpen`.
- **Sad**: High scores for `eyeLookDownLeft`, `eyeLookDownRight`, and `mouthShrugLower`.
- **Surprised**: High scores for `browOuterUpLeft`, `browOuterUpRight`, `mouthPucker`, or `jawOpen`.
- **Sleepy**: High scores for `eyeLookDownLeft`, `eyeLookDownRight`, and `eyeBlink`.

### Visualization
- **Landmarks**: Facial landmarks are drawn on the live video feed.
- **Graphs**:
  - A bar chart updates blendshape scores in real-time.
  - A time-series graph tracks selected blendshape scores over time.

## Usage
Run the script with the following command:
```bash
python graphical.py --model face_landmarker.task --numFaces 1 --camera 0 --width 640 --height 480
```
### Arguments:
- `--model`: Path to the Mediapipe model file.
- `--numFaces`: Number of faces to detect.
- `--camera`: Camera ID (default is 0).
- `--width` and `--height`: Resolution of the video feed.

### Example:
```bash
python graphical.py --model face_landmarker.task --numFaces 2 --camera 1 --width 1280 --height 720
```

## Key Functions
1. **Facial Landmark Detection**: Detects and normalizes 3D points to identify facial movements.
2. **Emotion Counters**: Maintains a real-time count of detected emotions.
3. **Graphs**: Utilizes Matplotlib’s `FuncAnimation` to dynamically update graphs.

## Performance Metrics
- **FPS (Frames Per Second)**: Displayed in the top-left corner of the video feed for monitoring real-time performance.
- **Blendshape Scores**: Visualized in real-time for detailed analysis.

## FAQs
1. **Why are results unaffected by head movements?**
   - The model analyzes 3D landmarks and normalizes them to a 2D projection, ensuring consistency regardless of head position.

2. **What is the significance of the 0.5 threshold?**
   - This value represents a balanced midpoint, allowing reliable classification between presence and absence of specific expressions.

3. **How are graphs updated in real-time?**
   - Matplotlib’s `FuncAnimation` dynamically updates the bar and line graphs with new data every 100ms.

4. **What are blendshape scores?**
   - Blendshape scores quantify facial movements numerically, aiding in emotion classification.

## Acknowledgments
This project utilizes Mediapipe’s robust facial recognition and landmark detection models for precise emotion analysis.

