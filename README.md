# Age and Gender Detection Using OpenCV

This project demonstrates a real-time application for detecting faces and predicting both age and gender using deep learning models. The system is built with OpenCV and pre-trained deep learning models for face detection, age estimation, and gender classification.

## Features:
- **Face Detection**: Detects faces in video frames using a pre-trained SSD face detection model.
- **Age Prediction**: Estimates the age group of detected faces using a pre-trained deep learning model.
- **Gender Prediction**: Classifies the gender of detected faces as male or female.
- **Real-Time Video Processing**: Processes each frame from a video file and overlays the predicted age and gender on the detected face.

## Requirements:
- OpenCV
- Pre-trained Caffe models for face detection, age prediction, and gender classification
- Numpy

## Pre-Trained Models:
- **Face Detection Model**: SSD model trained on the WIDER FACE dataset.
- **Age Prediction Model**: Model trained on the Adience dataset.
- **Gender Prediction Model**: Model trained on the Adience dataset.

## How It Works:
1. The input video is processed frame by frame.
2. Faces are detected using a pre-trained SSD model.
3. For each detected face, age and gender predictions are made using separate pre-trained models.
4. The predictions (age group and gender) are displayed as labels on the video output.

## Usage:
- Clone the repository and ensure the required pre-trained models are in the same directory as the script.
- Run the Python script with your input video file:

```bash
python your_script_name.py
