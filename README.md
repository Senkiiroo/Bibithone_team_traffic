# Traffic Safety & Intersection Event Detection — WIUT Hackathon 2026

This repository contains an automated computer vision pipeline designed for vehicle and pedestrian detection, multi-object tracking, and intersection event and violation analysis from video streams.

---

## Installation & Usage

### Setup & Dependencies

Ensure the environment runs Python 3.10 or higher and has CUDA support for optimal inference performance. Install the required Python packages:

```bash
pip install -r requirements.txt
Before running evaluation, download the pre-trained model weights by executing the initialization script (requires active internet connection, run once):

Bash
bash weights/download.sh
Offline Inference
Process video files from a target input directory and generate structured output predictions:

Bash
python run_submission.py --videos /data/test --out predictions.json
Format Validation
Validate the output schema and structural integrity of the generated predictions prior to submission:

Bash
python evaluate.py --pred predictions.json --validate-only
Technical Approach
Architecture & Models
The object detection pipeline relies on YOLOv8 models (yolov8s.pt / yolov8n.pt) implemented via the Ultralytics framework to identify target classes including vehicles (cars, buses, trucks) and pedestrians. Object tracking and persistent identity assignment across video frames are handled using integrated ByteTrack / BoT-SORT algorithms.

Traffic flow dynamics and density estimation are processed using Farneback dense optical flow (cv2.calcOpticalFlowFarneback) to calculate velocity vectors and identify motion patterns within designated intersection zones. Background subtraction (MOG2) combined with morphological filtering serves as an auxiliary mechanism for detecting stationary vehicles and queue accumulation.

Model Training & Datasets
Object detection models utilize base weights pre-trained on the COCO Dataset (released under the Creative Commons Attribution 4.0 International License). Fine-tuning and validation were conducted on publicly available intersection surveillance data annotated for traffic monitoring tasks.

Learned vs. Rule-Based Components
The system separates detection and rule logic into two stages:

Learned Components: Neural network inference for object bounding box detection, category classification, feature embedding, and cross-frame association tracking.

Rule-Based Components: Geometric trajectory evaluation against predefined intersection boundaries, optical flow vector thresholding for anomaly detection, and deterministic spatial-temporal collision/violation criteria.

Determinism & Reproducibility
To guarantee deterministic execution and repeatable evaluation outputs across environments, pseudo-random number generators across Python, NumPy, and PyTorch are initialized with fixed seeds:

Python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
Team Contributions
Detection & Tracking Specialist: Configured YOLOv8 detection pipelines, tuned ByteTrack hyperparameters, and implemented frame-level candidate extraction.

Computer Vision & Flow Engineer: Developed Farneback optical flow calculation modules, motion vector analytics, and rule-based event trigger logic.

Pipeline & Infrastructure Lead: Engineered the run_submission.py execution workflow, automated weight management scripts, and implemented output format validation.