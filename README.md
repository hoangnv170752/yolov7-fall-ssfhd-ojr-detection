# YOLOv7-based Fall Detection System with SSHFD and OJR

This project implements a fall detection system using YOLOv7, combining SSHFD (Smart System for Human Fall Detection) and OJR (Occluded Joints Recovery) techniques to detect falls and recover occluded keypoints in smart home applications.

## Overview

The fall detection system operates in two main steps:

1. **SSHFD (Smart System for Human Fall Detection)**: Detects falling behavior quickly after detecting people
2. **OJR (Occluded Joints Recovery)**: Predicts and recovers joint points when they are obscured by objects or other people

The system first uses YOLOv7 to detect people in frames, then estimates keypoints using the SSHFD component. If any keypoints are occluded, the OJR component recovers them using temporal information from previous frames. Finally, the system classifies the sequence as a fall or non-fall event.

![System Architecture](docs/architecture_diagram.png)

## Project Structure

```
fall_detection_project/
├── config/
│   └── config.yaml           # Configuration parameters
├── data/
│   ├── train/                # Training data
│   ├── val/                  # Validation data
│   └── test/                 # Test data
├── models/
│   ├── yolov7/               # YOLOv7 implementation
│   ├── sshfd.py              # SSHFD model implementation
│   ├── ojr.py                # OJR model implementation
│   └── model_utils.py        # Shared utilities for models
├── utils/
│   ├── data_utils.py         # Dataset and dataloader utilities
│   ├── visual_utils.py       # Visualization tools
│   └── metrics.py            # Evaluation metrics 
├── scripts/
│   ├── prepare_data.py       # Data preparation script
│   ├── download_weights.py   # Script to download pre-trained weights
│   └── convert_dataset.py    # Script to convert datasets to project format
├── weights/
│   ├── yolov7.pt             # YOLOv7 pre-trained weights
│   ├── sshfd_best.pt         # Best SSHFD model weights
│   └── ojr_best.pt           # Best OJR model weights
├── logs/                     # Training logs
├── outputs/                  # Inference outputs and visualizations
├── docs/                     # Documentation
├── train.py                  # Main training script
├── evaluate.py               # Evaluation script
├── detect_falls.py           # Fall detection inference script
├── requirements.txt          # Project dependencies
└── README.md                 # Project information
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA (recommended for faster training and inference)

### Setup

1. Clone the repository:
   ```bash
   git clone ...
   cd fall-detection-yolov7
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv7 weights and set up the project structure:
   ```bash
   python scripts/download_weights.py
   ```

## Dataset Preparation

The system supports several fall detection datasets:

- URFD (UR Fall Detection)
- Multi-Cam Fall Dataset
- UP-Fall Detection
- Le2i Fall Detection
- Custom datasets

To prepare a dataset:

```bash
python scripts/prepare_data.py --dataset urfd --raw-data /path/to/urfd --output-dir data
```

## Training

### Train SSHFD

To train the SSHFD model:

```bash
python train.py --train-sshfd --config config/config.yaml --gpu 0
```

### Train OJR

To train the OJR model (requires a pre-trained SSHFD model):

```bash
python train.py --train-ojr --config config/config.yaml --gpu 0
```

### Train Both Models

To train both models sequentially:

```bash
python train.py --config config/config.yaml --gpu 0
```

## Evaluation

Evaluate the trained models:

```bash
python evaluate.py --config config/config.yaml --output-dir outputs/evaluation
```

To evaluate only SSHFD or OJR:

```bash
python evaluate.py --sshfd-only --config config/config.yaml
python evaluate.py --ojr-only --config config/config.yaml
```

## Inference

Run fall detection on a video:

```bash
python detect_falls.py --video path/to/video.mp4 --output outputs/output.mp4
```

Run fall detection using a webcam:

```bash
python detect_falls.py --camera 0 --output outputs/webcam.mp4
```

## Results

The system achieves state-of-the-art performance on several fall detection benchmarks:



## Extensions

The system can be extended to detect other abnormal movements:

- Violent behavior
- Fainting/unconsciousness
- Unusual gait patterns
- Repetitive behaviors
- Unusual interactions with the environment

## Acknowledgements

- YOLOv7 by WongKinYiu (https://github.com/WongKinYiu/yolov7)
- Fall detection datasets: URFD, UP-Fall, Le2i, Multi-Cam