# PPE Detection using YOLOv8

A real-time Personal Protective Equipment (PPE) detection system built with YOLOv8m, capable of detecting safety gear on workers from video feeds, images, or webcam.

## Detected Classes

| Class | mAP50 |
|---|---|
| Protective Boots | 0.990 |
| Protective Helmet | 0.975 |
| Dust Mask | 0.963 |
| Glove | 0.937 |
| Jacket | 0.935 |
| Eye Wear | 0.849 |
| Shield | 0.639 |
| **Overall** | **0.898** |

## Project Structure

```
kit_detect/
├── best.pt                    # Trained YOLOv8m weights
├── yolov8m.pt                 # Base YOLOv8m pretrained weights
├── detect.py                  # Inference script
├── ppe_detection_kit.ipynb    # Training notebook (Google Colab)
├── requirements.txt
├── .env                       # API keys (not committed)
├── .env.example               # Template for .env
├── .gitignore
├── InputDemoVideo/            # Sample input videos
├── OutputDemoVideos/          # Sample output videos with detections
└── predicted/                 # Validation metrics and sample predictions
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd kit_detect
pip install -r requirements.txt
```

Install PyTorch for your hardware (choose one):

```bash
# CPU only
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your Roboflow API key (needed for training only)
```

### 3. Download model weights

The trained model `best.pt` (~50MB) is required for inference. Place it in the project root.

## Usage

### Run on a video file

```bash
python detect.py --source InputDemoVideo/demo.mp4 --conf 0.3 --save
```

### Run on webcam

```bash
python detect.py --source 0
```

### Run on an image

```bash
python detect.py --source path/to/image.jpg --save
```

### All options

```
--source    Source: video file, image, directory, or 0 for webcam (default: 0)
--model     Path to model weights (default: best.pt)
--conf      Confidence threshold 0.0-1.0 (default: 0.3)
--save      Save output to disk
--no-show   Disable real-time display window
```

## Training

Open `ppe_detection_kit.ipynb` in Google Colab. The notebook covers:

1. Environment setup (GPU check, dependency install)
2. Dataset download from Roboflow (requires `ROBOFLOW_API_KEY` in `.env`)
3. YOLOv8m fine-tuning for 90 epochs
4. Validation and metric visualization
5. Inference on test images and videos
6. ONNX export

## Model Export (ONNX)

```bash
yolo export model=best.pt format=onnx
```

## Dataset

- Source: [Roboflow](https://roboflow.com) — `objet-detect-yolov5 / eep_detection-u9bbd v1`
- Format: YOLOv5 PyTorch
- ~6,482 images across train/valid/test splits

## Results

Training curves and validation predictions are saved in the `predicted/` folder.
