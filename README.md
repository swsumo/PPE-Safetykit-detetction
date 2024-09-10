# PPE-Safetykit-detetction


## Overview

This project is designed to detect Personal Protective Equipment (PPE) and safety kits using a YOLOv8 model. The model is trained and deployed using the Ultralytics YOLOv8 framework and integrates with an HTML-based frontend for displaying detection results.

## Features

- **PPE Detection**: Detects various types of personal protective equipment, such as helmets, vests, gloves, and more.
- **Real-Time Detection**: Provides real-time detection of PPE items using the YOLOv8 model.
- **Easy Deployment**: Utilizes the Ultralytics YOLOv8 framework for easy model deployment and management.
- **Web-based Interface**: A user-friendly HTML frontend to visualize and interact with the detection results.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7+
- Ultralytics YOLOv8 (`ultralytics`)
- OpenCV
- Flask (for serving the frontend, if necessary)

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/ppe-safetykit-detection.git
    cd ppe-safetykit-detection
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download YOLOv8 Weights**:
    - Download the pre-trained YOLOv8 weights from the [Ultralytics YOLOv8 Model Zoo](https://github.com/ultralytics/ultralytics#model-zoo).
    - Place the weights file in the `models/` directory.

### Running the Project

1. **Model Inference**:
    - Run the detection script to perform inference on images or videos:
    ```bash
    python detect.py --weights models/yolov8.pth --source data/test_images
    ```

2. **Web Frontend**:
    - Serve the HTML frontend to visualize the results:
    ```bash
    flask run
    ```
    - Access the web interface at `http://127.0.0.1:5000/`.

## Project Structure

```plaintext
ppe-safetykit-detection/
│
├── InputDemoVideo/           # Directory for input demo videos
│   └── demo4.mp4             # Sample input video
│
├── OutputDemoVideos/         # Directory for output videos after detection
│
├── predicted/                # Directory for storing predicted frames
│
├── best.pt                   # Pre-trained YOLOv8 weights (best version)
├── yolov8m.pt                # Another set of YOLOv8 weights
│
├── cli.log                   # Log file for command line operations
├── detect.py                 # Script to run YOLOv8 detection
├── ppe_detection_kit.ipynb   # Jupyter notebook for further analysis and testing
└── README.md                 # Project README file

```

## API References

### Ultralytics YOLOv8

- **Website**: [Ultralytics](https://ultralytics.com)
- **Documentation**: [YOLOv8 Documentation](https://docs.ultralytics.com)



## Acknowledgments

- **Ultralytics** for the YOLOv8 framework.
- **OpenCV** for image processing.
- **Flask** for the web framework.



https://github.com/user-attachments/assets/2c8a4361-ac55-4bbc-b00d-4b188bab8a0c




