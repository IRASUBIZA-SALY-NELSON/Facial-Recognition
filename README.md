# Facial Recognition System

A real-time facial recognition system built with Python, OpenCV, MediaPipe, and LBPH (Local Binary Patterns Histograms) face recognition algorithm.

## Features

- **Face Capture**: Automatically captures and saves face images using MediaPipe Face Mesh for accurate face detection
- **Model Training**: Trains an LBPH face recognition model on captured datasets
- **Real-time Recognition**: Recognizes faces in real-time with confidence scores
- **Multi-camera Support**: Automatically detects and uses available cameras
- **Robust Detection**: Uses MediaPipe Face Mesh for reliable face detection even in varying lighting conditions

## Project Structure

```
facial_Recog/
├── capture.py          # Capture face images for training
├── train.py            # Train the LBPH recognition model
├── predict.py          # Real-time face recognition
├── dataset/            # Stores captured face images (organized by person)
│   ├── person1/
│   └── person2/
└── models/             # Stores trained models
    ├── lbph_model.xml  # Trained LBPH model
    └── label_map.json  # Maps label IDs to person names
```

## Requirements

- Python 3.7+
- OpenCV (with contrib modules for face recognition)
- MediaPipe
- NumPy

## Installation

1. **Clone or download this repository**

2. **Install required packages:**

```bash
pip install opencv-contrib-python mediapipe numpy
```

> **Note**: Make sure to install `opencv-contrib-python` (not just `opencv-python`) as it includes the face recognition modules.

## Usage

### 1. Capture Face Images

Run the capture script to collect training images for a person:

```bash
python capture.py
```

- Enter the person's name when prompted
- The script will automatically detect your face using the webcam
- It captures up to 100 images and saves them in `dataset/<name>/`
- Press `q` to quit early

**Tips:**
- Move your head slightly to capture different angles
- Ensure good lighting conditions
- Look directly at the camera

### 2. Train the Model

After capturing images for one or more people, train the recognition model:

```bash
python train.py
```

This will:
- Load all images from the `dataset/` folder
- Train an LBPH face recognition model
- Save the model to `models/lbph_model.xml`
- Save the label mapping to `models/label_map.json`

### 3. Run Face Recognition

Start real-time face recognition:

```bash
python predict.py
```

- The system will detect faces and display the recognized person's name
- Confidence scores are shown (lower is better)
- Press `q` to quit

## How It Works

### Face Detection
The system uses **MediaPipe Face Mesh** for face detection, which provides:
- 468 facial landmarks for precise face localization
- Robust performance in various lighting conditions
- Real-time processing capabilities

### Face Recognition
The system uses **LBPH (Local Binary Patterns Histograms)** algorithm:
- Analyzes local texture patterns in face images
- Robust to illumination changes
- Efficient and fast for real-time recognition
- Confidence scores indicate match quality (lower = better match)

### Camera Detection
All scripts automatically:
- Try multiple camera indices (0, 1, 2)
- Use DirectShow backend on Windows for better compatibility
- Validate that cameras return valid, non-black frames
- Provide helpful error messages if no camera is found

## Troubleshooting

### Camera Not Found
If you get a "No working camera found" error:
1. Check that your webcam is connected
2. Close other applications using the camera (Zoom, Teams, etc.)
3. Grant camera permissions to Python/Terminal
4. Try running the script as administrator

### Poor Recognition Accuracy
If recognition is inaccurate:
1. Capture more training images (the script captures up to 100)
2. Ensure consistent lighting between training and recognition
3. Capture images from multiple angles during training
4. Retrain the model after adding more images

### Import Errors
If you get import errors:
- Make sure you installed `opencv-contrib-python`, not just `opencv-python`
- Verify all packages are installed: `pip list | grep -E "opencv|mediapipe|numpy"`

## Technical Details

### LBPH Parameters
The LBPH recognizer uses default parameters which work well for most cases:
- **Radius**: 1
- **Neighbors**: 8
- **Grid X**: 8
- **Grid Y**: 8

### MediaPipe Face Mesh Settings
- **Max faces**: 1 (processes one face at a time)
- **Detection confidence**: 0.5
- **Tracking confidence**: 0.5
- **Refine landmarks**: True (for better accuracy)

## Future Enhancements

Potential improvements for this system:
- [ ] Add support for multiple face recognition simultaneously
- [ ] Implement face verification (1:1 matching)
- [ ] Add a GUI interface
- [ ] Support for video file input
- [ ] Database integration for larger datasets
- [ ] Deep learning models (FaceNet, ArcFace) for better accuracy
- [ ] Anti-spoofing measures (liveness detection)

## License

This project is open source and available for educational purposes.

## Acknowledgments

- **OpenCV**: Computer vision library
- **MediaPipe**: Face detection and landmark estimation
- **LBPH**: Face recognition algorithm
