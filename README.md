# VULCAN-FaceID

## Overview
A facial recognition system for Vulcan (ERC Humanoid - BITS PILANI Goa) that:
- Identifies individuals and stores their face IDs
- Maintains contextual data and temporal identification records
- Provides real-time face detection and tracking

## Dependencies

Install the required Python packages:
```bash
pip install opencv-python opencv-contrib-python numpy scipy
```

### Required Models
For optimal performance, download these models to your script directory:

1. **Face Detection Models**:
```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

2. **Face Recognition Model**:
```bash
wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 -O openface_nn4.small2.v1.t7
```

## Usage

### Basic Usage
Run with default webcam:
```bash
python Vulcan_FaceId_Code_Iter1.py
```

### Advanced Usage
To use a video file instead of webcam:
```python
system.run(video_source='path/to/video.mp4')
```

## Directory Structure
For optimal performance, ensure your directory contains these files:

| File | Purpose |
|------|---------|
| `deploy.prototxt` | Face detection model configuration |
| `face_database.pkl` | Database file (auto-created after first run) |
| `openface_nn4.small2.v1.t7` | Face recognition model |
| `res10_300x300_ssd_iter_140000.caffemodel` | Face detection model weights |
| `Vulcan_FaceId_Code_Iter1.py` | Main script |

## Future Implementations
- ERC member training
- Model accuracy (false positive reduction)
- Change database to save {name field} along with ID
- Change the name while model is running by taing person ID as input.
- Remove stanger data from pkl before next model run, where-ever name is specified keep those
