# VULCAN-FaceID
This Script set is used on Vulcan ( ERC Humanoid - BITS PILANI Goa ) to:

identify individuals and store their face ids. Store contextual data and temporal identification.

---------------------------------------------------------------------

TO SET UP DEPENDENCIES for this script:
# Install dependencies
pip install opencv-python opencv-contrib-python numpy scipy

# For better accuracy, download these models:
# 1. Face Detection (place in script directory):
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# 2. Face Recognition:
wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 -O openface_nn4.small2.v1.t7

---------------------------------------------------------------------

USAGE: 
# Run with default webcam
python script.py

# Or modify to use video file
system.run(video_source='path/to/video.mp4')

---------------------------------------------------------------------

PLEASE NOTE : STRUCTURE FOR RUNNING DIRECTORY FOR OPTIMAL PERFORMACE :
-a----        03-11-2025     21:54          28104 deploy.prototxt
-a----        03-11-2025     22:03           2689 face_database.pkl                           [SHALL BE CREATED AFTER FIRST RUN / DATABASE FOR SAVED IDs]
-a----        03-11-2025     21:53       31510785 openface_nn4.small2.v1.t7
-a----        03-11-2025     21:55       10666211 res10_300x300_ssd_iter_140000.caffemodel
-a----        03-11-2025     21:19          14109 Vulcan_FaceId_Code.py
