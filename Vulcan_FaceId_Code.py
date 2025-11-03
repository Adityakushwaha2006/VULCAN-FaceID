"""
Real-Time Face Detection and Identification System
Detects faces in video stream, assigns unique IDs, and tracks them across frames
"""

import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path

class FaceIdentificationSystem:
    def __init__(self, 
                 db_path='face_database.pkl',
                 similarity_threshold=0.6,
                 use_cosine=True):
        """
        Initialize the Face Identification System
        
        Args:
            db_path: Path to save/load face database
            similarity_threshold: Maximum distance for matching faces
            use_cosine: Use cosine distance (True) or Euclidean (False)
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.use_cosine = use_cosine
        self.face_db = {}
        self.next_id = 1
        
        # Load existing database if available
        self.load_database()
        
        # Initialize face detector (OpenCV DNN)
        self.detector = self._load_face_detector()
        
        # Initialize face recognizer (OpenCV DNN - FaceNet-like model)
        self.recognizer = self._load_face_recognizer()
        
        print(f"System initialized with {len(self.face_db)} known faces")
    
    def _load_face_detector(self):
        """Load OpenCV DNN face detector"""
        try:
            # Try to load pre-trained models
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(proto_path) and os.path.exists(model_path):
                detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                print("Loaded Caffe face detector")
            else:
                # Fallback to Haar Cascade
                detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("Using Haar Cascade detector (download DNN models for better accuracy)")
            return detector
        except Exception as e:
            print(f"Error loading detector: {e}")
            return cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_face_recognizer(self):
        """Load face recognition model"""
        try:
            # OpenCV DNN face recognition model
            model_path = "openface_nn4.small2.v1.t7"
            if os.path.exists(model_path):
                recognizer = cv2.dnn.readNetFromTorch(model_path)
                print("Loaded OpenFace recognition model")
                return recognizer
            else:
                print("Recognition model not found. Using simple feature extraction.")
                print("Download openface_nn4.small2.v1.t7 for better accuracy")
                return None
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            return None
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        if isinstance(self.detector, cv2.CascadeClassifier):
            # Haar Cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [(x, y, w, h) for (x, y, w, h) in faces]
        else:
            # DNN detection
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
                (104.0, 177.0, 123.0)
            )
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
    
    def get_embedding(self, face_img):
        """
        Extract face embedding from face image
        
        Args:
            face_img: Cropped face image
            
        Returns:
            NumPy array representing face embedding
        """
        if self.recognizer is not None:
            # Use OpenFace model
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0/255, (96, 96), (0, 0, 0), 
                swapRB=True, crop=False
            )
            self.recognizer.setInput(blob)
            embedding = self.recognizer.forward()
            return embedding.flatten()
        else:
            # Fallback: use simple feature extraction
            face_resized = cv2.resize(face_img, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            # Simple histogram-based feature
            embedding = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            return embedding.flatten() / np.linalg.norm(embedding.flatten())
    
    def compute_distance(self, emb1, emb2):
        """
        Compute distance between two embeddings
        
        Args:
            emb1, emb2: Face embeddings
            
        Returns:
            Distance value
        """
        if self.use_cosine:
            return cosine(emb1, emb2)
        else:
            return euclidean(emb1, emb2)
    
    def match_or_create_id(self, embedding):
        """
        Match embedding against database or create new ID
        
        Args:
            embedding: Face embedding to match
            
        Returns:
            Tuple of (person_id, is_new)
        """
        if not self.face_db:
            # First face in database
            person_id = f"P{self.next_id:03d}"
            self.next_id += 1
            return person_id, True
        
        # Compare with all stored embeddings
        best_match_id = None
        best_distance = float('inf')
        
        for person_id, data in self.face_db.items():
            distance = self.compute_distance(embedding, data['embedding'])
            if distance < best_distance:
                best_distance = distance
                best_match_id = person_id
        
        # Check if best match is within threshold
        if best_distance < self.similarity_threshold:
            return best_match_id, False
        else:
            # Create new ID
            person_id = f"P{self.next_id:03d}"
            self.next_id += 1
            return person_id, True
    
    def update_database(self, person_id, embedding, is_new=False):
        """
        Update face database with new or updated information
        
        Args:
            person_id: Person identifier
            embedding: Face embedding
            is_new: Whether this is a new person
        """
        current_time = datetime.now().isoformat()
        
        if is_new:
            self.face_db[person_id] = {
                'embedding': embedding,
                'first_seen': current_time,
                'last_seen': current_time,
                'appearances': 1
            }
        else:
            # Update existing entry
            self.face_db[person_id]['last_seen'] = current_time
            self.face_db[person_id]['appearances'] += 1
            # Update embedding with moving average for robustness
            old_emb = self.face_db[person_id]['embedding']
            self.face_db[person_id]['embedding'] = 0.9 * old_emb + 0.1 * embedding
    
    def save_database(self):
        """Save face database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'face_db': self.face_db,
                    'next_id': self.next_id
                }, f)
            print(f"Database saved to {self.db_path}")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self):
        """Load face database from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_db = data['face_db']
                    self.next_id = data['next_id']
                print(f"Database loaded from {self.db_path}")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.face_db = {}
                self.next_id = 1
    
    def process_frame(self, frame):
        """
        Process a single frame: detect faces, identify, and annotate
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with bounding boxes and IDs
        """
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Ensure coordinates are within frame bounds
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
            
            # Get embedding
            embedding = self.get_embedding(face_img)
            
            # Match or create ID
            person_id, is_new = self.match_or_create_id(embedding)
            
            # Update database
            self.update_database(person_id, embedding, is_new)
            
            # Draw bounding box and label
            color = (0, 255, 0) if not is_new else (0, 165, 255)  # Green for known, Orange for new
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label
            label = f"{person_id}"
            if is_new:
                label += " (NEW)"
            appearances = self.face_db[person_id]['appearances']
            label += f" [{appearances}]"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        # Display statistics
        stats = f"Known Faces: {len(self.face_db)} | Current Frame: {len(faces)} faces"
        cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self, video_source=0):
        """
        Run the face identification system
        
        Args:
            video_source: Camera index (0 for default) or video file path
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        print("Starting face identification system...")
        print("Press 'q' to quit, 's' to save database, 'r' to reset database")
        
        frame_count = 0
        save_interval = 50  # Auto-save every 50 frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Face Identification System', annotated_frame)
            
            # Auto-save periodically
            frame_count += 1
            if frame_count % save_interval == 0:
                self.save_database()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_database()
                print("Database saved manually")
            elif key == ord('r'):
                response = input("Reset database? (yes/no): ")
                if response.lower() == 'yes':
                    self.face_db = {}
                    self.next_id = 1
                    print("Database reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.save_database()
        print("System shutdown complete")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Real-Time Face Detection and Identification System")
    print("=" * 60)
    print("\nNote: For best results, download these models:")
    print("1. Face Detection:")
    print("   - deploy.prototxt")
    print("   - res10_300x300_ssd_iter_140000.caffemodel")
    print("2. Face Recognition:")
    print("   - openface_nn4.small2.v1.t7")
    print("\nWithout these models, the system will use Haar Cascade")
    print("(less accurate but functional)\n")
    
    # Initialize system
    system = FaceIdentificationSystem(
        db_path='face_database.pkl',
        similarity_threshold=0.6,  # Adjust based on your needs
        use_cosine=True
    )
    
    # Run with webcam (change to video file path if needed)
    system.run(video_source=0)


if __name__ == "__main__":
    main()