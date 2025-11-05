"""
Enhanced Real-Time Face Detection and Identification System for Vulcan ERC Humanoid

Improvements:
- Name field management for ERC members
- Training mode for reducing false positives
- Dynamic name assignment during runtime
- Database cleanup for unnamed strangers
- Improved accuracy with confidence scores

Author: Aaditya Kushawaha, Anirudh Singh Air
Project: Vulcan ERC
"""

import cv2
import numpy as np
import pickle
import os
import json
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path
from collections import deque

class FaceIdentificationSystem:
    def __init__(self, 
                 db_path='face_database.pkl',
                 similarity_threshold=0.6,
                 use_cosine=True,
                 training_mode=False,
                 min_confidence=0.7):
        """
        Initialize the Enhanced Face Identification System
        
        Args:
            db_path: Path to save/load face database
            similarity_threshold: Maximum distance for matching faces
            use_cosine: Use cosine distance (True) or Euclidean (False)
            training_mode: Enable training mode for ERC members
            min_confidence: Minimum confidence for face detection
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.use_cosine = use_cosine
        self.training_mode = training_mode
        self.min_confidence = min_confidence
        self.face_db = {}
        self.next_id = 1
        
        # Track recent detections for stability
        self.detection_history = {}  # person_id -> deque of recent embeddings
        self.history_size = 5
        
        # Load existing database if available
        self.load_database()
        
        # Initialize face detector and recognizer
        self.detector = self._load_face_detector()
        self.recognizer = self._load_face_recognizer()
        
        print(f"System initialized with {len(self.face_db)} known faces")
        print(f"Training mode: {'ON' if training_mode else 'OFF'}")
    
    def _load_face_detector(self):
        """Load OpenCV DNN face detector"""
        try:
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(proto_path) and os.path.exists(model_path):
                detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                print("Loaded Caffe face detector")
            else:
                detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("Using Haar Cascade detector")
            return detector
        except Exception as e:
            print(f"Error loading detector: {e}")
            return cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_face_recognizer(self):
        """Load face recognition model"""
        try:
            model_path = "openface_nn4.small2.v1.t7"
            if os.path.exists(model_path):
                recognizer = cv2.dnn.readNetFromTorch(model_path)
                print("Loaded OpenFace recognition model")
                return recognizer
            else:
                print("Recognition model not found. Using simple feature extraction.")
                return None
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            return None
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame with confidence scores
        
        Returns:
            List of tuples [(x, y, w, h, confidence), ...]
        """
        if isinstance(self.detector, cv2.CascadeClassifier):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [(x, y, w, h, 1.0) for (x, y, w, h) in faces]
        else:
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
                if confidence > self.min_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
            return faces
    
    def get_embedding(self, face_img):
        """Extract face embedding from face image"""
        if self.recognizer is not None:
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0/255, (96, 96), (0, 0, 0), 
                swapRB=True, crop=False
            )
            self.recognizer.setInput(blob)
            embedding = self.recognizer.forward()
            return embedding.flatten()
        else:
            face_resized = cv2.resize(face_img, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            embedding = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            return embedding.flatten() / np.linalg.norm(embedding.flatten())
    
    def compute_distance(self, emb1, emb2):
        """Compute distance between two embeddings"""
        if self.use_cosine:
            return cosine(emb1, emb2)
        else:
            return euclidean(emb1, emb2)
    
    def match_or_create_id(self, embedding):
        """
        Match embedding against database with improved accuracy
        
        Returns:
            Tuple of (person_id, is_new, confidence)
        """
        if not self.face_db:
            person_id = f"P{self.next_id:03d}"
            self.next_id += 1
            return person_id, True, 0.0
        
        best_match_id = None
        best_distance = float('inf')
        
        for person_id, data in self.face_db.items():
            distance = self.compute_distance(embedding, data['embedding'])
            if distance < best_distance:
                best_distance = distance
                best_match_id = person_id
        
        # Calculate confidence (inverse of distance)
        confidence = 1 - min(best_distance, 1.0)
        
        if best_distance < self.similarity_threshold:
            return best_match_id, False, confidence
        else:
            person_id = f"P{self.next_id:03d}"
            self.next_id += 1
            return person_id, True, 0.0
    
    def update_database(self, person_id, embedding, is_new=False):
        """Update face database with temporal smoothing"""
        current_time = datetime.now().isoformat()
        
        # Initialize detection history
        if person_id not in self.detection_history:
            self.detection_history[person_id] = deque(maxlen=self.history_size)
        
        self.detection_history[person_id].append(embedding)
        
        # Use averaged embedding for stability
        avg_embedding = np.mean(list(self.detection_history[person_id]), axis=0)
        
        if is_new:
            self.face_db[person_id] = {
                'embedding': avg_embedding,
                'first_seen': current_time,
                'last_seen': current_time,
                'appearances': 1,
                'name': None,  # NEW: Name field
                'is_erc_member': False  # NEW: ERC member flag
            }
        else:
            self.face_db[person_id]['last_seen'] = current_time
            self.face_db[person_id]['appearances'] += 1
            # Smoother update for known faces
            old_emb = self.face_db[person_id]['embedding']
            self.face_db[person_id]['embedding'] = 0.85 * old_emb + 0.15 * avg_embedding
    
    def assign_name(self, person_id, name):
        """
        Assign a name to a person ID
        
        Args:
            person_id: Person identifier (e.g., 'P001')
            name: Name to assign
        """
        if person_id in self.face_db:
            self.face_db[person_id]['name'] = name
            self.face_db[person_id]['is_erc_member'] = True
            print(f"✓ Assigned name '{name}' to {person_id}")
            self.save_database()
            return True
        else:
            print(f"✗ Person ID {person_id} not found in database")
            return False
    
    def cleanup_strangers(self):
        """
        Remove all unnamed (stranger) entries from database
        Keep only entries with assigned names (ERC members)
        """
        before_count = len(self.face_db)
        
        # Keep only entries with names
        self.face_db = {
            pid: data for pid, data in self.face_db.items() 
            if data.get('name') is not None
        }
        
        # Clear detection history for removed persons
        removed_ids = [pid for pid in self.detection_history if pid not in self.face_db]
        for pid in removed_ids:
            del self.detection_history[pid]
        
        after_count = len(self.face_db)
        removed = before_count - after_count
        
        print(f"✓ Cleaned database: removed {removed} stranger(s), kept {after_count} named person(s)")
        self.save_database()
        return removed
    
    def list_all_persons(self):
        """List all persons in the database"""
        print("\n" + "="*70)
        print("DATABASE CONTENTS")
        print("="*70)
        
        if not self.face_db:
            print("Database is empty")
            return
        
        named = []
        unnamed = []
        
        for person_id, data in sorted(self.face_db.items()):
            entry = {
                'id': person_id,
                'name': data.get('name', 'UNNAMED'),
                'appearances': data['appearances'],
                'first_seen': data['first_seen'][:19],  # Trim milliseconds
                'last_seen': data['last_seen'][:19],
                'erc_member': data.get('is_erc_member', False)
            }
            
            if data.get('name'):
                named.append(entry)
            else:
                unnamed.append(entry)
        
        if named:
            print(f"\nERC MEMBERS ({len(named)}):")
            print("-"*70)
            for e in named:
                print(f"{e['id']}: {e['name']} | Appearances: {e['appearances']} | "
                      f"Last seen: {e['last_seen']}")
        
        if unnamed:
            print(f"\nSTRANGERS ({len(unnamed)}):")
            print("-"*70)
            for e in unnamed:
                print(f"{e['id']}: [UNNAMED] | Appearances: {e['appearances']} | "
                      f"Last seen: {e['last_seen']}")
        
        print("="*70 + "\n")
    
    def export_database(self, json_path='face_database.json'):
        """Export database to JSON for backup/inspection"""
        export_data = {}
        for person_id, data in self.face_db.items():
            export_data[person_id] = {
                'name': data.get('name'),
                'appearances': data['appearances'],
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'is_erc_member': data.get('is_erc_member', False)
            }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Database exported to {json_path}")
    
    def save_database(self):
        """Save face database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'face_db': self.face_db,
                    'next_id': self.next_id
                }, f)
            print(f"✓ Database saved to {self.db_path}")
        except Exception as e:
            print(f"✗ Error saving database: {e}")
    
    def load_database(self):
        """Load face database from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_db = data['face_db']
                    self.next_id = data['next_id']
                    
                    # Migrate old database format
                    for person_id in self.face_db:
                        if 'name' not in self.face_db[person_id]:
                            self.face_db[person_id]['name'] = None
                        if 'is_erc_member' not in self.face_db[person_id]:
                            self.face_db[person_id]['is_erc_member'] = False
                    
                print(f"✓ Database loaded from {self.db_path}")
            except Exception as e:
                print(f"✗ Error loading database: {e}")
                self.face_db = {}
                self.next_id = 1
    
    def process_frame(self, frame):
        """Process a single frame with enhanced annotations"""
        faces = self.detect_faces(frame)
        
        for (x, y, w, h, confidence) in faces:
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            
            embedding = self.get_embedding(face_img)
            person_id, is_new, match_confidence = self.match_or_create_id(embedding)
            self.update_database(person_id, embedding, is_new)
            
            # Get person data
            person_data = self.face_db[person_id]
            name = person_data.get('name')
            is_erc = person_data.get('is_erc_member', False)
            
            # Color coding: Blue=ERC Member, Green=Known, Orange=New
            if is_erc:
                color = (255, 100, 0)  # Blue
            elif not is_new:
                color = (0, 255, 0)    # Green
            else:
                color = (0, 165, 255)  # Orange
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label
            if name:
                label = f"{name} ({person_id})"
            else:
                label = f"{person_id}"
                if is_new:
                    label += " (NEW)"
            
            appearances = person_data['appearances']
            label += f" [{appearances}]"
            
            if not is_new:
                label += f" {match_confidence:.0%}"
            
            # Draw label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
        
        # Display statistics
        erc_count = sum(1 for d in self.face_db.values() if d.get('is_erc_member'))
        stats = f"Total: {len(self.face_db)} | ERC: {erc_count} | Current: {len(faces)}"
        cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self, video_source=0):
        """Run the enhanced face identification system"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open video source {video_source}")
            return
        
        print("\n" + "="*70)
        print("FACE IDENTIFICATION SYSTEM - RUNNING")
        print("="*70)
        print("CONTROLS:")
        print("  q - Quit")
        print("  s - Save database")
        print("  n - Assign name to person")
        print("  l - List all persons")
        print("  c - Cleanup strangers")
        print("  e - Export database to JSON")
        print("  r - Reset entire database")
        print("="*70 + "\n")
        
        frame_count = 0
        save_interval = 100
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            annotated_frame = self.process_frame(frame)
            cv2.imshow('Vulcan CV', annotated_frame)
            
            frame_count += 1
            if frame_count % save_interval == 0:
                self.save_database()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_database()
            elif key == ord('n'):
                # Assign name
                cv2.waitKey(1)  # Clear buffer
                person_id = input("\nEnter Person ID (e.g., P001): ").strip().upper()
                name = input("Enter Name: ").strip()
                if person_id and name:
                    self.assign_name(person_id, name)
            elif key == ord('l'):
                self.list_all_persons()
            elif key == ord('c'):
                response = input("\nCleanup strangers? This will remove all unnamed entries. (yes/no): ")
                if response.lower() == 'yes':
                    self.cleanup_strangers()
            elif key == ord('e'):
                self.export_database()
            elif key == ord('r'):
                response = input("\nReset entire database? (yes/no): ")
                if response.lower() == 'yes':
                    self.face_db = {}
                    self.next_id = 1
                    self.detection_history = {}
                    print("✓ Database reset complete")
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_database()
        print("\n✓ System shutdown complete")


def main():
    """Main entry point with menu"""
    print("="*70)
    print("ENHANCED FACE IDENTIFICATION SYSTEM FOR VULCAN ERC")
    print("="*70)
    
    mode = input("\nSelect mode:\n1. Normal Mode\n2. Training Mode (for ERC members)\nChoice (1/2): ").strip()
    training_mode = (mode == '2')
    
    system = FaceIdentificationSystem(
        db_path='face_database.pkl',
        similarity_threshold=0.55,  # Slightly stricter for better accuracy
        use_cosine=True,
        training_mode=training_mode,
        min_confidence=0.7
    )
    
    # Pre-run menu
    while True:
        print("\nPRE-RUN OPTIONS:")
        print("1. Start camera")
        print("2. List all persons")
        print("3. Assign name to person")
        print("4. Cleanup strangers")
        print("5. Export database")
        choice = input("Choice (1-5): ").strip()
        
        if choice == '1':
            break
        elif choice == '2':
            system.list_all_persons()
        elif choice == '3':
            person_id = input("Enter Person ID: ").strip().upper()
            name = input("Enter Name: ").strip()
            if person_id and name:
                system.assign_name(person_id, name)
        elif choice == '4':
            system.cleanup_strangers()
        elif choice == '5':
            system.export_database()
    
    system.run(video_source=0)


if __name__ == "__main__":
    main()