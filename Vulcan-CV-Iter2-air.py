"""
Enhanced Real-Time Face Detection, Pose Estimation, and Hand Tracking System
for Vulcan ERC Humanoid

New Features:
- Full body pose estimation (33 landmarks)
- Hand skeleton tracking (21 landmarks per hand)
- Separate visualization grids for hands and posture
- Toggle controls for each feature
- Real-time skeleton visualization
- Combined view for main display and grids

Author: Aaditya Kushawaha, Anirudh Singh Air
Project: Vulcan ERC
Enhanced with Pose & Hand Tracking + Visualization Grids
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

# MediaPipe for pose and hand tracking
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not installed. Install with: pip install mediapipe")

class FaceIdentificationSystem:
    def __init__(self,
                 db_path='face_database.pkl',
                 similarity_threshold=0.4,
                 use_cosine=True,
                 training_mode=False,
                 min_confidence=0.7,
                 enable_pose=True,
                 enable_hands=True):
        """
        Initialize the Enhanced Face Identification System with Pose & Hand Tracking

        Args:
            db_path: Path to save/load face database
            similarity_threshold: Maximum distance for matching faces
            use_cosine: Use cosine distance (True) or Euclidean (False)
            training_mode: Enable training mode for ERC members
            min_confidence: Minimum confidence for face detection
            enable_pose: Enable pose estimation
            enable_hands: Enable hand tracking
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.use_cosine = use_cosine
        self.training_mode = training_mode
        self.min_confidence = min_confidence
        self.face_db = {}
        self.next_id = 1

        # Feature toggles
        self.enable_pose = enable_pose and MEDIAPIPE_AVAILABLE
        self.enable_hands = enable_hands and MEDIAPIPE_AVAILABLE
        self.enable_face_detection = True
        self.show_grids = True  # Toggle for side grids

        # Grid dimensions
        self.grid_width = 400
        self.grid_height = 300

        # Track recent detections for stability
        self.detection_history = {}
        self.history_size = 5

        # Load existing database if available
        self.load_database()

        # Initialize face detector and recognizer
        self.detector = self._load_face_detector()
        self.recognizer = self._load_face_recognizer()

        # Initialize MediaPipe Pose and Hands
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            if self.enable_pose:
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1
                )
                print("✓ Pose estimation initialized")

            if self.enable_hands:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✓ Hand tracking initialized")

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
        """Detect faces in the frame with confidence scores"""
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
        """Match embedding against database with improved accuracy"""
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

        if person_id not in self.detection_history:
            self.detection_history[person_id] = deque(maxlen=self.history_size)

        self.detection_history[person_id].append(embedding)
        avg_embedding = np.mean(list(self.detection_history[person_id]), axis=0)

        if is_new:
            self.face_db[person_id] = {
                'embedding': avg_embedding,
                'first_seen': current_time,
                'last_seen': current_time,
                'appearances': 1,
                'name': None,
                'is_erc_member': False
            }
        else:
            self.face_db[person_id]['last_seen'] = current_time
            self.face_db[person_id]['appearances'] += 1
            old_emb = self.face_db[person_id]['embedding']
            self.face_db[person_id]['embedding'] = 0.85 * old_emb + 0.15 * avg_embedding

    def create_pose_grid(self, pose_results):
        """Create a dedicated visualization grid for pose"""
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        # Add grid background with lines
        cv2.rectangle(grid, (0, 0), (self.grid_width, self.grid_height), (30, 30, 30), -1)

        # Draw grid lines
        for i in range(0, self.grid_width, 50):
            cv2.line(grid, (i, 0), (i, self.grid_height), (50, 50, 50), 1)
        for i in range(0, self.grid_height, 50):
            cv2.line(grid, (0, i), (self.grid_width, i), (50, 50, 50), 1)

        # Add title
        cv2.putText(grid, "POSTURE ANALYSIS", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if not pose_results or not pose_results.pose_landmarks:
            cv2.putText(grid, "No pose detected", (self.grid_width//2 - 80, self.grid_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            return grid

        # Draw pose on grid
        landmarks = pose_results.pose_landmarks.landmark

        # Scale landmarks to grid
        points = []
        for lm in landmarks:
            x = int(lm.x * self.grid_width)
            y = int(lm.y * self.grid_height)
            points.append((x, y, lm.visibility))

        # Draw connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(points) and end_idx < len(points) and
                points[start_idx][2] > 0.5 and points[end_idx][2] > 0.5):
                cv2.line(grid, points[start_idx][:2], points[end_idx][:2],
                        (0, 255, 0), 2)

        # Draw landmarks
        for idx, (x, y, vis) in enumerate(points):
            if vis > 0.5:
                # Color code different body parts
                if idx in [11, 12, 23, 24]:  # Shoulders and hips
                    color = (0, 0, 255)  # Red
                elif idx in [13, 14, 15, 16]:  # Arms
                    color = (255, 0, 0)  # Blue
                elif idx in [25, 26, 27, 28]:  # Legs
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 255, 255)  # White
                cv2.circle(grid, (x, y), 4, color, -1)

        # Calculate and display posture metrics
        if (11 < len(points) and 12 < len(points) and
            23 < len(points) and 24 < len(points)):

            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
                left_hip.visibility > 0.5 and right_hip.visibility > 0.5):

                shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
                hip_slope = abs(left_hip.y - right_hip.y)

                # Spine alignment (vertical)
                spine_alignment = abs((left_shoulder.x + right_shoulder.x) / 2 -
                                     (left_hip.x + right_hip.x) / 2)

                posture_status = "GOOD" if (shoulder_slope < 0.05 and
                                           hip_slope < 0.05 and
                                           spine_alignment < 0.05) else "NEEDS ADJUSTMENT"
                color = (0, 255, 0) if posture_status == "GOOD" else (0, 165, 255)

                # Display metrics
                y_offset = 50
                cv2.putText(grid, f"Status: {posture_status}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(grid, f"Shoulder Level: {shoulder_slope:.3f}", (10, y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.putText(grid, f"Hip Level: {hip_slope:.3f}", (10, y_offset + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.putText(grid, f"Spine Alignment: {spine_alignment:.3f}", (10, y_offset + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return grid

    def create_hand_grid(self, hand_results):
        """Create a dedicated visualization grid for hands"""
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        # Add grid background
        cv2.rectangle(grid, (0, 0), (self.grid_width, self.grid_height), (30, 30, 30), -1)

        # Draw grid lines
        for i in range(0, self.grid_width, 50):
            cv2.line(grid, (i, 0), (i, self.grid_height), (50, 50, 50), 1)
        for i in range(0, self.grid_height, 50):
            cv2.line(grid, (0, i), (self.grid_width, i), (50, 50, 50), 1)

        # Add title
        cv2.putText(grid, "HAND TRACKING", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if not hand_results or not hand_results.multi_hand_landmarks:
            cv2.putText(grid, "No hands detected", (self.grid_width//2 - 90, self.grid_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            return grid

        # Calculate grid sections for multiple hands
        num_hands = len(hand_results.multi_hand_landmarks)
        section_width = self.grid_width // max(num_hands, 1)

        for idx, (hand_landmarks, handedness) in enumerate(
            zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):

            # Get hand label
            hand_label = handedness.classification[0].label
            hand_score = handedness.classification[0].score

            # Calculate offset for this hand
            x_offset = idx * section_width

            # Scale landmarks to grid section
            points = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * section_width) + x_offset
                y = int(lm.y * (self.grid_height - 80)) + 60  # Leave space for label
                points.append((x, y))

            # Draw connections
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                cv2.line(grid, points[start_idx], points[end_idx],
                        (0, 255, 255), 2)

            # Draw landmarks with color coding
            finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

            for lm_idx, (x, y) in enumerate(points):
                if lm_idx in finger_tips:
                    color = (0, 0, 255)  # Red for fingertips
                    radius = 6
                elif lm_idx == 0:
                    color = (255, 0, 255)  # Magenta for wrist
                    radius = 8
                else:
                    color = (255, 255, 0)  # Yellow for other joints
                    radius = 4
                cv2.circle(grid, (x, y), radius, color, -1)

            # Draw hand label
            label_x = x_offset + section_width // 2 - 50
            cv2.putText(grid, f"{hand_label} Hand", (label_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(grid, f"Conf: {hand_score:.2f}", (label_x, self.grid_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Calculate and display finger states
            self._draw_finger_states(grid, points, x_offset, section_width)

        return grid

    def _draw_finger_states(self, grid, points, x_offset, section_width):
        """Draw finger extended/bent states"""
        # Simple heuristic: if fingertip is above middle joint, finger is extended
        fingers = {
            'Thumb': (4, 3),
            'Index': (8, 6),
            'Middle': (12, 10),
            'Ring': (16, 14),
            'Pinky': (20, 18)
        }

        y_pos = 70
        for finger_name, (tip_idx, mid_idx) in fingers.items():
            if tip_idx < len(points) and mid_idx < len(points):
                is_extended = points[tip_idx][1] < points[mid_idx][1]
                status = "Extended" if is_extended else "Bent"
                color = (0, 255, 0) if is_extended else (100, 100, 100)

                text = f"{finger_name[0]}: {status}"
                text_x = x_offset + 10
                cv2.putText(grid, text, (text_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_pos += 20

    def draw_pose_landmarks(self, frame, results):
        """Draw pose skeleton on frame"""
        if not results.pose_landmarks:
            return

        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
            left_hip.visibility > 0.5 and right_hip.visibility > 0.5):

            shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
            hip_slope = abs(left_hip.y - right_hip.y)

            posture_status = "Good" if shoulder_slope < 0.05 and hip_slope < 0.05 else "Tilted"
            color = (0, 255, 0) if posture_status == "Good" else (0, 165, 255)

            cv2.putText(frame, f"Posture: {posture_status}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_hand_landmarks(self, frame, results):
        """Draw hand skeleton on frame"""
        if not results.multi_hand_landmarks:
            return

        hand_count = 0
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

            hand_label = handedness.classification[0].label
            hand_score = handedness.classification[0].score

            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            x, y = int(wrist.x * w), int(wrist.y * h)

            cv2.putText(frame, f"{hand_label} ({hand_score:.2f})", (x - 50, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            hand_count += 1

        cv2.putText(frame, f"Hands: {hand_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def assign_name(self, person_id, name):
        """Assign a name to a person ID"""
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
        """Remove all unnamed (stranger) entries from database"""
        before_count = len(self.face_db)

        self.face_db = {
            pid: data for pid, data in self.face_db.items()
            if data.get('name') is not None
        }

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
                'first_seen': data['first_seen'][:19],
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
        """Process a single frame with face detection, pose, and hand tracking"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = None
        hand_results = None

        # Process pose
        if self.enable_pose:
            pose_results = self.pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                self.draw_pose_landmarks(frame, pose_results)

        # Process hands
        if self.enable_hands:
            hand_results = self.hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                self.draw_hand_landmarks(frame, hand_results)

        # Process faces
        if self.enable_face_detection:
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

                person_data = self.face_db[person_id]
                name = person_data.get('name')
                is_erc = person_data.get('is_erc_member', False)

                if is_erc:
                    color = (255, 100, 0)
                elif not is_new:
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

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

                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 2)

        # Display statistics
        erc_count = sum(1 for d in self.face_db.values() if d.get('is_erc_member'))
        stats = f"Total: {len(self.face_db)} | ERC: {erc_count}"
        cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2)

        features = []
        if self.enable_face_detection:
            features.append("Face")
        if self.enable_pose:
            features.append("Pose")
        if self.enable_hands:
            features.append("Hands")

        status = f"Active: {', '.join(features)}"
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)

        # Create visualization grids
        pose_grid = None
        hand_grid = None

        if self.show_grids:
            if self.enable_pose:
                pose_grid = self.create_pose_grid(pose_results)
            if self.enable_hands:
                hand_grid = self.create_hand_grid(hand_results)

        return frame, pose_grid, hand_grid

    def run(self, video_source=0):
        """Run the enhanced face identification system"""
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"✗ Error: Could not open video source {video_source}")
            return

        print("\n" + "="*70)
        print("FACE ID + POSE + HAND TRACKING SYSTEM - RUNNING")
        print("="*70)
        print("CONTROLS:")
        print("  q - Quit")
        print("  s - Save database")
        print("  n - Assign name to person")
        print("  l - List all persons")
        print("  c - Cleanup strangers")
        print("  e - Export database to JSON")
        print("  r - Reset entire database")
        print("  f - Toggle face detection")
        print("  p - Toggle pose estimation")
        print("  h - Toggle hand tracking")
        print("  g - Toggle visualization grids")
        print("="*70 + "\n")

        frame_count = 0
        save_interval = 100

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            annotated_frame, pose_grid, hand_grid = self.process_frame(frame)

            # --- Start of new combined display logic ---
            if self.show_grids:
                # Create placeholders if grids are disabled or None
                if pose_grid is None:
                    pose_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
                    cv2.putText(pose_grid, "Pose Disabled", (self.grid_width//2 - 60, self.grid_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                if hand_grid is None:
                    hand_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
                    cv2.putText(hand_grid, "Hands Disabled", (self.grid_width//2 - 60, self.grid_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

                # Vertically stack the two grids to create a side panel
                side_panel = np.vstack((pose_grid, hand_grid))

                # Get dimensions for resizing
                main_h, main_w, _ = annotated_frame.shape
                side_h, side_w, _ = side_panel.shape

                # Resize the main frame and side panel to have the same height
                # while maintaining aspect ratio
                combined_h = max(main_h, side_h)

                # Resize main frame
                scale_main = combined_h / main_h
                new_main_w = int(main_w * scale_main)
                resized_main = cv2.resize(annotated_frame, (new_main_w, combined_h))

                # Resize side panel
                scale_side = combined_h / side_h
                new_side_w = int(side_w * scale_side)
                resized_side = cv2.resize(side_panel, (new_side_w, combined_h))

                # Horizontally stack the main frame and the side panel
                combined_frame = np.hstack((resized_main, resized_side))

                cv2.imshow('Vulcan Enhanced CV - Combined View', combined_frame)
            else:
                # If grids are not shown, just display the main frame
                cv2.imshow('Vulcan Enhanced CV - Combined View', annotated_frame)
            # --- End of new combined display logic ---

            frame_count += 1
            if frame_count % save_interval == 0:
                self.save_database()

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_database()
            elif key == ord('f'):
                self.enable_face_detection = not self.enable_face_detection
                print(f"Face detection: {'ON' if self.enable_face_detection else 'OFF'}")
            elif key == ord('p'):
                if MEDIAPIPE_AVAILABLE:
                    self.enable_pose = not self.enable_pose
                    print(f"Pose estimation: {'ON' if self.enable_pose else 'OFF'}")
            elif key == ord('h'):
                if MEDIAPIPE_AVAILABLE:
                    self.enable_hands = not self.enable_hands
                    print(f"Hand tracking: {'ON' if self.enable_hands else 'OFF'}")
            elif key == ord('g'):
                self.show_grids = not self.show_grids
                print(f"Visualization grids: {'ON' if self.show_grids else 'OFF'}")
            elif key == ord('n'):
                cv2.waitKey(1)
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
    print("ENHANCED FACE ID + POSE + HAND TRACKING SYSTEM FOR VULCAN ERC")
    print("="*70)

    if not MEDIAPIPE_AVAILABLE:
        print("\n⚠ WARNING: MediaPipe not available")
        print("Install with: pip install mediapipe")
        print("Continuing with face detection only...\n")

    mode = input("\nSelect mode:\n1. Normal Mode\n2. Training Mode (for ERC members)\nChoice (1/2): ").strip()
    training_mode = (mode == '2')

    enable_pose = True
    enable_hands = True

    if MEDIAPIPE_AVAILABLE:
        features = input("\nEnable features:\n1. All (Face + Pose + Hands)\n2. Custom\nChoice (1/2): ").strip()
        if features == '2':
            enable_pose = input("Enable pose estimation? (y/n): ").lower() == 'y'
            enable_hands = input("Enable hand tracking? (y/n): ").lower() == 'y'

    system = FaceIdentificationSystem(
        db_path='face_database.pkl',
        similarity_threshold=0.55,
        use_cosine=True,
        training_mode=training_mode,
        min_confidence=0.7,
        enable_pose=enable_pose,
        enable_hands=enable_hands
    )

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