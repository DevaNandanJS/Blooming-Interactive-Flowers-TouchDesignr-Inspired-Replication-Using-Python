"""
Hand Tracking Module — MediaPipe Tasks API
Tracks 2 hands, returns per-hand openness, fingertip positions,
finger direction vectors, and full landmark data for skeleton overlay.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe hand landmark connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

# Fingertip landmark indices
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# The joint just before each fingertip (for direction computation)
FINGER_DIP_IDS = [3, 7, 11, 15, 19]

# MCP (base) joint of each finger (for openness computation)
FINGER_MCP_IDS = [2, 5, 9, 13, 17]


class HandTracker:
    def __init__(self):
        self.model_path = 'hand_landmarker.task'
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe model to {self.model_path}...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def process_frame(self, frame):
        """Process a BGR frame and detect hands."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.results = self.detector.detect(mp_image)
        return self.results

    def get_hand_data(self):
        """
        Returns a dict with per-hand data:
        {
            'hands': [
                {
                    'label': 'Left' or 'Right',
                    'openness': float 0-1 (how open the palm is),
                    'fingertips': [(x, y), ...],  # 5 normalized positions
                    'finger_directions': [(dx, dy), ...],  # 5 direction unit vectors
                    'landmarks': [(x, y), ...],  # All 21 normalized positions
                    'pinch_distance': float,  # thumb-to-index distance
                },
                ...
            ]
        }
        """
        data = {'hands': []}

        if not self.results or not self.results.hand_landmarks:
            return data

        for hand_idx, hand_landmarks in enumerate(self.results.hand_landmarks):
            # Determine handedness
            label = 'Right'
            if (self.results.handedness and
                    hand_idx < len(self.results.handedness)):
                label = self.results.handedness[hand_idx][0].category_name

            # Extract all 21 landmarks as (x, y)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks]

            # Fingertip positions
            fingertips = [landmarks[i] for i in FINGERTIP_IDS]

            # Finger direction vectors (DIP → TIP, normalized)
            finger_directions = []
            for tip_id, dip_id in zip(FINGERTIP_IDS, FINGER_DIP_IDS):
                tip = np.array(landmarks[tip_id])
                dip = np.array(landmarks[dip_id])
                direction = tip - dip
                length = np.linalg.norm(direction)
                if length > 1e-6:
                    direction = direction / length
                else:
                    direction = np.array([0.0, -1.0])  # default: upward
                finger_directions.append(tuple(direction))

            # Palm openness: average distance from each fingertip to its MCP,
            # normalized by the hand size (wrist-to-middle-MCP distance)
            wrist = np.array(landmarks[0])
            middle_mcp = np.array(landmarks[9])
            hand_size = np.linalg.norm(middle_mcp - wrist)

            if hand_size > 1e-6:
                openness_sum = 0.0
                for tip_id, mcp_id in zip(FINGERTIP_IDS, FINGER_MCP_IDS):
                    tip = np.array(landmarks[tip_id])
                    mcp = np.array(landmarks[mcp_id])
                    openness_sum += np.linalg.norm(tip - mcp) / hand_size
                openness = np.clip(openness_sum / 5.0, 0.0, 1.0)
                # Remap: fist ≈ 0.3, open hand ≈ 0.9
                openness = np.clip((openness - 0.3) / 0.6, 0.0, 1.0)
            else:
                openness = 0.0

            # Pinch distance (thumb tip to index tip) for debug display
            thumb_tip = np.array(landmarks[4])
            index_tip = np.array(landmarks[8])
            pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))

            data['hands'].append({
                'label': label,
                'openness': float(openness),
                'fingertips': fingertips,
                'finger_directions': finger_directions,
                'landmarks': landmarks,
                'pinch_distance': pinch_dist,
            })

        return data

    def draw_skeleton(self, frame, hand_data):
        """Draw hand skeletons and labels onto a BGR frame using OpenCV."""
        h, w, _ = frame.shape

        for hand in hand_data.get('hands', []):
            landmarks = hand['landmarks']
            label = hand['label']
            openness = hand['openness']

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h)
                x2, y2 = int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h)
                # Blue-purple gradient based on connection
                color = (255, 150, 50)  # BGR: warm blue-ish
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw landmark points
            for i, (lx, ly) in enumerate(landmarks):
                cx, cy = int(lx * w), int(ly * h)
                if i in FINGERTIP_IDS:
                    cv2.circle(frame, (cx, cy), 5, (0, 200, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, cy), 3, (255, 200, 100), -1, cv2.LINE_AA)

            # Draw label
            wrist_x = int(landmarks[0][0] * w)
            wrist_y = int(landmarks[0][1] * h)
            cv2.putText(frame, f"{label} | Open: {openness:.2f}",
                        (wrist_x - 50, wrist_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2, cv2.LINE_AA)

        return frame
