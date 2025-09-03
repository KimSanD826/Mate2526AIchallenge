import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time

class KalmanFilter:
    """Kalman Filter for tracking object motion"""
    def __init__(self):
        # State: [x, y, vx, vy]
        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 10
        
        self.x = np.zeros((4, 1), dtype=np.float32)
        
    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2].flatten()
    
    def update(self, measurement):
        # Ensure measurement is 2D column vector
        if measurement.ndim == 1:
            measurement = measurement.reshape(-1, 1)
        
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()

class Track:
    """Individual track for a detected object"""
    def __init__(self, bbox, feature, track_id):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.feature = feature
        self.kalman = KalmanFilter()
        
        # Fix the broadcasting issue by properly reshaping the array
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.kalman.x[:2] = np.array([[center_x], [center_y]])
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        
        # Store feature history for re-identification
        self.feature_history = deque(maxlen=50)
        self.feature_history.append(feature)
        
    def predict(self):
        """Predict next position using Kalman filter"""
        predicted_pos = self.kalman.predict()
        center_x, center_y = predicted_pos
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        
        self.bbox = [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox, feature):
        """Update track with new detection"""
        self.bbox = bbox
        self.feature = feature
        self.feature_history.append(feature)
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Fix the broadcasting issue by properly reshaping the measurement array
        measurement = np.array([[center_x], [center_y]])
        self.kalman.update(measurement)
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
    def get_feature_similarity(self, feature):
        """Calculate feature similarity for re-identification"""
        if len(self.feature_history) == 0:
            return 0
        
        # Use cosine similarity with the most recent features
        recent_features = list(self.feature_history)[-5:]  # Last 5 features
        similarities = [cosine_similarity([f], [feature])[0][0] for f in recent_features]
        return max(similarities)

class DeepSORTTracker:
    """DeepSORT tracker implementation"""
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, feature_threshold=0.7):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        
        
    def iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def extract_features(self, frame, bbox):
        """Extract deep features from detected region"""
        # Simple feature extraction using the image patch
        # In a real implementation, you'd use a deep neural network
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))
        
        # Ensure valid patch dimensions
        if x2 <= x1 or y2 <= y1:
            return np.zeros(128)
        
        try:
            patch = frame[y1:y2, x1:x2]
            
            if patch.size == 0:
                return np.zeros(128)
            
            # Resize to fixed size and flatten
            patch_resized = cv2.resize(patch, (16, 16))
            features = patch_resized.flatten()[:128]  # Take first 128 values
            
            # Normalize features
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            return features
        except Exception:
            return np.zeros(128)
    
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Extract features for new detections
        detection_features = []
        for det in detections:
            bbox = det[:4]  # Assuming det is [x1, y1, x2, y2, conf, class]
            feature = self.extract_features(frame, bbox)
            detection_features.append(feature)
        
        # Predict new positions for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Initialize variables to avoid UnboundLocalError
        track_indices = []
        detection_indices = []
        cost_matrix = np.zeros((0, 0))  # Initialize empty cost matrix
        
        # Associate detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            
            # Calculate cost matrix
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for i, track in enumerate(self.tracks):
                for j, (det, feature) in enumerate(zip(detections, detection_features)):
                    # IOU cost
                    iou_cost = 1 - self.iou(track.bbox, det[:4])
                    
                    # Feature similarity cost
                    feature_cost = 1 - track.get_feature_similarity(feature)
                    
                    # Combined cost (weighted average)
                    cost_matrix[i, j] = 0.7 * iou_cost + 0.3 * feature_cost
            
            # Hungarian algorithm for assignment
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, det_idx] < self.iou_threshold:
                    track = self.tracks[track_idx]
                    track.update(detections[det_idx][:4], detection_features[det_idx])
        
        # Create new tracks for unmatched detections
        matched_detections = set()
        if len(track_indices) > 0 and len(detection_indices) > 0:
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, det_idx] < self.iou_threshold:
                    matched_detections.add(det_idx)
        
        for i, (det, feature) in enumerate(zip(detections, detection_features)):
            if i not in matched_detections:
                track = Track(det[:4], feature, self.next_id)
                self.tracks.append(track)
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update < self.max_age and track.hits >= self.min_hits]
        
        return self.tracks
    
    def get_tracked_objects(self):
        """Get current tracked objects with IDs"""
        tracked_objects = []
        for track in self.tracks:
            if track.time_since_update == 0:  # Only return updated tracks
                tracked_objects.append({
                    'id': track.track_id,
                    'bbox': track.bbox,
                    'age': track.age,
                    'hits': track.hits
                })
        return tracked_objects
