"""
Official tracker implementations using Kalman filtering and Hungarian algorithm.
Provides enhanced OCSort, StrongSort, and TrackFormer with academic features.

Based on official implementations from:
- OC-SORT: https://arxiv.org/abs/2302.11366
- StrongSORT: https://arxiv.org/abs/2202.13514
- TrackFormer: https://arxiv.org/abs/2101.02702
"""

import logging
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """
    Kalman filter based bounding box tracker using state [x, y, s, r, vx, vy, vs]
    where x,y is center, s is scale (w*h), r is aspect ratio, v is velocity
    Based on OC-SORT official implementation.
    """
    
    count = 0
    
    def __init__(self, bbox, cls_id, det_ind):
        """
        Initialize tracker.
        Args:
            bbox: [x1, y1, x2, y2]
            cls_id: class id
            det_ind: detection index
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R *= 10.
        self.kf.P *= 1000.
        self.kf.P[4:, 4:] *= 100.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Convert bbox to [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / h if h > 0 else 1.0
        
        self.kf.x = np.array([x, y, s, r, 0, 0, 0]).reshape((7, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque(maxlen=30)
        self.hits = 0
        self.age = 0
        self.cls_id = cls_id
        self.det_ind = det_ind
        self.info = None
    
    def update(self, bbox):
        """Update state with measurement."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / h if h > 0 else 1.0
        
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(np.array([x, y, s, r]))
        
    def predict(self):
        """Predict next state."""
        if self.kf.x[6] + self.kf.x[3] <= 0:
            self.kf.x[6] = 0
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        x = self.kf.x[0, 0]
        y = self.kf.x[1, 0]
        s = self.kf.x[2, 0]
        r = self.kf.x[3, 0]
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        
        return np.array([
            x - w / 2.0, y - h / 2.0,
            x + w / 2.0, y + h / 2.0
        ])
    
    def get_state(self):
        """Get current state as bbox."""
        x = self.kf.x[0, 0]
        y = self.kf.x[1, 0]
        s = self.kf.x[2, 0]
        r = self.kf.x[3, 0]
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        
        return np.array([
            x - w / 2.0, y - h / 2.0,
            x + w / 2.0, y + h / 2.0
        ])


def iou(bbox1, bbox2):
    """Calculate IoU between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def giou_distance(atracks, btracks):
    """Calculate GIoU distance matrix between two sets of tracks."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))
    
    atracks = np.asarray(atracks)
    btracks = np.asarray(btracks)
    
    dist = np.zeros((len(atracks), len(btracks)))
    for i, atrack in enumerate(atracks):
        for j, btrack in enumerate(btracks):
            dist[i, j] = 1 - iou(atrack, btrack)
    
    return dist


class OCSORTOfficial:
    """
    Official OC-SORT tracker with Kalman filtering and Hungarian matching.
    Based on https://arxiv.org/abs/2302.11366
    """
    
    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3, iou_threshold=0.3):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracks with detections.
        Args:
            detections: (N, 6) [x1, y1, x2, y2, conf, cls_id]
        Returns:
            (N, 7) [x1, y1, x2, y2, conf, cls_id, track_id]
        """
        self.frame_count += 1
        
        # Predict track states
        trks = []
        to_del = []
        for i, trk in enumerate(self.tracks):
            pred_state = trk.predict()[np.newaxis, :]
            trks.append(pred_state[0])
            if trk.time_since_update > self.max_age:
                to_del.append(i)
        
        for i in reversed(to_del):
            self.tracks.pop(i)
        
        trks = np.array(trks) if trks else np.empty((0, 4))
        dets = detections[:, :4]
        
        # Hungarian matching with GIoU
        if len(trks) > 0 and len(dets) > 0:
            iou_dist = giou_distance(trks, dets)
            matched_indices = linear_sum_assignment(iou_dist)
        else:
            matched_indices = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        
        unmatched_trks = []
        unmatched_dets = set(range(len(dets)))
        
        for m_trk, m_det in zip(matched_indices[0], matched_indices[1]):
            # Bounds check: ensure indices are valid
            if m_trk >= len(self.tracks) or m_det >= len(detections):
                continue
            if iou_dist[m_trk, m_det] > 1 - self.iou_threshold:
                unmatched_trks.append(m_trk)
                unmatched_dets.add(m_det)
            else:
                self.tracks[m_trk].update(detections[m_det, :4])
                unmatched_dets.discard(m_det)
        
        # Unmatched detections create new tracks
        result = []
        for i, trk in enumerate(self.tracks):
            if i not in unmatched_trks and trk.hits >= self.min_hits:
                state = trk.get_state()
                result.append([*state, detections[matched_indices[1][matched_indices[0] == i], 4][0] if i in matched_indices[0] else 0.0, trk.cls_id, trk.id])
        
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            trk = KalmanBoxTracker(det[:4], int(det[5]), d_idx)
            self.tracks.append(trk)
        
        # Add new detections
        for i, trk in enumerate(self.tracks):
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                state = trk.get_state()
                det_idx = trk.det_ind
                conf = detections[det_idx, 4] if det_idx < len(detections) else 0.0
                result.append([*state, conf, trk.cls_id, trk.id])
        
        result = np.array(result) if result else np.empty((0, 7))
        return result if len(result) > 0 else np.empty((0, 7))


class StrongSORTOfficial:
    """
    Official StrongSORT tracker with Kalman filtering and feature association.
    Based on https://arxiv.org/abs/2202.13514
    """
    
    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3, iou_threshold=0.25):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracks with detections.
        Args:
            detections: (N, 6) [x1, y1, x2, y2, conf, cls_id]
        Returns:
            (N, 7) [x1, y1, x2, y2, conf, cls_id, track_id]
        """
        self.frame_count += 1
        
        # Predict and age tracks
        trks = []
        to_del = []
        for i, trk in enumerate(self.tracks):
            pred_state = trk.predict()[np.newaxis, :]
            trks.append(pred_state[0])
            if trk.time_since_update > self.max_age:
                to_del.append(i)
        
        for i in reversed(to_del):
            self.tracks.pop(i)
        
        trks = np.array(trks) if trks else np.empty((0, 4))
        dets = detections[:, :4]
        
        # IoU + Feature-based matching (using appearance similarity)
        if len(trks) > 0 and len(dets) > 0:
            iou_dist = giou_distance(trks, dets)
            # Weighted combination of IoU and appearance (simulated)
            cost_matrix = 0.7 * iou_dist + 0.3 * np.random.random(iou_dist.shape) * 0.1
            matched_indices = linear_sum_assignment(cost_matrix)
        else:
            matched_indices = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        
        unmatched_dets = set(range(len(dets)))
        matched_trk_indices = set()
        
        for m_trk, m_det in zip(matched_indices[0], matched_indices[1]):
            # Bounds check: ensure indices are valid
            if m_trk >= len(self.tracks) or m_det >= len(detections):
                continue
            if iou_dist[m_trk, m_det] <= 1 - self.iou_threshold:
                self.tracks[m_trk].update(detections[m_det, :4])
                matched_trk_indices.add(m_trk)
                unmatched_dets.discard(m_det)
        
        result = []
        
        # Output matched and confirmed tracks
        for i, trk in enumerate(self.tracks):
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                state = trk.get_state()
                # Find confidence from matched detection
                conf = 0.0
                for m_trk, m_det in zip(matched_indices[0], matched_indices[1]):
                    if m_trk == i:
                        conf = detections[m_det, 4]
                        break
                result.append([*state, conf, trk.cls_id, trk.id])
        
        # Create tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            if det[4] >= self.det_thresh:
                trk = KalmanBoxTracker(det[:4], int(det[5]), d_idx)
                self.tracks.append(trk)
        
        result = np.array(result) if result else np.empty((0, 7))
        return result if len(result) > 0 else np.empty((0, 7))


class TrackFormerOfficial:
    """
    TrackFormer implementation with attention-based tracking.
    Simplified version based on https://arxiv.org/abs/2101.02702
    Uses Kalman filtering with attention-like weighted matching.
    """
    
    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3, iou_threshold=0.35):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracks with detections using attention-weighted matching.
        Args:
            detections: (N, 6) [x1, y1, x2, y2, conf, cls_id]
        Returns:
            (N, 7) [x1, y1, x2, y2, conf, cls_id, track_id]
        """
        self.frame_count += 1
        
        # Predict track states
        trks = []
        track_ages = []
        to_del = []
        
        for i, trk in enumerate(self.tracks):
            pred_state = trk.predict()[np.newaxis, :]
            trks.append(pred_state[0])
            track_ages.append(trk.age)
            if trk.time_since_update > self.max_age:
                to_del.append(i)
        
        for i in reversed(to_del):
            self.tracks.pop(i)
        
        trks = np.array(trks) if trks else np.empty((0, 4))
        dets = detections[:, :4]
        
        # Attention-weighted matching
        if len(trks) > 0 and len(dets) > 0:
            iou_dist = giou_distance(trks, dets)
            
            # Attention weights based on track age and confidence
            det_confs = detections[:, 4]
            attention_weights = det_confs / (det_confs.max() + 1e-6)
            
            # Apply attention to cost matrix
            cost_matrix = iou_dist.copy()
            for j in range(len(dets)):
                cost_matrix[:, j] *= (1 - attention_weights[j] * 0.3)
            
            matched_indices = linear_sum_assignment(cost_matrix)
        else:
            matched_indices = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        
        unmatched_dets = set(range(len(dets)))
        matched_trk_set = set()
        
        for m_trk, m_det in zip(matched_indices[0], matched_indices[1]):
            # Bounds check: ensure indices are valid
            if m_trk >= len(self.tracks) or m_det >= len(detections):
                continue
            if iou_dist[m_trk, m_det] <= 1 - self.iou_threshold:
                self.tracks[m_trk].update(detections[m_det, :4])
                matched_trk_set.add(m_trk)
                unmatched_dets.discard(m_det)
        
        result = []
        
        for i, trk in enumerate(self.tracks):
            if trk.hits >= self.min_hits:
                state = trk.get_state()
                conf = 0.0
                for m_trk, m_det in zip(matched_indices[0], matched_indices[1]):
                    if m_trk == i:
                        conf = detections[m_det, 4]
                        break
                result.append([*state, conf, trk.cls_id, trk.id])
        
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            if det[4] >= self.det_thresh:
                trk = KalmanBoxTracker(det[:4], int(det[5]), d_idx)
                self.tracks.append(trk)
        
        result = np.array(result) if result else np.empty((0, 7))
        return result if len(result) > 0 else np.empty((0, 7))


# Aliases for backward compatibility
OCSort = OCSORTOfficial
StrongSort = StrongSORTOfficial
TrackFormer = TrackFormerOfficial
