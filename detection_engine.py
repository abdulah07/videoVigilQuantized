"""
detection_engine.py
────────────────────────────────────────────────────────────────────────────────
Core engine for YOLO object detection + ByteTrack / BoTSORT tracking + Re-ID.
Supports:
  • PyTorch  (.pt)  – fp32 / fp16 via Ultralytics
  • OpenVINO (.xml) – fp32 or INT8 quantised models via Ultralytics / OV runtime
  • Re-ID feature extraction with EMA gallery (torchreid / built-in OSNet)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE: Dict[int, Tuple[int, int, int]] = {}


def _colour(track_id: int) -> Tuple[int, int, int]:
    """Return a stable BGR colour for a given track-id."""
    if track_id not in PALETTE:
        rng = np.random.default_rng(track_id + 42)
        PALETTE[track_id] = tuple(int(c) for c in rng.integers(80, 230, 3))
    return PALETTE[track_id]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2] between two L2-normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(1.0 - np.dot(a, b))


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Gallery
# ─────────────────────────────────────────────────────────────────────────────

class ReIDGallery:
    """
    Minimal EMA-based Re-ID gallery.
    Maps tracker IDs → persistent global IDs using cosine feature distance.
    """

    def __init__(self, thresh: float = 0.4, ema_rate: float = 0.9,
                 max_dist: float = 0.6):
        self.thresh = thresh
        self.ema_rate = ema_rate
        self.max_dist = max_dist

        self._gallery: Dict[int, np.ndarray] = {}   # reid_id → feature
        self._tracker_to_reid: Dict[int, int] = {}   # tracker_id → reid_id
        self._next_id = 1

    def update(self, tracker_id: int, feature: np.ndarray) -> int:
        """Return stable reid_id for this tracker_id, updating the gallery."""
        feat = feature / (np.linalg.norm(feature) + 1e-6)

        # Already assigned
        if tracker_id in self._tracker_to_reid:
            reid_id = self._tracker_to_reid[tracker_id]
            old = self._gallery[reid_id]
            # EMA update
            self._gallery[reid_id] = (
                self.ema_rate * old + (1 - self.ema_rate) * feat
            )
            return reid_id

        # Match against gallery
        best_id, best_dist = None, float("inf")
        for gid, gfeat in self._gallery.items():
            d = _cosine_distance(feat, gfeat)
            if d < best_dist:
                best_dist = d
                best_id = gid

        if best_id is not None and best_dist < self.max_dist:
            reid_id = best_id
        else:
            reid_id = self._next_id
            self._next_id += 1
            self._gallery[reid_id] = feat

        self._tracker_to_reid[tracker_id] = reid_id
        return reid_id


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ReIDExtractor:
    """Lightweight OSNet-based feature extractor via torchreid (if available)."""

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu"):
        self._extractor = None
        self._device = device
        try:
            import torchreid
            self._extractor = torchreid.utils.FeatureExtractor(
                model_name=model_name,
                device=device,
            )
            logger.info("Re-ID extractor loaded: %s on %s", model_name, device)
        except ImportError:
            logger.warning(
                "torchreid not installed – Re-ID will use mean-colour features. "
                "Install with: pip install torchreid"
            )

    def extract(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Return a 1-D feature vector for the given BGR crop."""
        if bgr_crop is None or bgr_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self._extractor is not None:
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            import torch
            with torch.no_grad():
                feat = self._extractor([rgb])
            return feat[0].cpu().numpy().flatten()

        # Fallback: 64-dim colour histogram feature
        resized = cv2.resize(bgr_crop, (64, 128))
        hists = []
        for ch in range(3):
            h, _ = np.histogram(resized[:, :, ch], bins=32, range=(0, 256))
            hists.append(h.astype(np.float32))
        feat = np.concatenate(hists)
        return feat / (feat.sum() + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  Detection Engine
# ─────────────────────────────────────────────────────────────────────────────

class DetectionEngine:
    """
    Wraps Ultralytics YOLO for detection + tracking.
    Handles PyTorch (.pt) and OpenVINO (.xml) models transparently.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        det = cfg.get("detection", {})
        vis = cfg.get("visualisation", {})
        reid_cfg = cfg.get("reid", {})

        self.conf = det.get("conf_threshold", 0.35)
        self.iou = det.get("iou_threshold", 0.45)
        self.classes = det.get("target_classes", [0]) or None
        self.imgsz = det.get("imgsz", 640)
        self.max_det = det.get("max_detections", 300)

        self.show_labels = vis.get("show_labels", True)
        self.show_conf = vis.get("show_confidence", True)
        self.show_tid = vis.get("show_track_id", True)
        self.show_rid = vis.get("show_reid_id", True)
        self.show_trail = vis.get("show_trail", True)
        self.trail_len = vis.get("trail_length", 40)
        self.box_thick = vis.get("box_thickness", 2)
        self.font_scale = vis.get("font_scale", 0.55)
        self.show_fps = vis.get("show_fps", True)

        # Tracking
        track_cfg = cfg.get("tracking", {})
        self._tracker_name = track_cfg.get("tracker", "bytetrack")

        # Re-ID
        self._reid_enabled = reid_cfg.get("enabled", True)
        self._reid_extractor: Optional[ReIDExtractor] = None
        self._gallery: Optional[ReIDGallery] = None
        if self._reid_enabled:
            self._reid_extractor = ReIDExtractor(
                model_name=reid_cfg.get("model", "osnet_x0_25"),
                device=reid_cfg.get("device", "cpu"),
            )
            self._gallery = ReIDGallery(
                thresh=reid_cfg.get("reid_thresh", 0.4),
                ema_rate=reid_cfg.get("feature_update_rate", 0.9),
                max_dist=reid_cfg.get("max_reid_distance", 0.6),
            )

        # Trail history  track_id → deque of (cx, cy)
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_len)
        )

        self._model = None
        self._model_path: Optional[str] = None

        # FPS
        self._fps_buffer: deque = deque(maxlen=30)
        self._t_prev = time.perf_counter()

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self, model_key: str):
        """
        Load (or reload) a model by its key from config.

        Ultralytics YOLO expects:
          • .pt  files  → pass path directly
          • OpenVINO    → pass the *folder* that contains
                          <name>.xml + <name>.bin + metadata.yaml
                          (NOT the bare .xml path)
        """
        paths = self.cfg.get("models", {})
        model_path = paths.get(model_key, "").strip()
        if not model_path:
            raise ValueError(f"Model key '{model_key}' not found in config.")

        p = Path(model_path)

        # If config still points to a bare .xml, auto-redirect to its parent folder
        if p.suffix.lower() == ".xml":
            logger.warning(
                "Path points to a .xml file ('%s'). "
                "Ultralytics needs the *folder* for OpenVINO models. "
                "Redirecting to parent folder: '%s'", p, p.parent
            )
            p = p.parent

        if not p.exists():
            raise FileNotFoundError(
                f"Model path not found: {p}\n"
                "For OpenVINO models the path must be a folder containing:\n"
                "  <name>.xml  +  <name>.bin  +  metadata.yaml\n"
                "Example: models/openvino/yolo11s_int8_openvino_model/"
            )

        from ultralytics import YOLO  # lazy import

        logger.info("Loading model: %s  (%s)", model_key, str(p))
        self._model = YOLO(str(p))
        self._model_path = str(p)
        self._model_key = model_key
        logger.info("Model ready.")

    # ── Per-frame inference ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run detection + tracking + Re-ID on one BGR frame.
        Returns (annotated_frame, stats_dict).
        """
        if self._model is None:
            return frame, {}

        ov_cfg = self.cfg.get("openvino", {})
        device_arg = ov_cfg.get("device", "CPU") if self._is_openvino() else None

        # ── Inference ─────────────────────────────────────────────────────────
        kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            max_det=self.max_det,
            tracker=f"{self._tracker_name}.yaml",
            persist=True,
            verbose=False,
        )
        if device_arg:
            kwargs["device"] = device_arg

        results = self._model.track(frame, **kwargs)

        # ── Build annotated frame ─────────────────────────────────────────────
        annotated = frame.copy()
        stats = {"detections": 0, "tracks": 0}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            n = len(boxes)
            stats["detections"] = n

            for i in range(n):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf_val = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                tid = int(boxes.id[i].cpu()) if boxes.id is not None else -1

                x1, y1, x2, y2 = xyxy
                colour = _colour(tid)

                # Re-ID
                reid_id = None
                if self._reid_enabled and tid >= 0:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        feat = self._reid_extractor.extract(crop)
                        reid_id = self._gallery.update(tid, feat)

                # Trail
                if self.show_trail and tid >= 0:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self._trails[tid].append((cx, cy))
                    pts = list(self._trails[tid])
                    for j in range(1, len(pts)):
                        alpha = j / len(pts)
                        t_col = tuple(int(c * alpha) for c in colour)
                        cv2.line(annotated, pts[j - 1], pts[j], t_col,
                                 max(1, self.box_thick - 1))

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour,
                               self.box_thick)

                # Label
                label_parts = []
                if self.show_labels:
                    cls_name = results[0].names.get(cls_id, str(cls_id))
                    label_parts.append(cls_name)
                if self.show_conf:
                    label_parts.append(f"{conf_val:.2f}")
                if self.show_tid and tid >= 0:
                    label_parts.append(f"T{tid}")
                if self.show_rid and reid_id is not None:
                    label_parts.append(f"R{reid_id}")

                if label_parts:
                    label = " ".join(label_parts)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, 1
                    )
                    cv2.rectangle(annotated,
                                  (x1, y1 - th - 6), (x1 + tw + 4, y1),
                                  colour, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.font_scale, (255, 255, 255), 1,
                                cv2.LINE_AA)

            if boxes.id is not None:
                stats["tracks"] = int(boxes.id.numel())

        # ── FPS overlay ───────────────────────────────────────────────────────
        now = time.perf_counter()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            self._fps_buffer.append(1.0 / dt)
        fps = sum(self._fps_buffer) / max(len(self._fps_buffer), 1)
        stats["fps"] = round(fps, 1)

        if self.show_fps:
            cv2.putText(annotated, f"FPS {fps:.1f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 100), 2, cv2.LINE_AA)

        return annotated, stats

    # ── Utility ───────────────────────────────────────────────────────────────

    def _is_openvino(self) -> bool:
        p = Path(self._model_path or "")
        # OpenVINO model is either a folder or a .xml file
        return p.is_dir() or p.suffix.lower() == ".xml"

    def reset_trails(self):
        self._trails.clear()

    @staticmethod
    def load_config(path: str = "config.yaml") -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)
