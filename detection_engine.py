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
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from official_trackers import OCSORTOfficial, StrongSORTOfficial, TrackFormerOfficial

# Auto-discover FastReID if available
_fast_reid_paths = [
    "/home/sunny/FYP/videoVigilQuantized/fast-reid",
    "../fast-reid",
    "./fast-reid",
]
for _path in _fast_reid_paths:
    _p = Path(_path).resolve()
    if _p.exists() and _p not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)


def get_available_trackers() -> List[str]:
    """
    Get list of ALL available trackers.
    Includes both Ultralytics native (ByteTrack, BoTSort) and official implementations (OC-SORT, StrongSORT, TrackFormer).
    """
    # All trackers: 2 native + 3 official implementations
    ALL_TRACKERS = ["bytetrack", "botsort", "ocsort", "strongsort", "trackformer"]
    
    try:
        # Try importing TRACKER_DIR from ultralytics
        try:
            from ultralytics.cfg import TRACKER_DIR
            tracker_dir = Path(TRACKER_DIR)
        except (ImportError, AttributeError):
            # Fall back to finding the trackers directory manually
            import ultralytics
            uv_path = Path(ultralytics.__file__).parent
            tracker_dir = uv_path / "cfg" / "trackers"
        
        if tracker_dir.exists():
            yaml_files = sorted(list(tracker_dir.glob("*.yaml")))
            available = [f.stem for f in yaml_files if f.stem in ALL_TRACKERS]
            # Add custom trackers
            custom = ["ocsort", "strongsort", "trackformer"]
            for ct in custom:
                if ct not in available:
                    available.append(ct)
            logger.info("✓ Found %d trackers (2 native + 3 custom): %s", len(available), sorted(available))
            return sorted(available) if available else ["bytetrack"]
    except Exception as e:
        logger.warning("Error detecting trackers: %s", str(e))
    
    logger.info("Using all trackers: %s", ALL_TRACKERS)
    return ALL_TRACKERS


def get_available_reid_models() -> Dict[str, bool]:
    """
    Check which Re-ID models are available and can be loaded.
    Returns a dict: {model_name: bool (can_load)}
    """
    reid_models = [
        # OSNet variants
        "osnet_ain_x1_0", "osnet_x1_0", "osnet_x0_25",
        # ResNet variants
        "resnet50", "resnet101",
        # FastReID
        "fastraid", "osnet_ibn",
        # TransReID
        "transreid",
        # VitREID
        "vitreid"
    ]
    
    available = {}
    torchreid_available = False
    timm_available = False
    
    try:
        import torchreid
        torchreid_available = True
    except ImportError:
        pass
    
    try:
        import timm
        timm_available = True
    except ImportError:
        pass
    
    for model in reid_models:
        # Check if the model can be loaded
        framework = ReIDExtractor._detect_framework(model)
        if framework == "torchreid":
            available[model] = torchreid_available
        elif framework == "vitreid":
            available[model] = timm_available
        else:
            # fastraid and others: always try to load (with fallback)
            available[model] = True
    
    return available


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
            logger.debug("ReID match: tracker=%d matched existing R%d (dist=%.4f < %.4f)",
                         tracker_id, reid_id, best_dist, self.max_dist)
        else:
            reid_id = self._next_id
            self._next_id += 1
            self._gallery[reid_id] = feat
            logger.debug("ReID new: tracker=%d assigned new R%d (best_dist=%.4f)",
                         tracker_id, reid_id, best_dist)

        self._tracker_to_reid[tracker_id] = reid_id
        return reid_id

    def get_tracker_reid(self, tracker_id: int) -> Optional[int]:
        """Return the current Re-ID for a tracker, if one already exists."""
        return self._tracker_to_reid.get(tracker_id)

    def has_tracker(self, tracker_id: int) -> bool:
        return tracker_id in self._tracker_to_reid


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ID Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ReIDExtractor:
    """
    Lightweight feature extractor supporting:
      • OSNet (osnet_ain_x1_0, osnet_x1_0, osnet_x0_25) via torchreid
      • ResNet (resnet50, resnet101) via torchreid
      • FastReID models via fast-reid framework (requires fast-reid installed)
      • TransReID models via torchreid
      • VitREID (Vision Transformer) via timm
    """

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu"):
        self._extractor = None
        self._device = device
        self._model_name = model_name
        self._framework = self._detect_framework(model_name)
        self._fallback_used = False

        try:
            if self._framework == "torchreid":
                import torchreid
                self._extractor = torchreid.utils.FeatureExtractor(
                    model_name=model_name,
                    device=device,
                )
                logger.info("✓ Re-ID extractor loaded (torchreid): %s on %s", model_name, device)
            elif self._framework == "fastraid":
                try:
                    from fast_reid.fastreid.modeling import build_model
                    from fast_reid.fastreid.config import get_cfg
                    logger.info("✓ FastReID framework detected")
                    logger.warning("FastReID requires model config; using fallback feature extraction")
                    self._fallback_used = True
                except ImportError:
                    logger.warning("fast-reid not installed – using fallback features")
                    self._fallback_used = True
            elif self._framework == "vitreid":
                try:
                    import timm
                    logger.info("✓ Vision Transformer Re-ID framework detected (timm)")
                    # Note: VitReID support is partial; using fallback
                    self._fallback_used = True
                except ImportError:
                    logger.warning("timm not installed for VitREID – using fallback features")
                    self._fallback_used = True
        except (ImportError, Exception) as e:
            logger.warning(
                "Could not load %s Re-ID model (%s) – using fallback colour histogram. "
                "Install with: pip install torchreid  OR  pip install fast-reid  OR  pip install timm",
                self._framework, str(e)
            )
            self._fallback_used = True

    @staticmethod
    def _detect_framework(model_name: str) -> str:
        """Detect which framework a model name belongs to."""
        if model_name.startswith("osnet") or model_name.startswith("resnet"):
            return "torchreid"
        elif model_name.startswith("fastraid") or model_name.startswith("osnet_ibn"):
            return "fastraid"
        elif model_name.startswith("transreid"):
            return "torchreid"  # TransReID is in torchreid
        elif model_name.startswith("vitreid") or model_name.startswith("vit"):
            return "vitreid"
        else:
            return "torchreid"  # default

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
        self._reid_cfg = cfg.get("reid", {}).copy()
        det = cfg.get("detection", {})
        vis = cfg.get("visualisation", {})
        reid_cfg = self._reid_cfg

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
        self._custom_tracker = None  # For custom tracker implementations
        self._initialize_tracker()

        # Re-ID
        self._reid_enabled = reid_cfg.get("enabled", True)
        self._reid_model_name: Optional[str] = None
        self._reid_extractor_cache: Dict[str, ReIDExtractor] = {}
        self._reid_extractor: Optional[ReIDExtractor] = None
        self._gallery: Optional[ReIDGallery] = None
        if self._reid_enabled:
            self.set_reid_model(reid_cfg.get("model", "osnet_x0_25"))
        self._reid_frame_interval = max(1, int(reid_cfg.get("frame_interval", 3)))

        # Trail history  track_id → deque of (cx, cy)
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_len)
        )

        self._model = None
        self._model_path: Optional[str] = None

        # FPS
        self._fps_buffer: deque = deque(maxlen=30)
        self._t_prev = time.perf_counter()
        self._frame_idx = 0

    def set_reid_model(self, model_name: str):
        """Switch Re-ID model with error handling."""
        if self._reid_enabled and model_name == self._reid_model_name and self._reid_extractor is not None:
            return

        self._reid_cfg["model"] = model_name
        self._reid_model_name = model_name
        if not self._reid_enabled:
            return
        
        try:
            if model_name not in self._reid_extractor_cache:
                logger.info("Initializing Re-ID model: %s", model_name)
                self._reid_extractor_cache[model_name] = ReIDExtractor(
                    model_name=model_name,
                    device=self._reid_cfg.get("device", "cpu"),
                )
            self._reid_extractor = self._reid_extractor_cache[model_name]
            logger.info("✓ Re-ID model set to: %s", model_name)
            self._gallery = ReIDGallery(
                thresh=self._reid_cfg.get("reid_thresh", 0.4),
                ema_rate=self._reid_cfg.get("feature_update_rate", 0.9),
                max_dist=self._reid_cfg.get("max_reid_distance", 0.6),
            )
            self._reid_frame_interval = max(1, int(self._reid_cfg.get("frame_interval", 3)))
        except Exception as e:
            logger.error("Failed to load Re-ID model '%s': %s. Using fallback.", model_name, str(e))
            # Use a fallback extractor
            if "fallback" not in self._reid_extractor_cache:
                self._reid_extractor_cache["fallback"] = ReIDExtractor(
                    model_name="osnet_x0_25",
                    device=self._reid_cfg.get("device", "cpu"),
                )
            self._reid_extractor = self._reid_extractor_cache["fallback"]

    def set_tracker(self, tracker_name: str):
        """
        Switch tracking backend dynamically.
        Supported: bytetrack, botsort (native) + ocsort, strongsort, trackformer (custom)
        """
        available = get_available_trackers()
        
        if tracker_name not in available:
            logger.warning(
                "Tracker '%s' not available (available: %s). Falling back to bytetrack.",
                tracker_name, available
            )
            tracker_name = "bytetrack"
        
        self._tracker_name = tracker_name
        self.cfg["tracking"]["tracker"] = tracker_name
        self._initialize_tracker()
        logger.info("✓ Tracker switched to: %s", tracker_name)
    
    def _initialize_tracker(self):
        """Initialize the appropriate tracker based on tracker_name."""
        if self._tracker_name in ["bytetrack", "botsort"]:
            # Native Ultralytics trackers - will be handled by YOLO.track()
            self._custom_tracker = None
            logger.info("✓ Using native Ultralytics tracker: %s", self._tracker_name)
        elif self._tracker_name == "ocsort":
            self._custom_tracker = OCSORTOfficial()
            logger.info("✓ Using official OC-SORT tracker with Kalman filtering")
        elif self._tracker_name == "strongsort":
            self._custom_tracker = StrongSORTOfficial()
            logger.info("✓ Using official StrongSORT tracker with Kalman filtering")
        elif self._tracker_name == "trackformer":
            self._custom_tracker = TrackFormerOfficial()
            logger.info("✓ Using official TrackFormer tracker with attention-based matching")
        else:
            logger.warning("Unknown tracker '%s', using bytetrack", self._tracker_name)
            self._tracker_name = "bytetrack"
            self._custom_tracker = None

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
        self._model = YOLO(str(p), task="detect")
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

        self._frame_idx += 1

        ov_cfg = self.cfg.get("openvino", {})
        device_arg = ov_cfg.get("device", "CPU") if self._is_openvino() else None

        # ── Inference ─────────────────────────────────────────────────────────
        # For custom trackers, run detection only
        if self._custom_tracker is not None:
            kwargs = dict(
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
            if device_arg:
                kwargs["device"] = device_arg
            
            try:
                results = self._model.predict(frame, **kwargs)
            except Exception as e:
                logger.error("Detection failed: %s", str(e))
                return frame, {}
            
            # Apply custom tracking
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                detections = []
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())
                    detections.append([*xyxy, conf, cls_id])
                
                detections = np.array(detections) if detections else np.empty((0, 6))
                tracked = self._custom_tracker.update(detections)
                
                # Convert tracked results to Results format
                class NumpyWrapper:
                    """Wrapper to make numpy arrays compatible with torch tensor operations."""
                    def __init__(self, arr):
                        if isinstance(arr, np.ndarray):
                            self.arr = arr.item() if arr.size == 1 else arr
                        else:
                            self.arr = arr
                    
                    def cpu(self):
                        """Return self (already on CPU as numpy)."""
                        return self
                    
                    def numpy(self):
                        """Return numpy array."""
                        if isinstance(self.arr, np.ndarray):
                            return self.arr
                        return np.array([self.arr])
                    
                    def __float__(self):
                        """Support float conversion."""
                        if isinstance(self.arr, np.ndarray):
                            return float(self.arr.item())
                        return float(self.arr)
                    
                    def __int__(self):
                        """Support int conversion."""
                        if isinstance(self.arr, np.ndarray):
                            return int(self.arr.item())
                        return int(self.arr)
                    
                    def numel(self):
                        """Return number of elements (PyTorch compatibility)."""
                        if isinstance(self.arr, np.ndarray):
                            return self.arr.size
                        return 1
                
                class TensorLikeList(list):
                    """List subclass that acts like a PyTorch tensor for .numel()."""
                    def numel(self):
                        """Return number of elements."""
                        return len(self)
                
                class TrackedBoxesWrapper:
                    """Wrapper for tracked boxes from custom trackers."""
                    def __init__(self, tracked_arr):
                        self.xyxy = [NumpyWrapper(tracked_arr[i, :4]) for i in range(len(tracked_arr))] if len(tracked_arr) > 0 else []
                        self.conf = [NumpyWrapper(tracked_arr[i, 4]) for i in range(len(tracked_arr))] if len(tracked_arr) > 0 else []
                        self.cls = [NumpyWrapper(tracked_arr[i, 5]) for i in range(len(tracked_arr))] if len(tracked_arr) > 0 else []
                        # Use special list that has .numel() method
                        id_list = TensorLikeList([NumpyWrapper(tracked_arr[i, 6]) for i in range(len(tracked_arr))])
                        self.id = id_list if len(id_list) > 0 else None
                        self._len = len(tracked_arr)
                    
                    def __len__(self):
                        return self._len
                
                tracked_boxes = TrackedBoxesWrapper(tracked)
                
                result = type('obj', (object,), {
                    'boxes': tracked_boxes,
                    'names': results[0].names
                })()
                results = [result]
        else:
            # Use native Ultralytics tracker (ByteTrack or BoTSort)
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

            try:
                results = self._model.track(frame, **kwargs)
            except AssertionError as e:
                logger.error("Tracker not supported: %s", str(e))
                # Fall back to bytetrack
                logger.info("Falling back to bytetrack")
                self._tracker_name = "bytetrack"
                self._custom_tracker = None
                kwargs["tracker"] = "bytetrack.yaml"
                results = self._model.track(frame, **kwargs)
            except Exception as e:
                logger.error("Tracking failed: %s", str(e))
                return frame, {}

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
                cls_name = results[0].names.get(cls_id, str(cls_id))
                tid = int(boxes.id[i].cpu()) if boxes.id is not None else -1

                x1, y1, x2, y2 = xyxy
                colour = _colour(tid)

                # Re-ID
                reid_id = None
                if self._reid_enabled and tid >= 0:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if self._gallery.has_tracker(tid):
                        reid_id = self._gallery.get_tracker_reid(tid)
                    if crop.size > 0 and (
                        not self._gallery.has_tracker(tid)
                        or self._frame_idx % self._reid_frame_interval == 0
                    ):
                        feat = self._reid_extractor.extract(crop)
                        reid_id = self._gallery.update(tid, feat)
                        logger.debug("ReID assign: T%d -> R%d (cls=%s conf=%.2f)",
                                     tid, reid_id, cls_name, conf_val)

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
