# YOLO Vision — Agent Guide

A Python application for real-time person/object detection, multi-object tracking, and person Re-ID using YOLOv11 models in multiple formats (PyTorch fp32/fp16, OpenVINO fp32/INT8) with ByteTrack/BoTSORT tracking and an EMA-based Re-ID gallery.

---

## Architecture Overview

### Core Modules

| Module | Purpose |
|--------|---------|
| **app.py** | Tkinter GUI entry point; manages UI rendering, video sources, and configuration UI |
| **detection_engine.py** | Core inference pipeline: YOLO detection → tracking → Re-ID feature extraction |
| **system_utils.py** | System hardware detection and intelligent model recommendations |
| **config.yaml** | Centralized configuration for detection, tracking, Re-ID, and hardware settings |

### Key Dependencies

- **ultralytics** (YOLO v11): Detection models, built-in trackers (ByteTrack, BoTSORT)
- **openvino**: INT8/FP32 quantized model inference (CPU-efficient)
- **opencv-python**: Video I/O and frame annotation
- **torchreid** (optional): OSNet/ResNet Re-ID models (fallback: 64-dim HSV histogram)
- **fast-reid** (submodule): FastReID models for person Re-ID

### Model Organization

```
models/
  ├── fp32/          PyTorch .pt (baseline accuracy) + OpenVINO .xml/.bin
  ├── fp16/          PyTorch .pt (FP16 - falls back to FP32 on CPU-only)
  └── openvino/      INT8 quantized .xml/.bin (fastest, CPU-optimized)
```

**Selection Convention**: Model names encode format and precision: `{model}_{precision}_{backend}` (e.g., `yolo11s_int8`, `yolo11n_fp32`).

---

## Key Design Patterns & Conventions

### 1. **Config-Driven Architecture**
- All runtime parameters live in [config.yaml](config.yaml). **Avoid hardcoding values.**
- Load via `DetectionEngine.load_config("config.yaml")` → YAML dictionary
- Sections: `detection`, `tracking`, `reid`, `openvino`, `video`, `visualisation`, `output`

### 2. **Hardware-Aware Model Selection**
- `system_utils.recommend_models()` detects CPU/GPU and recommends optimal model
- Priorities: **INT8 (fastest) → OpenVINO FP32 → PyTorch FP32 → FP16**
- Always check `SystemSpec.has_gpu` before assuming GPU availability

### 3. **Fallback Chains**
- **Tracker fallback**: Invalid tracker name → defaults to `bytetrack`
  - Check: `DetectionEngine.set_tracker()` accepts only available trackers from `get_available_trackers()`
- **Re-ID fallback**: If `torchreid` unavailable → 64-dim HSV histogram feature
- **FP16 fallback**: FP16 models on CPU-only machine → silently use FP32

### 4. **Re-ID Gallery (EMA-Based)**
- `ReIDGallery` class maintains per-track feature embeddings
- Features updated via exponential moving average (EMA) with rate `reid.feature_update_rate`
- Matching threshold: `reid.reid_thresh` (cosine similarity; lower = stricter)
- Extraction every N frames (`reid.frame_interval`) to optimize FPS

### 5. **Threading & Video Processing**
- `VideoWorker` runs in background thread; reads frames and queues results
- GUI polls output queue for frame updates (non-blocking)
- Always call `engine.reset_trails()` when switching sources

### 6. **Annotation & Visualization**
- Draw functions use OpenCV (BGR colors, not RGB)
- Trail visualization: stores centroid history in `deque` (bounded by `visualisation.trail_length`)
- Color assignment: deterministic per track-id (hash-based, reproducible)

---

## Quick Development Tasks

### Run the Application
```bash
python app.py
```

### Load & Test the Engine
```python
from detection_engine import DetectionEngine
from system_utils import recommend_models

cfg = DetectionEngine.load_config("config.yaml")
engine = DetectionEngine(cfg)

# Check available options
print(engine.get_available_models())
print(engine.get_available_trackers())

# Process a frame
frame = cv2.imread("sample.jpg")
annotated, stats = engine.process_frame(frame)
```

### Get System Recommendations
```python
from system_utils import recommend_models
spec, recommendations = recommend_models()
print(f"Recommended Re-ID: {recommendations['reid']}")
print(f"Recommended Tracker: {recommendations['tracker']}")
```

### Modify Config Parameters at Runtime
```python
engine.set_conf_threshold(0.5)    # Update detection confidence
engine.set_tracker("botsort")     # Switch to BoTSORT (validates against available trackers)
engine.set_reid_thresh(0.35)      # Stricter Re-ID matching
```

---

## Common Conventions

### Naming
- Config keys: `snake_case` (e.g., `track_buffer`, `reid_thresh`)
- Model names: lowercase with underscores (e.g., `yolo11n_int8`)
- Track IDs: positive integers (0–300, bounded by `detection.max_detections`)
- Class IDs: follow COCO 80-class standard (0 = person)

### Thresholds & Limits
- **Confidence** (`detection.conf_threshold`): 0.0–1.0; raise to reduce false positives
- **IoU** (`detection.iou_threshold`): 0.0–1.0; lower for dense scenes
- **Re-ID** (`reid.reid_thresh`): 0.0–1.0; lower = stricter matching (typically 0.3–0.5)
- **Track buffer** (`tracking.track_buffer`): frames to keep lost track alive (default 30)

### Device Strings
- OpenVINO device: `CPU` (default), `GPU`, `AUTO`, `MULTI:CPU,GPU`
- PyTorch device: `cpu`, `cuda` (auto-detected)
- Config always specifies default in `openvino.device`

### Output Organization
- Annotated frames saved to `output/` (if `output.save_video: true`)
- Logs written per `output.log_level` (default: INFO)

---

## Important Gotchas & Optimization Tips

1. **Frame Preprocessing**
   - Input frames are resized to `detection.imgsz` (default 640×640)
   - Smaller imgsz (320) → faster but less accurate; larger (1280) → slower but more accurate

2. **Re-ID Feature Extraction Overhead**
   - Features extracted every `reid.frame_interval` frames (default 3)
   - Increase to reduce Re-ID overhead; decrease for tighter tracking

3. **OpenVINO Stream Parallelism**
   - `openvino.num_streams` (default 2) controls parallel inference
   - CPU-only; GPU uses single stream. Tune based on available cores.

4. **Tracker Selection**
   - ByteTrack (default): lighter, fewer false positives
   - BoTSORT: uses Kalman filter, better for dense crowds
   - Validate tracker name exists before setting (see fallback chain)

5. **Re-ID Model Selection**
   - `osnet_ain_x1_0` (default): balanced speed/accuracy
   - Larger models (resnet50, resnet101) → better accuracy, slower inference
   - Set `reid.weights: auto` to auto-download pretrained weights

6. **Video Source Handling**
   - `video.source: 0` → webcam device 0
   - `video.source: "path/to/video.mp4"` → file path
   - HTTP/RTSP URLs supported but may require additional codecs

---

## Testing & Validation

**Unit tests** (if applicable) should validate:
- Model loading across all formats (PyTorch, OpenVINO fp32, INT8)
- Tracker fallback behavior
- Re-ID feature extraction and EMA updates
- Config parameter validation and bounds

**Integration tests** should verify:
- End-to-end pipeline: frame → detection → tracking → Re-ID annotation
- Video I/O: webcam, file, and stream sources
- GUI responsiveness during video playback

See [README.md](README.md) for quick-start guide and model comparison table.
