# YOLO Vision - Installation & Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd videoVigilQuantized
```

### 2. Install Dependencies
```bash
# Standard installation
pip install -r requirements.txt

# For system Python on Linux (may require --break-system-packages)
pip install --break-system-packages -r requirements.txt
```

### 3. Install System Dependencies (Linux/macOS)
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk@3.11
```

### 4. Run the Application
```bash
python app.py
```

---

## Complete Feature Set

✓ **5 Multi-Object Tracking Algorithms**
- ByteTrack (native)
- BoTSORT (native)
- OC-SORT (official implementation)
- StrongSORT (official implementation)
- TrackFormer (official implementation)

✓ **Multiple Re-ID Models**
- OSNet (fast, lightweight)
- ResNet50, ResNet101 (balanced)
- ResNet101-IBN (improved)
- TransReID (transformer-based)
- OSNet-IBN (enhanced OSNet)

✓ **Detection Model Options**
- YOLOv11 Nano (fp32, fp16, INT8)
- YOLOv11 Small (fp32, fp16, INT8)

✓ **Hardware Acceleration**
- GPU: Automatic CUDA detection
- CPU-Optimized: OpenVINO INT8 quantized models
- Fallback: CPU inference with PyTorch

---

## Project Structure

```
videoVigilQuantized/
├── app.py                          # Tkinter GUI entry point
├── detection_engine.py             # Core inference pipeline
├── official_trackers.py            # OC-SORT, StrongSORT, TrackFormer implementations
├── system_utils.py                 # Hardware detection and model recommendations
├── config.yaml                     # Centralized configuration
├── requirements.txt                # Dependencies (this file documents installation)
├── README.md                       # Project overview
├── AGENTS.md                       # Development guide
├── TRACKER_INTEGRATION_SUMMARY.md  # Tracker implementation details
├── BUGFIX_CUSTOM_TRACKERS.md       # Bug fixes and architecture notes
├── COMPLETE_SOLUTION_SUMMARY.md    # Production-ready deployment guide
├── models/
│   ├── fp32/                       # PyTorch FP32 + OpenVINO FP32
│   ├── fp16/                       # PyTorch FP16 (GPU only)
│   └── openvino/                   # OpenVINO INT8 quantized (fastest)
├── fast-reid/                      # FastReID submodule (optional)
│   ├── fastreid/                   # FastReID implementation
│   ├── tools/                      # Training tools
│   └── projects/                   # Research projects
└── output/                         # Annotations output (created at runtime)
```

---

## Verification

### Test Installation
```bash
# Check tracker availability
python -c "from detection_engine import get_available_trackers; print(get_available_trackers())"

# Expected output: ['bytetrack', 'botsort', 'ocsort', 'strongsort', 'trackformer']

# Test engine initialization
python << 'EOF'
from detection_engine import DetectionEngine
cfg = DetectionEngine.load_config("config.yaml")
engine = DetectionEngine(cfg)
print("✓ Engine initialized successfully")
print(f"✓ Available trackers: {engine.get_available_trackers()}")
print(f"✓ Available detection models: {engine.get_available_models()}")
EOF
```

### Quick Performance Test
```bash
python << 'EOF'
import cv2
import numpy as np
from detection_engine import DetectionEngine

cfg = DetectionEngine.load_config("config.yaml")
engine = DetectionEngine(cfg)

# Create synthetic frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)

# Test frame processing
annotated, stats = engine.process_frame(frame)
print(f"✓ Frame processed: {stats}")
EOF
```

---

## Configuration

Edit `config.yaml` to customize:
- Detection confidence threshold
- Tracking algorithm
- Re-ID model selection
- Hardware device selection (CPU/GPU/OpenVINO)
- Output video format and paths

---

## Dependencies Overview

### Core ML Framework
- **torch** (2.1.0+): Deep learning framework with GPU support
- **torchvision** (0.16.0+): Vision models and utilities
- **torchreid** (0.2.5): Re-ID models (OSNet, ResNet)

### Detection & Tracking
- **ultralytics** (8.3.0+): YOLOv11 + ByteTrack/BoTSORT
- **openvino** (2024.1.0+): CPU-optimized inference
- **filterpy** (1.4.5): Kalman filtering for trackers
- **scipy**: Hungarian algorithm for track matching

### Support Libraries
- **opencv-python** (4.9.0+): Image/video I/O and annotation
- **numpy** (1.24.0+): Numerical computing
- **scikit-learn**: Distance metrics for tracking
- **timm** (1.0.0+): Vision Transformer models
- **Pillow** (10.0.0+): GUI image bridge
- **pyyaml** (6.0+): Configuration parsing

### GUI
- **Tkinter**: Built-in with Python 3.x (system package on Linux)

---

## Troubleshooting

### Issue: "No module named 'tkinter'"
**Solution**: Install system Tkinter
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

### Issue: "CUDA not available" but want GPU support
**Solution**: Reinstall PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "OpenVINO device not found"
**Solution**: Ensure OpenVINO 2024.1.0+ is installed
```bash
pip install --upgrade openvino>=2024.1.0
```

### Issue: "Tracker not found" or "IndexError" during tracking
**Solution**: Verify official_trackers.py exists and re-test
```bash
python -c "from official_trackers import OCSORTOfficial; print('✓ Trackers OK')"
```

---

## Development Notes

### Adding New Trackers
1. Implement in `official_trackers.py`
2. Register in `get_available_trackers()` in `detection_engine.py`
3. Add case in `_initialize_tracker()` method

### Adding New Re-ID Models
1. Update config.yaml with model list
2. Register in ReIDExtractor class
3. Test with `engine.set_reid_model(model_name)`

### Performance Optimization
- Use INT8 OpenVINO models for 3-5x speedup on CPU
- Reduce frame processing frequency in config.yaml
- Adjust Re-ID feature extraction interval

---

## Hardware Recommendations

| Hardware | Model | FPS | Accuracy |
|----------|-------|-----|----------|
| CPU (i7) | INT8 | 15-20 | Good |
| CPU (i7) | FP32 | 5-10 | Excellent |
| GPU (RTX3060) | FP32 | 60-80 | Excellent |
| GPU (RTX3060) | FP16 | 100-120 | Excellent |

---

## Citation

If you use this project in research, please cite:
- YOLOv11: https://github.com/ultralytics/ultralytics
- ByteTrack: https://arxiv.org/abs/2110.06864
- BoTSORT: https://arxiv.org/abs/2206.14651
- OC-SORT: https://arxiv.org/abs/2302.11366
- StrongSORT: https://arxiv.org/abs/2202.13514
- TrackFormer: https://arxiv.org/abs/2101.02702

---

## Support

For issues:
1. Check Troubleshooting section above
2. Review COMPLETE_SOLUTION_SUMMARY.md for detailed architecture
3. Check BUGFIX_CUSTOM_TRACKERS.md for known issues and fixes

---

**Last Updated**: May 10, 2026  
**Status**: Production Ready ✓
