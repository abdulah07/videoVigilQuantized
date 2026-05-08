# YOLO Vision — Detection, Tracking & Re-ID

A modular Python application for real-time person/object detection using
YOLOv11 (PyTorch fp32/fp16) and OpenVINO INT8 quantised models, with
ByteTrack/BoTSORT tracking and Re-ID via a built-in EMA gallery.

---

## Project Structure

```
├── app.py               ← Tkinter GUI entry-point
├── detection_engine.py  ← Core detection + tracking + Re-ID logic
├── config.yaml          ← All tunable parameters (edit freely)
├── requirements.txt
└── models/
    ├── fp16/            ← yolo11n/s fp16 .pt weights
    ├── fp32/            ← yolo11n/s fp32 .pt + OpenVINO FP32 .xml
    └── openvino/        ← yolo11n/s INT8 .xml + .bin
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

For OSNet-based Re-ID (better accuracy):
```bash
pip install torchreid
```

### 2. Launch the GUI
```bash
python app.py
```

### 3. Workflow
1. **Select a model** from the drop-down (e.g. `yolo11s_int8` for fastest
   inference on CPU, `yolo11s_fp32` for best accuracy).
2. Click **⬡ Load Model** and wait for the status bar to confirm.
3. Select a **source** — leave as `Webcam (device 0)` or **📂 Browse** a
   video file.
4. Adjust **Confidence** / **IoU** sliders and toggle **Show Trail** / **Re-ID**.
5. Click **▶ START**.

---

## Model Guide

| Key                     | Format       | Precision | Speed  | Accuracy |
|-------------------------|--------------|-----------|--------|----------|
| `yolo11n_fp32`          | PyTorch .pt  | FP32      | Fast   | Good     |
| `yolo11s_fp32`          | PyTorch .pt  | FP32      | Medium | Better   |
| `yolo11n_fp16`          | PyTorch .pt  | FP16*     | Fast   | Good     |
| `yolo11s_fp16`          | PyTorch .pt  | FP16*     | Medium | Better   |
| `yolo11n_openvino_fp32` | OV .xml      | FP32      | Fast   | Good     |
| `yolo11s_openvino_fp32` | OV .xml      | FP32      | Medium | Better   |
| `yolo11n_int8`          | OV .xml INT8 | INT8      | Fastest| Good     |
| `yolo11s_int8`          | OV .xml INT8 | INT8      | Fast   | Better   |

\* FP16 falls back to FP32 on CPU-only machines.

---

## config.yaml — Key Parameters

```yaml
detection:
  conf_threshold: 0.35   # raise to reduce false positives
  iou_threshold:  0.45   # lower for dense scenes
  target_classes: [0]    # 0 = person; [] = all COCO classes
  imgsz: 640             # 320 = faster, 1280 = more accurate

tracking:
  tracker: bytetrack      # bytetrack | botsort
  track_buffer: 30        # frames to keep a lost track alive

reid:
  enabled: true
  reid_thresh: 0.4        # lower = stricter identity matching
  feature_update_rate: 0.9  # EMA weight for feature update

openvino:
  device: CPU             # CPU | GPU | AUTO
  num_streams: 2          # parallel inference streams
```

---

## Re-ID Notes

Without `torchreid` installed the engine falls back to a simple
64-dimensional HSV colour-histogram feature. This still gives useful
identity persistence but degrades with appearance changes.

Install `torchreid` for production-quality OSNet embeddings:
```bash
pip install torchreid
```

---

## Saving Output Video

Set `output.save_video: true` in `config.yaml` and specify `output_dir`.
Saved files are timestamped MP4s written by OpenCV.
