# 🎯 Crowd Detection — YOLOv8 + OpenCV

Real-time **person detection** and **crowd counting** using **YOLOv8** and **OpenCV**.

---

## 📁 Project Structure

```
MODEL/
├── main.py          ← Entry point (run this)
├── camera.py        ← Webcam capture handler
├── detector.py      ← YOLOv8 inference (persons only)
├── display.py       ← Bounding boxes, HUD, FPS overlay
├── logger.py        ← Periodic CSV crowd-count logger
├── config.py        ← All tuneable constants
├── requirements.txt ← Python dependencies
└── logs/
    └── crowd_log.csv  ← Auto-created on first run
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: replace the `torch` line in `requirements.txt` with the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Run

```bash
python main.py
```

The YOLOv8 nano weights (~6 MB) are downloaded automatically on the first run.

---

## 🖥️ HUD Elements

| Element | Description |
|---|---|
| Green boxes | Bounding box around each detected person |
| `Person XX%` label | Confidence score per detection |
| **Persons** counter | Live crowd count |
| **FPS** display | Frames per second (EMA-smoothed) |
| Density bar | Green → Orange → Red as crowd grows |
| Quit hint | Bottom-right corner |

---

## ⚙️ Configuration (`config.py`)

| Constant | Default | Effect |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `MODEL_PATH` | `yolov8n.pt` | YOLOv8 weight file |
| `CONFIDENCE_THRESHOLD` | `0.40` | Min detection confidence |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `FRAME_WIDTH/HEIGHT` | `1280×720` | Requested capture resolution |
| `CSV_LOG_PATH` | `logs/crowd_log.csv` | Output CSV file path |
| `CSV_LOG_INTERVAL` | `30` | Seconds between log entries |

---

## 📊 CSV Logging

Every **30 seconds** the system appends a row to `logs/crowd_log.csv`:

```csv
timestamp,crowd_count
2026-06-13 13:30:00,4
2026-06-13 13:30:30,7
2026-06-13 13:31:00,3
```

- The `logs/` folder and the CSV file are **created automatically** on first run.
- Existing log data is **preserved** across runs (append mode).
- Change `CSV_LOG_INTERVAL` in `config.py` to adjust the save frequency.

---

## 🚩 CLI Flags

```bash
python main.py --camera 1              # Use external camera (index 1)
python main.py --model yolov8s.pt      # Use a larger, more accurate model
python main.py --conf 0.55             # Raise confidence threshold
```

---

## 🔑 Controls

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |

---

## 🔧 Model Accuracy vs Speed

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ⚡ Fastest | Good |
| `yolov8s.pt` | 22 MB | Fast | Better |
| `yolov8m.pt` | 52 MB | Medium | Great |
| `yolov8l.pt` | 87 MB | Slow | Excellent |
| `yolov8x.pt` | 131 MB | Slowest | Best |

Change `MODEL_PATH` in `config.py` or use `--model` at runtime.
