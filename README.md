---
title: SightLine
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 👁️ SightLine: Object Detection API

Production-style computer vision service built with **FastAPI + YOLOv8 + OpenCV**.
Upload an image or video and get detections as JSON or an annotated result.

## Features
- `POST /detect`: JSON detections (label, confidence, box), class summary, inference time
- `POST /detect/annotated`: image with boxes drawn
- `POST /video` → `GET /video/{id}` → `GET /video/{id}/result`: async video processing with progress polling
- **Object tracking** (ByteTrack): video jobs return the number of *unique* objects per class, not just per-frame detections
- **Custom model training**: `training/train_ppe.py` fine-tunes YOLOv8 (e.g. helmet/vest detection) and exports to ONNX
- Web UI at `/`, interactive API docs at `/docs`
- Env-based config (`SIGHTLINE_MODEL_NAME`, `SIGHTLINE_CONF_THRESHOLD`)
- Dependency-injected detector, so tests run without model weights
- Dockerfile included

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```
Docker: `docker build -t sightline . && docker run -p 7860:7860 sightline` (then open http://localhost:7860)

## Test
```bash
pytest -q
```

## Architecture
```
Client ──> FastAPI (validation, jobs) ──> Detector (YOLOv8) ──> OpenCV (annotate / video I/O)
```

## Custom PPE model
1. Download a YOLO-format PPE dataset (Roboflow Universe) into `datasets/ppe`
2. `python training/train_ppe.py --epochs 50`
3. `SIGHTLINE_MODEL_NAME=runs/detect/ppe/weights/best.pt uvicorn app.main:app`

Report your mAP50 and the ONNX vs PyTorch latency here.

## Roadmap (great talking points in interviews)
- Swap in-memory job store for Redis + Celery
- Line-crossing counts (entries/exits) on top of tracking
- Fine-tune on a custom dataset (e.g. PPE / helmet detection)
- Export to ONNX / TensorRT for faster inference
- WebSocket live webcam stream
