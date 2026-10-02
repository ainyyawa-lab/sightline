"""Fine-tune YOLOv8 on a PPE dataset, then evaluate and export.

Usage:
    python training/train_ppe.py --epochs 50
Then run the API with the new weights:
    SIGHTLINE_MODEL_NAME=runs/detect/ppe/weights/best.pt uvicorn app.main:app
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/ppe.yaml")
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, name="ppe")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    model.export(format="onnx")  # faster CPU inference; compare latency in your README


if __name__ == "__main__":
    main()
