"""Thin wrapper around YOLOv8 so the API layer stays model-agnostic."""
import threading
import time
from collections import Counter
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from .config import settings


class Detector:
    def __init__(self, model_name: str = settings.model_name):
        self.model = YOLO(model_name)  # downloads weights on first run
        self.names = self.model.names
        self._video_lock = threading.Lock()  # tracker state is per-model

    def _predict(self, frame: np.ndarray, conf: float):
        return self.model.predict(frame, conf=conf, verbose=False)[0]

    def detect(self, frame: np.ndarray, conf: float | None = None) -> dict:
        conf = conf if conf is not None else settings.conf_threshold
        t0 = time.perf_counter()
        result = self._predict(frame, conf)
        ms = (time.perf_counter() - t0) * 1000
        dets = [
            {
                "label": self.names[int(b.cls)],
                "confidence": round(float(b.conf), 4),
                "box": [round(v, 1) for v in b.xyxy[0].tolist()],
            }
            for b in result.boxes
        ]
        h, w = frame.shape[:2]
        return {
            "width": w,
            "height": h,
            "count": len(dets),
            "summary": dict(Counter(d["label"] for d in dets)),
            "inference_ms": round(ms, 2),
            "detections": dets,
        }

    def annotate(self, frame: np.ndarray, conf: float | None = None) -> np.ndarray:
        conf = conf if conf is not None else settings.conf_threshold
        return self._predict(frame, conf).plot()

    def process_video(
        self, src: str, dst: str, on_progress: Callable[[float], None] | None = None
    ) -> dict[str, int]:
        """Annotate a video with ByteTrack IDs; return unique object count per class."""
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        seen: dict[str, set[int]] = {}
        i = 0
        with self._video_lock:
            self.model.predictor = None  # reset tracker state between videos
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    res = self.model.track(
                        frame, conf=settings.conf_threshold, persist=True,
                        tracker="bytetrack.yaml", verbose=False,
                    )[0]
                    if res.boxes.id is not None:
                        for cls, tid in zip(res.boxes.cls.tolist(), res.boxes.id.tolist()):
                            seen.setdefault(self.names[int(cls)], set()).add(int(tid))
                    out.write(res.plot())
                    i += 1
                    if on_progress and i % 5 == 0:
                        on_progress(min(i / total, 0.99))
            finally:
                cap.release()
                out.release()
        return {label: len(ids) for label, ids in seen.items()}
