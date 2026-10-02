"""Tests run with a fake detector, so no model download or GPU is needed."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app, get_detector


class FakeDetector:
    def detect(self, frame, conf=None):
        h, w = frame.shape[:2]
        return {"width": w, "height": h, "count": 1, "summary": {"person": 1},
                "inference_ms": 1.0,
                "detections": [{"label": "person", "confidence": 0.9, "box": [0, 0, 10, 10]}]}

    def annotate(self, frame, conf=None):
        return frame

    def process_video(self, src, dst, on_progress=None):
        open(dst, "wb").write(b"fake")
        return {"person": 2, "car": 1}


app.dependency_overrides[get_detector] = lambda: FakeDetector()
client = TestClient(app)


def _png():
    ok, buf = cv2.imencode(".png", np.zeros((32, 48, 3), np.uint8))
    return buf.tobytes()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_detect():
    r = client.post("/detect", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 48 and body["summary"] == {"person": 1}


def test_invalid_image():
    r = client.post("/detect", files={"file": ("a.txt", b"nope", "text/plain")})
    assert r.status_code == 400


def test_unknown_job():
    assert client.get("/video/doesnotexist").status_code == 404


def test_video_job_reports_unique_objects():
    r = client.post("/video", files={"file": ("v.mp4", b"data", "video/mp4")})
    assert r.status_code == 202
    job = client.get(f"/video/{r.json()['job_id']}").json()  # background task ran
    assert job["status"] == "done"
    assert job["unique_objects"] == {"person": 2, "car": 1}
