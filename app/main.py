import os
import uuid
from functools import lru_cache

import cv2
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import DetectResponse, JobStatus

app = FastAPI(title="SightLine", version="1.0.0",
              description="Object detection API for images and video (YOLOv8)")

os.makedirs(settings.work_dir, exist_ok=True)
JOBS: dict[str, dict] = {}  # swap for Redis/DB in production


@lru_cache
def get_detector():
    from .detector import Detector  # lazy import: keeps startup/tests fast
    return Detector()


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_name}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), conf: float | None = None,
                 det=Depends(get_detector)):
    return det.detect(await read_image(file), conf)


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...), conf: float | None = None,
                           det=Depends(get_detector)):
    out = det.annotate(await read_image(file), conf)
    ok, buf = cv2.imencode(".jpg", out)
    return Response(buf.tobytes(), media_type="image/jpeg")


def _run_video_job(job_id: str, src: str, dst: str, det):
    job = JOBS[job_id]
    job["status"] = "processing"
    try:
        unique = det.process_video(src, dst, lambda p: job.update(progress=p))
        job.update(status="done", progress=1.0, output=dst, unique_objects=unique)
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", error=str(e))
    finally:
        if os.path.exists(src):
            os.remove(src)


@app.post("/video", response_model=JobStatus, status_code=202)
async def submit_video(background: BackgroundTasks, file: UploadFile = File(...),
                       det=Depends(get_detector)):
    job_id = uuid.uuid4().hex[:12]
    src = os.path.join(settings.work_dir, f"{job_id}_in")
    dst = os.path.join(settings.work_dir, f"{job_id}_out.mp4")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Video too large (max {settings.max_upload_mb} MB)")
    with open(src, "wb") as f:
        f.write(data)
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0, "error": None}
    background.add_task(_run_video_job, job_id, src, dst, det)
    return JOBS[job_id]


@app.get("/video/{job_id}", response_model=JobStatus)
def video_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.get("/video/{job_id}/result")
def video_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not ready")
    return FileResponse(job["output"], media_type="video/mp4", filename="annotated.mp4")


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"),
                           html=True), name="static")
