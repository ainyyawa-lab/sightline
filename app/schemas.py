from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] in pixels


class DetectResponse(BaseModel):
    width: int
    height: int
    count: int
    summary: dict[str, int]
    inference_ms: float
    detections: list[Detection]


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | done | failed
    progress: float = 0.0
    error: str | None = None
    unique_objects: dict[str, int] | None = None  # distinct tracked objects per class
