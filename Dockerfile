FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp SIGHTLINE_WORK_DIR=/tmp/sightline
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image so the first request is fast
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY --chown=user app app
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
