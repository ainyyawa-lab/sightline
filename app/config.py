from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "yolov8n.pt"      # n/s/m/l/x: speed vs accuracy
    conf_threshold: float = 0.35
    max_upload_mb: int = 20
    work_dir: str = "/tmp/sightline"

    class Config:
        env_prefix = "SIGHTLINE_"
        protected_namespaces = ("settings_",)


settings = Settings()
