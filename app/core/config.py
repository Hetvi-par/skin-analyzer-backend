from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Skin Analysis API"
    app_version: str = "0.1.0"
    debug: bool = True
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 10

    # Models
    face_landmarker_model: str = str(
        BACKEND_ROOT / "models" / "face_landmarker.task"
    )
    face_parsing_model: str = str(BACKEND_ROOT / "models" / "resnet18.onnx")
    face_parsing_input_size: int = 512
    face_detection_confidence: float = 0.5
    face_presence_confidence: float = 0.5
    face_tracking_confidence: float = 0.5

    # Acne YOLO (ACNE04 fine-tune — academic use only)
    acne_model: str = str(BACKEND_ROOT / "models" / "acne_yolov8n.pt")
    acne_confidence: float = 0.15
    acne_iou: float = 0.45
    acne_imgsz: int = 640


@lru_cache
def get_settings() -> Settings:
    return Settings()
