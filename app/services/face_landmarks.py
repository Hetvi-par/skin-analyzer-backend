"""MediaPipe Face Landmarker service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from app.core.config import get_settings


@dataclass
class FaceLandmarkResult:
    landmarks: list[dict[str, float]]
    face_bbox: dict[str, float]
    num_landmarks: int
    detection_score: float | None = None


class FaceLandmarkService:
    """Detect 478 face mesh landmarks using MediaPipe FaceLandmarker."""

    def __init__(self) -> None:
        settings = get_settings()
        model_path = Path(settings.face_landmarker_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=settings.face_detection_confidence,
            min_face_presence_confidence=settings.face_presence_confidence,
            min_tracking_confidence=settings.face_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> FaceLandmarkResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3)")

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError("No face detected. Please upload a clear face photo.")

        face = result.face_landmarks[0]
        landmarks = [
            {
                "index": i,
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
            }
            for i, lm in enumerate(face)
        ]

        xs = [p["x"] for p in landmarks]
        ys = [p["y"] for p in landmarks]
        face_bbox = {
            "x_min": float(min(xs)),
            "y_min": float(min(ys)),
            "x_max": float(max(xs)),
            "y_max": float(max(ys)),
            "width": float(max(xs) - min(xs)),
            "height": float(max(ys) - min(ys)),
        }

        return FaceLandmarkResult(
            landmarks=landmarks,
            face_bbox=face_bbox,
            num_landmarks=len(landmarks),
        )

    def close(self) -> None:
        self._landmarker.close()


_face_landmark_service: FaceLandmarkService | None = None


def get_face_landmark_service() -> FaceLandmarkService:
    global _face_landmark_service
    if _face_landmark_service is None:
        _face_landmark_service = FaceLandmarkService()
    return _face_landmark_service
