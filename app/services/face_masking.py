"""ResNet18 ONNX face-parsing / skin-masking service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import get_settings

# CelebAMask-HQ / BiSeNet-style 19-class face parsing labels
FACE_PARSING_LABELS: dict[int, str] = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}

# Facial skin for analysis: skin + nose + neck (chin often labeled neck)
SKIN_CLASS_IDS = (1, 10, 14)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FaceMaskResult:
    parsing_map: np.ndarray  # HxW int labels in original image size
    skin_mask: np.ndarray  # HxW uint8 binary mask
    class_coverage: dict[str, float]
    model_input_size: int


class FaceMaskingService:
    """Run ResNet18 ONNX face parsing and produce a skin mask."""

    def __init__(self) -> None:
        settings = get_settings()
        model_path = Path(settings.face_parsing_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face parsing ONNX model not found at {model_path}"
            )

        self.input_size = settings.face_parsing_input_size
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        original_hw = (image_rgb.shape[0], image_rgb.shape[1])
        resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # 1x3xHxW
        return tensor.astype(np.float32), original_hw

    def predict(self, image_rgb: np.ndarray) -> FaceMaskResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3)")

        tensor, (orig_h, orig_w) = self._preprocess(image_rgb)
        outputs = self._session.run(None, {self._input_name: tensor})

        # Model returns 3 multi-scale logits; use the primary output.
        logits = outputs[0]  # 1x19x512x512
        class_map_small = np.argmax(logits[0], axis=0).astype(np.uint8)

        parsing_map = cv2.resize(
            class_map_small,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        skin_mask = np.isin(parsing_map, SKIN_CLASS_IDS).astype(np.uint8) * 255

        total_pixels = float(parsing_map.size)
        class_coverage: dict[str, float] = {}
        for class_id, label in FACE_PARSING_LABELS.items():
            count = int(np.count_nonzero(parsing_map == class_id))
            if count > 0:
                class_coverage[label] = round(count / total_pixels * 100.0, 3)

        return FaceMaskResult(
            parsing_map=parsing_map,
            skin_mask=skin_mask,
            class_coverage=class_coverage,
            model_input_size=self.input_size,
        )


_face_masking_service: FaceMaskingService | None = None


def get_face_masking_service() -> FaceMaskingService:
    global _face_masking_service
    if _face_masking_service is None:
        _face_masking_service = FaceMaskingService()
    return _face_masking_service
