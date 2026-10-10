"""YOLOv8 acne lesion detector (ACNE04 fine-tune).

Runs once on the full face (BGR). Region crops alone kill recall — ACNE04
was trained on full-face photos, not grey-masked cheek patches.
Detections are then assigned to landmark∩skin regions by box center.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.services.face_regions import RegionCrop
from app.utils.image import encode_image_base64

ACNE_REGION_NAMES = ("forehead", "left_cheek", "right_cheek", "chin", "nose")


def _level_label(score: float) -> str:
    if score <= 0.5:
        return "none"
    if score <= 1.5:
        return "mild"
    if score <= 2.5:
        return "moderate"
    if score <= 3.5:
        return "moderately severe"
    return "severe"


def lesion_count_to_score(count: int) -> float:
    """Map lesion count → 0–4 severity (heuristic placeholder)."""
    if count <= 0:
        return 0.0
    if count <= 2:
        return 1.0
    if count <= 5:
        return 2.0
    if count <= 10:
        return 3.0
    return 4.0


@dataclass
class AcneDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    region: str


@dataclass
class RegionAcneResult:
    name: str
    lesion_count: int
    severity_score: float
    severity_level: str
    detections: list[AcneDetection] = field(default_factory=list)


@dataclass
class AcneAnalysisResult:
    overall_lesion_count: int
    overall_severity_score: float
    overall_severity_level: str
    regions: list[RegionAcneResult]
    overlay_b64: str
    model_name: str


class AcneDetectorService:
    def __init__(self) -> None:
        settings = get_settings()
        model_path = Path(settings.acne_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Acne YOLO model not found at {model_path}. "
                "Place acne_yolov8n.pt under backend/models/."
            )
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self.conf = settings.acne_confidence
        self.iou = settings.acne_iou
        self.imgsz = settings.acne_imgsz
        self.model_name = model_path.name

    def _face_roi(
        self, image_rgb: np.ndarray, regions: list[RegionCrop]
    ) -> tuple[np.ndarray, int, int]:
        """Tight padded crop around union of acne regions (falls back to full)."""
        h, w = image_rgb.shape[:2]
        xs: list[int] = []
        ys: list[int] = []
        for region in regions:
            if region.name not in ACNE_REGION_NAMES or region.mask is None:
                continue
            ys_idx, xs_idx = np.where(region.mask > 0)
            if len(xs_idx) == 0:
                continue
            xs.extend([int(xs_idx.min()), int(xs_idx.max())])
            ys.extend([int(ys_idx.min()), int(ys_idx.max())])

        if not xs:
            return image_rgb, 0, 0

        pad = int(0.06 * max(h, w))
        x0 = max(0, min(xs) - pad)
        y0 = max(0, min(ys) - pad)
        x1 = min(w, max(xs) + pad)
        y1 = min(h, max(ys) + pad)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return image_rgb, 0, 0
        return image_rgb[y0:y1, x0:x1], x0, y0

    def _assign_region(
        self,
        cx: int,
        cy: int,
        regions: list[RegionCrop],
    ) -> str | None:
        for region in regions:
            if region.name not in ACNE_REGION_NAMES or region.mask is None:
                continue
            mask = region.mask
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
                return region.name
        return None

    def analyze(
        self,
        image_rgb: np.ndarray,
        regions: list[RegionCrop],
    ) -> AcneAnalysisResult:
        h, w = image_rgb.shape[:2]
        overlay = image_rgb.copy()

        roi_rgb, ox, oy = self._face_roi(image_rgb, regions)
        # Ultralytics expects BGR for ndarray inputs (OpenCV convention).
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

        preds = self._model.predict(
            source=roi_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        by_region: dict[str, list[AcneDetection]] = {
            name: [] for name in ACNE_REGION_NAMES
        }
        unassigned: list[AcneDetection] = []

        if preds:
            boxes = preds[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    gx1 = float(ox + x1)
                    gy1 = float(oy + y1)
                    gx2 = float(ox + x2)
                    gy2 = float(oy + y2)
                    cx = int((gx1 + gx2) * 0.5)
                    cy = int((gy1 + gy2) * 0.5)
                    region_name = self._assign_region(cx, cy, regions)
                    det = AcneDetection(
                        x1=gx1,
                        y1=gy1,
                        x2=gx2,
                        y2=gy2,
                        confidence=float(conf),
                        region=region_name or "face",
                    )
                    if region_name:
                        by_region[region_name].append(det)
                    else:
                        unassigned.append(det)

                    cv2.rectangle(
                        overlay,
                        (int(gx1), int(gy1)),
                        (int(gx2), int(gy2)),
                        (40, 80, 255),
                        2,
                        cv2.LINE_AA,
                    )

        region_results: list[RegionAcneResult] = []
        for name in ACNE_REGION_NAMES:
            dets = by_region[name]
            count = len(dets)
            score = lesion_count_to_score(count)
            region_results.append(
                RegionAcneResult(
                    name=name,
                    lesion_count=count,
                    severity_score=score,
                    severity_level=_level_label(score),
                    detections=dets,
                )
            )

        # Count all boxes that landed on skin regions (ignore pure background).
        total = sum(r.lesion_count for r in region_results)
        # If model fired but centers missed masks (tight masks), still credit them.
        if total == 0 and unassigned:
            total = len(unassigned)
        overall = lesion_count_to_score(total)

        cv2.putText(
            overlay,
            f"acne lesions: {total}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 80, 255),
            2,
            cv2.LINE_AA,
        )

        return AcneAnalysisResult(
            overall_lesion_count=total,
            overall_severity_score=overall,
            overall_severity_level=_level_label(overall),
            regions=region_results,
            overlay_b64=encode_image_base64(overlay),
            model_name=self.model_name,
        )


_acne_service: AcneDetectorService | None = None


def get_acne_detector() -> AcneDetectorService:
    global _acne_service
    if _acne_service is None:
        _acne_service = AcneDetectorService()
    return _acne_service
