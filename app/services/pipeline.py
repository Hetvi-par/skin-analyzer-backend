"""Skin-analysis pipeline: landmarks + mask + regions + color + acne + texture + tone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.acne_detector import AcneAnalysisResult, get_acne_detector
from app.services.color_metrics import ColorAnalysisResult, analyze_color_metrics
from app.services.face_landmarks import FaceLandmarkResult, get_face_landmark_service
from app.services.face_masking import FaceMaskResult, get_face_masking_service
from app.services.face_regions import FaceRegionsResult, extract_face_regions
from app.services.skin_tone import SkinToneResult, analyze_skin_tone
from app.services.texture_metrics import TextureAnalysisResult, analyze_texture_metrics
from app.utils.image import draw_landmarks, encode_image_base64, overlay_mask


@dataclass
class InitialAnalysisResult:
    landmarks: FaceLandmarkResult
    masking: FaceMaskResult
    regions: FaceRegionsResult
    color: ColorAnalysisResult
    acne: AcneAnalysisResult
    texture: TextureAnalysisResult
    skin_tone: SkinToneResult
    landmark_preview_b64: str
    skin_mask_preview_b64: str
    image_width: int
    image_height: int


class SkinAnalysisPipeline:
    def __init__(self) -> None:
        self.landmarks = get_face_landmark_service()
        self.masking = get_face_masking_service()
        self.acne = get_acne_detector()

    def run_initial(self, image_rgb: np.ndarray) -> InitialAnalysisResult:
        landmark_result = self.landmarks.detect(image_rgb)
        mask_result = self.masking.predict(image_rgb)
        regions_result = extract_face_regions(
            image_rgb=image_rgb,
            landmarks=landmark_result.landmarks,
            skin_mask=mask_result.skin_mask,
            encode_image_base64=encode_image_base64,
            overlay_mask=overlay_mask,
            parsing_map=mask_result.parsing_map,
        )
        color_result = analyze_color_metrics(
            image_rgb=image_rgb,
            regions=regions_result.regions,
            full_skin_mask=mask_result.skin_mask,
        )
        acne_result = self.acne.analyze(
            image_rgb=image_rgb,
            regions=regions_result.regions,
        )
        texture_result = analyze_texture_metrics(
            image_rgb=image_rgb,
            regions=regions_result.regions,
        )
        tone_result = analyze_skin_tone(
            image_rgb=image_rgb,
            full_skin_mask=mask_result.skin_mask,
            regions=regions_result.regions,
        )

        landmark_preview = draw_landmarks(image_rgb, landmark_result.landmarks)
        skin_preview = overlay_mask(image_rgb, mask_result.skin_mask)

        return InitialAnalysisResult(
            landmarks=landmark_result,
            masking=mask_result,
            regions=regions_result,
            color=color_result,
            acne=acne_result,
            texture=texture_result,
            skin_tone=tone_result,
            landmark_preview_b64=encode_image_base64(landmark_preview),
            skin_mask_preview_b64=encode_image_base64(skin_preview),
            image_width=image_rgb.shape[1],
            image_height=image_rgb.shape[0],
        )


_pipeline: SkinAnalysisPipeline | None = None


def get_pipeline() -> SkinAnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SkinAnalysisPipeline()
    return _pipeline
