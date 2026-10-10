from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import get_settings
from app.schemas.analysis import (
    AcneDetectionSchema,
    AcneMetricsSchema,
    ColorMetricsSchema,
    FaceBBox,
    InitialAnalysisResponse,
    LandmarkPoint,
    RegionAcneSchema,
    RegionBBox,
    RegionColorSchema,
    RegionSchema,
    RegionTextureSchema,
    SkinToneSchema,
    TextureMetricsSchema,
)
from app.services.pipeline import get_pipeline
from app.utils.image import ALLOWED_CONTENT_TYPES, decode_image_bytes

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/")
async def api_root() -> dict[str, str]:
    return {
        "message": "Welcome to the Skin Analysis API",
        "docs": "/docs",
        "analyze": f"{settings.api_prefix}/analyze/initial",
    }


@router.post("/analyze/initial", response_model=InitialAnalysisResponse)
async def analyze_initial(
    file: UploadFile = File(..., description="Face photo (jpg/png/webp)"),
    include_landmarks: bool = Query(
        False,
        description="If true, return all landmark points in the JSON response.",
    ),
) -> InitialAnalysisResponse:
    """
    Pipeline:
    1. MediaPipe face landmarks
    2. ResNet18 ONNX face-parsing / skin mask
    3. Region crops (forehead, cheeks, nose, chin)
    4. CIELAB redness + pigmentation scores (0-4)
    5. YOLOv8 acne lesion detection on face ROI
    6. Pore / wrinkle texture heuristics (Laplacian + Canny)
    7. Skin tone / undertone (ITA + Lab median)
    """
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Use jpeg, png, or webp.",
        )

    data = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {settings.max_upload_size_mb} MB.",
        )

    try:
        image_rgb = decode_image_bytes(data)
        result = get_pipeline().run_initial(image_rgb)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    skin_pixels = int((result.masking.skin_mask > 0).sum())
    total_pixels = int(result.masking.skin_mask.size)
    skin_coverage = round(skin_pixels / total_pixels * 100.0, 3)

    landmarks_payload: list[LandmarkPoint] = []
    if include_landmarks:
        landmarks_payload = [
            LandmarkPoint(**point) for point in result.landmarks.landmarks
        ]

    regions = [
        RegionSchema(
            name=region.name,
            bbox=RegionBBox(**region.bbox),
            skin_pixel_count=region.skin_pixel_count,
            skin_coverage_percent=region.skin_coverage_percent,
            preview=region.preview_b64,
            mask_preview=region.mask_preview_b64,
        )
        for region in result.regions.regions
    ]

    color = result.color
    color_metrics = ColorMetricsSchema(
        overall_redness=color.overall_redness,
        overall_pigmentation=color.overall_pigmentation,
        overall_redness_level=color.overall_redness_level,
        overall_pigmentation_level=color.overall_pigmentation_level,
        baseline_forehead_a=color.baseline_forehead_a,
        baseline_forehead_l=color.baseline_forehead_l,
        mean_skin_l=color.mean_skin_l,
        mean_skin_a=color.mean_skin_a,
        mean_skin_b=color.mean_skin_b,
        regions=[
            RegionColorSchema(
                name=item.name,
                mean_l=item.mean_l,
                mean_a=item.mean_a,
                mean_b=item.mean_b,
                dark_pixel_percent=item.dark_pixel_percent,
                redness_score=item.redness_score,
                pigmentation_score=item.pigmentation_score,
                redness_level=item.redness_level,
                pigmentation_level=item.pigmentation_level,
            )
            for item in color.regions
        ],
    )

    acne = result.acne
    acne_metrics = AcneMetricsSchema(
        overall_lesion_count=acne.overall_lesion_count,
        overall_severity_score=acne.overall_severity_score,
        overall_severity_level=acne.overall_severity_level,
        model_name=acne.model_name,
        overlay=acne.overlay_b64,
        regions=[
            RegionAcneSchema(
                name=item.name,
                lesion_count=item.lesion_count,
                severity_score=item.severity_score,
                severity_level=item.severity_level,
                detections=[
                    AcneDetectionSchema(
                        x1=d.x1,
                        y1=d.y1,
                        x2=d.x2,
                        y2=d.y2,
                        confidence=d.confidence,
                        region=d.region,
                    )
                    for d in item.detections
                ],
            )
            for item in acne.regions
        ],
    )

    texture = result.texture
    texture_metrics = TextureMetricsSchema(
        overall_pores=texture.overall_pores,
        overall_wrinkles=texture.overall_wrinkles,
        overall_pores_level=texture.overall_pores_level,
        overall_wrinkles_level=texture.overall_wrinkles_level,
        method=texture.method,
        regions=[
            RegionTextureSchema(
                name=item.name,
                laplacian_variance=item.laplacian_variance,
                highpass_extreme_percent=item.highpass_extreme_percent,
                edge_density_percent=item.edge_density_percent,
                pore_score=item.pore_score,
                wrinkle_score=item.wrinkle_score,
                pore_level=item.pore_level,
                wrinkle_level=item.wrinkle_level,
            )
            for item in texture.regions
        ],
    )

    tone = result.skin_tone
    skin_tone = SkinToneSchema(
        median_l=tone.median_l,
        median_a=tone.median_a,
        median_b=tone.median_b,
        median_r=tone.median_r,
        median_g=tone.median_g,
        median_b_rgb=tone.median_b_rgb,
        ita_degrees=tone.ita_degrees,
        depth=tone.depth,
        undertone=tone.undertone,
        fitzpatrick_estimate=tone.fitzpatrick_estimate,
        hex_color=tone.hex_color,
        sample_pixel_count=tone.sample_pixel_count,
        method=tone.method,
    )

    return InitialAnalysisResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        num_landmarks=result.landmarks.num_landmarks,
        face_bbox=FaceBBox(**result.landmarks.face_bbox),
        landmarks=landmarks_payload,
        include_all_landmarks=include_landmarks,
        skin_coverage_percent=skin_coverage,
        class_coverage=result.masking.class_coverage,
        regions=regions,
        color_metrics=color_metrics,
        acne_metrics=acne_metrics,
        texture_metrics=texture_metrics,
        skin_tone=skin_tone,
        landmark_preview=result.landmark_preview_b64,
        skin_mask_preview=result.skin_mask_preview_b64,
        regions_overlay=result.regions.regions_overlay_b64,
    )
