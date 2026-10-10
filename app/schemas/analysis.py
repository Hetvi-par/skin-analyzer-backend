from pydantic import BaseModel, Field


class LandmarkPoint(BaseModel):
    index: int
    x: float
    y: float
    z: float


class FaceBBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class RegionBBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class RegionSchema(BaseModel):
    name: str
    bbox: RegionBBox
    skin_pixel_count: int
    skin_coverage_percent: float
    preview: str
    mask_preview: str


class RegionColorSchema(BaseModel):
    name: str
    mean_l: float
    mean_a: float
    mean_b: float
    dark_pixel_percent: float
    redness_score: float = Field(..., description="0-4 redness severity")
    pigmentation_score: float = Field(..., description="0-4 pigmentation severity")
    redness_level: str
    pigmentation_level: str


class ColorMetricsSchema(BaseModel):
    overall_redness: float
    overall_pigmentation: float
    overall_redness_level: str
    overall_pigmentation_level: str
    baseline_forehead_a: float | None = None
    baseline_forehead_l: float | None = None
    mean_skin_l: float | None = None
    mean_skin_a: float | None = None
    mean_skin_b: float | None = None
    regions: list[RegionColorSchema] = Field(default_factory=list)


class AcneDetectionSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    region: str


class RegionAcneSchema(BaseModel):
    name: str
    lesion_count: int
    severity_score: float = Field(..., description="0-4 acne severity")
    severity_level: str
    detections: list[AcneDetectionSchema] = Field(default_factory=list)


class AcneMetricsSchema(BaseModel):
    overall_lesion_count: int
    overall_severity_score: float
    overall_severity_level: str
    model_name: str
    regions: list[RegionAcneSchema] = Field(default_factory=list)
    overlay: str | None = None


class RegionTextureSchema(BaseModel):
    name: str
    laplacian_variance: float
    highpass_extreme_percent: float
    edge_density_percent: float
    pore_score: float = Field(..., description="0-4 pore visibility")
    wrinkle_score: float = Field(..., description="0-4 wrinkle visibility")
    pore_level: str
    wrinkle_level: str


class TextureMetricsSchema(BaseModel):
    overall_pores: float
    overall_wrinkles: float
    overall_pores_level: str
    overall_wrinkles_level: str
    method: str = "laplacian+canny"
    regions: list[RegionTextureSchema] = Field(default_factory=list)


class SkinToneSchema(BaseModel):
    median_l: float
    median_a: float
    median_b: float
    median_r: int
    median_g: int
    median_b_rgb: int
    ita_degrees: float = Field(..., description="Individual Typology Angle")
    depth: str = Field(..., description="very_light…dark from ITA")
    undertone: str = Field(..., description="warm | cool | neutral")
    fitzpatrick_estimate: str = Field(..., description="Rough I–VI estimate")
    hex_color: str
    sample_pixel_count: int
    method: str = "ita+lab_median"


class InitialAnalysisResponse(BaseModel):
    status: str = "ok"
    stage: str = "skin_tone"
    message: str = (
        "Face landmarks, skin mask, region crops, CIELAB color scores, "
        "acne lesion detection, pore/wrinkle texture, and skin tone completed."
    )
    image_width: int
    image_height: int
    num_landmarks: int
    face_bbox: FaceBBox
    landmarks: list[LandmarkPoint] = Field(default_factory=list)
    include_all_landmarks: bool = False
    skin_coverage_percent: float
    class_coverage: dict[str, float]
    regions: list[RegionSchema] = Field(default_factory=list)
    color_metrics: ColorMetricsSchema | None = None
    acne_metrics: AcneMetricsSchema | None = None
    texture_metrics: TextureMetricsSchema | None = None
    skin_tone: SkinToneSchema | None = None
    landmark_preview: str
    skin_mask_preview: str
    regions_overlay: str | None = None
