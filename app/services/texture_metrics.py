"""Heuristic pore / wrinkle texture scores (0-4) from region crops.

Pores: Laplacian variance + high-pass extremes on cheeks/nose.
Wrinkles: horizontal edge density (Sobel-Y / Canny) on forehead.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.face_regions import RegionCrop

PORE_REGIONS = frozenset({"left_cheek", "right_cheek", "nose"})
WRINKLE_REGIONS = frozenset({"forehead", "chin"})


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _to_score_0_4(value: float, low: float, high: float) -> float:
    if high == low:
        return 2.0
    t = _clip01((value - low) / (high - low))
    return round(t * 4.0, 2)


def _level(score: float) -> str:
    if score < 1.0:
        return "low"
    if score < 2.0:
        return "mild"
    if score < 3.0:
        return "moderate"
    return "high"


@dataclass
class RegionTextureMetrics:
    name: str
    laplacian_variance: float
    highpass_extreme_percent: float
    edge_density_percent: float
    pore_score: float
    wrinkle_score: float
    pore_level: str
    wrinkle_level: str


@dataclass
class TextureAnalysisResult:
    regions: list[RegionTextureMetrics]
    overall_pores: float
    overall_wrinkles: float
    overall_pores_level: str
    overall_wrinkles_level: str
    method: str = "laplacian+canny"


def _region_gray_mask(
    image_rgb: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if mask is None or not np.any(mask > 0):
        return None
    sel = mask > 0
    if int(sel.sum()) < 80:
        return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray, sel


def _texture_stats(
    image_rgb: np.ndarray, mask: np.ndarray
) -> dict[str, float] | None:
    packed = _region_gray_mask(image_rgb, mask)
    if packed is None:
        return None
    gray, sel = packed

    # Mild denoise so single-pixel sensor noise does not dominate pores.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
    lap_var = float(np.var(lap[sel]))

    # High-pass residual: fine detail vs local average.
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5)
    high = cv2.absdiff(gray.astype(np.float32), low.astype(np.float32))
    high_vals = high[sel]
    # Threshold relative to region: strong local peaks ≈ pores / bumps.
    thr = float(np.percentile(high_vals, 75) + 8.0)
    extreme_pct = float((high_vals > thr).mean() * 100.0)

    # Horizontal wrinkles: Sobel-Y magnitude + Canny edges inside mask.
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.abs(sobel_y)
    mag_norm = mag / (float(np.percentile(mag[sel], 95)) + 1e-6)
    strong_y = (mag_norm[sel] > 0.45).mean() * 100.0

    edges = cv2.Canny(blur, 40, 120)
    edge_pct = float((edges[sel] > 0).mean() * 100.0)
    # Blend: horizontal emphasis + general edge density.
    edge_density = 0.65 * float(strong_y) + 0.35 * edge_pct

    return {
        "laplacian_variance": round(lap_var, 2),
        "highpass_extreme_percent": round(extreme_pct, 2),
        "edge_density_percent": round(edge_density, 2),
    }


def analyze_texture_metrics(
    image_rgb: np.ndarray,
    regions: list[RegionCrop],
) -> TextureAnalysisResult:
    """
    Per-region texture heuristics → 0–4 pore / wrinkle visibility scores.

    Pores emphasize cheeks + nose (Laplacian / high-pass).
    Wrinkles emphasize forehead (+ chin) via horizontal edge density.
    """
    region_metrics: list[RegionTextureMetrics] = []
    pore_scores: list[float] = []
    wrinkle_scores: list[float] = []

    for region in regions:
        if region.mask is None:
            continue
        stats = _texture_stats(image_rgb, region.mask)
        if stats is None:
            continue

        lap_var = stats["laplacian_variance"]
        extreme = stats["highpass_extreme_percent"]
        edge_pct = stats["edge_density_percent"]

        # Typical selfie skin: lap variance ~20–400; extreme ~5–35%.
        pore_from_lap = _to_score_0_4(lap_var, 30.0, 350.0)
        pore_from_hp = _to_score_0_4(extreme, 8.0, 32.0)
        pore = round(0.55 * pore_from_lap + 0.45 * pore_from_hp, 2)

        # Forehead edge density often ~2–25% when lines are visible.
        wrinkle = _to_score_0_4(edge_pct, 3.0, 22.0)

        # Down-weight pore score outside cheek/nose; wrinkles outside forehead/chin.
        if region.name not in PORE_REGIONS:
            pore = round(pore * 0.45, 2)
        if region.name not in WRINKLE_REGIONS:
            wrinkle = round(wrinkle * 0.4, 2)

        item = RegionTextureMetrics(
            name=region.name,
            laplacian_variance=lap_var,
            highpass_extreme_percent=extreme,
            edge_density_percent=edge_pct,
            pore_score=pore,
            wrinkle_score=wrinkle,
            pore_level=_level(pore),
            wrinkle_level=_level(wrinkle),
        )
        region_metrics.append(item)

        if region.name in PORE_REGIONS:
            pore_scores.append(pore)
        if region.name in WRINKLE_REGIONS:
            wrinkle_scores.append(wrinkle)

    overall_p = (
        round(float(np.mean(pore_scores)), 2) if pore_scores else 0.0
    )
    overall_w = (
        round(float(np.mean(wrinkle_scores)), 2) if wrinkle_scores else 0.0
    )

    return TextureAnalysisResult(
        regions=region_metrics,
        overall_pores=overall_p,
        overall_wrinkles=overall_w,
        overall_pores_level=_level(overall_p),
        overall_wrinkles_level=_level(overall_w),
    )
