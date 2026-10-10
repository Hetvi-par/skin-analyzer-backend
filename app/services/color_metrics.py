"""CIELAB color heuristics for redness and pigmentation (0-4 scores)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.face_regions import RegionCrop

# OpenCV LAB: L in [0,255], a/b centered at 128.
# Converted: L* ≈ L*100/255, a* ≈ a-128, b* ≈ b-128


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _to_score_0_4(value: float, low: float, high: float) -> float:
    """Map value into 0-4 severity (higher = more of the condition)."""
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
class RegionColorMetrics:
    name: str
    mean_l: float
    mean_a: float
    mean_b: float
    dark_pixel_percent: float
    redness_score: float
    pigmentation_score: float
    redness_level: str
    pigmentation_level: str


@dataclass
class ColorAnalysisResult:
    regions: list[RegionColorMetrics]
    overall_redness: float
    overall_pigmentation: float
    overall_redness_level: str
    overall_pigmentation_level: str
    baseline_forehead_a: float | None
    baseline_forehead_l: float | None
    mean_skin_l: float | None
    mean_skin_a: float | None
    mean_skin_b: float | None


def _lab_stats(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    if mask is None or not np.any(mask > 0):
        return None

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    # Mild blur reduces noise / single-pixel spikes
    a = cv2.GaussianBlur(a, (5, 5), 0)

    sel = mask > 0
    if int(sel.sum()) < 40:
        return None

    mean_l = float(L[sel].astype(np.float32).mean()) * 100.0 / 255.0
    mean_a = float(a[sel].mean())
    mean_b = float(b[sel].mean())

    # Pigmentation proxy: morphological blackhat (locally darker than surround).
    # No CLAHE — it inflates local contrast and creates false dark spots.
    L_smooth = cv2.medianBlur(L, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(L_smooth, cv2.MORPH_BLACKHAT, kernel)

    baseline_l = float(np.median(L[sel].astype(np.float32)))
    dark_thr = int(max(12, min(20, baseline_l * 0.10)))
    dark_frac = float((blackhat[sel] > dark_thr).mean() * 100.0)

    return {
        "mean_l": round(mean_l, 2),
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "dark_pixel_percent": round(dark_frac, 2),
    }


def analyze_color_metrics(
    image_rgb: np.ndarray,
    regions: list[RegionCrop],
    full_skin_mask: np.ndarray | None = None,
) -> ColorAnalysisResult:
    """
    Per-region CIELAB stats + 0-4 redness/pigmentation scores.

    Redness: cheek a* relative to forehead baseline when available,
             otherwise absolute mean a*.
    Pigmentation: percentage of locally-dark skin pixels.
    """
    by_name = {r.name: r for r in regions if r.mask is not None}

    forehead_stats = None
    if "forehead" in by_name:
        forehead_stats = _lab_stats(image_rgb, by_name["forehead"].mask)

    baseline_a = forehead_stats["mean_a"] if forehead_stats else None
    baseline_l = forehead_stats["mean_l"] if forehead_stats else None

    region_metrics: list[RegionColorMetrics] = []
    cheek_redness: list[float] = []
    cheek_pigment: list[float] = []
    all_redness: list[float] = []
    all_pigment: list[float] = []

    for region in regions:
        if region.mask is None:
            continue
        stats = _lab_stats(image_rgb, region.mask)
        if stats is None:
            continue

        mean_a = stats["mean_a"]
        dark_pct = stats["dark_pixel_percent"]

        if baseline_a is not None and region.name != "forehead":
            # Tone-aware: redness = how much redder than forehead
            delta_a = mean_a - baseline_a
            redness = _to_score_0_4(delta_a, -1.0, 10.0)
        else:
            # Absolute a* fallback (typical skin a* roughly 5-25)
            redness = _to_score_0_4(mean_a, 4.0, 22.0)

        # Typical blackhat dark-spot fraction without CLAHE: ~0.5–20%
        pigmentation = _to_score_0_4(dark_pct, 1.0, 20.0)

        item = RegionColorMetrics(
            name=region.name,
            mean_l=stats["mean_l"],
            mean_a=mean_a,
            mean_b=stats["mean_b"],
            dark_pixel_percent=dark_pct,
            redness_score=redness,
            pigmentation_score=pigmentation,
            redness_level=_level(redness),
            pigmentation_level=_level(pigmentation),
        )
        region_metrics.append(item)
        all_redness.append(redness)
        all_pigment.append(pigmentation)
        if region.name in ("left_cheek", "right_cheek"):
            cheek_redness.append(redness)
            cheek_pigment.append(pigmentation)

    # Overall: prefer cheeks (most clinically relevant for redness)
    if cheek_redness:
        overall_r = round(float(np.mean(cheek_redness)), 2)
        overall_p = round(float(np.mean(cheek_pigment)), 2)
    elif all_redness:
        overall_r = round(float(np.mean(all_redness)), 2)
        overall_p = round(float(np.mean(all_pigment)), 2)
    else:
        overall_r, overall_p = 0.0, 0.0

    skin_stats = None
    if full_skin_mask is not None:
        skin_stats = _lab_stats(image_rgb, full_skin_mask)

    return ColorAnalysisResult(
        regions=region_metrics,
        overall_redness=overall_r,
        overall_pigmentation=overall_p,
        overall_redness_level=_level(overall_r),
        overall_pigmentation_level=_level(overall_p),
        baseline_forehead_a=baseline_a,
        baseline_forehead_l=baseline_l,
        mean_skin_l=skin_stats["mean_l"] if skin_stats else None,
        mean_skin_a=skin_stats["mean_a"] if skin_stats else None,
        mean_skin_b=skin_stats["mean_b"] if skin_stats else None,
    )
