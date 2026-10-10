"""Skin depth + undertone from masked skin Lab / RGB medians.

Uses Individual Typology Angle (ITA) for depth buckets and a*/b* balance
for warm / cool / neutral undertone. Prefer forehead + cheeks (less lip/nose bias).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.face_regions import RegionCrop

# Stable sampling regions for global tone (avoid lips / extreme nose oil).
TONE_REGIONS = frozenset({"forehead", "left_cheek", "right_cheek"})


@dataclass
class SkinToneResult:
    median_l: float
    median_a: float
    median_b: float
    median_r: int
    median_g: int
    median_b_rgb: int
    ita_degrees: float
    depth: str
    undertone: str
    fitzpatrick_estimate: str
    hex_color: str
    sample_pixel_count: int
    method: str = "ita+lab_median"


def _depth_from_ita(ita: float) -> str:
    # Del Bino / COLIPA-style ITA bands
    if ita > 55:
        return "very_light"
    if ita > 41:
        return "light"
    if ita > 28:
        return "intermediate"
    if ita > 10:
        return "tan"
    if ita > -30:
        return "brown"
    return "dark"


def _fitzpatrick_estimate(ita: float) -> str:
    """Rough ITA → Fitzpatrick I–VI mapping (heuristic, not clinical)."""
    if ita > 55:
        return "I"
    if ita > 41:
        return "II"
    if ita > 28:
        return "III"
    if ita > 10:
        return "IV"
    if ita > -30:
        return "V"
    return "VI"


def _undertone(median_a: float, median_b: float) -> str:
    """
    Warm: yellower (higher b*).
    Cool: redder / less yellow relative to a*.
    Neutral: close balance.
    """
    delta = median_b - median_a
    if delta >= 3.0:
        return "warm"
    if delta <= -1.5:
        return "cool"
    return "neutral"


def _build_sample_mask(
    full_skin_mask: np.ndarray,
    regions: list[RegionCrop] | None,
) -> np.ndarray:
    if regions:
        combined = np.zeros_like(full_skin_mask)
        used = False
        for region in regions:
            if region.name not in TONE_REGIONS or region.mask is None:
                continue
            combined = np.maximum(combined, region.mask)
            used = True
        if used and int((combined > 0).sum()) >= 200:
            return combined
    return full_skin_mask


def analyze_skin_tone(
    image_rgb: np.ndarray,
    full_skin_mask: np.ndarray,
    regions: list[RegionCrop] | None = None,
) -> SkinToneResult:
    mask = _build_sample_mask(full_skin_mask, regions)
    sel = mask > 0
    n = int(sel.sum())
    if n < 80:
        raise ValueError("Not enough skin pixels to estimate skin tone.")

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)[sel] * 100.0 / 255.0
    a = lab[:, :, 1].astype(np.float32)[sel] - 128.0
    b = lab[:, :, 2].astype(np.float32)[sel] - 128.0

    r = image_rgb[:, :, 0][sel].astype(np.float32)
    g = image_rgb[:, :, 1][sel].astype(np.float32)
    blue = image_rgb[:, :, 2][sel].astype(np.float32)

    med_l = float(np.median(L))
    med_a = float(np.median(a))
    med_b = float(np.median(b))
    med_r = int(np.clip(round(float(np.median(r))), 0, 255))
    med_g = int(np.clip(round(float(np.median(g))), 0, 255))
    med_blue = int(np.clip(round(float(np.median(blue))), 0, 255))

    # ITA = arctan((L* - 50) / b*) in degrees; guard near-zero b*
    denom = med_b if abs(med_b) > 0.35 else (0.35 if med_b >= 0 else -0.35)
    ita = float(np.degrees(np.arctan2(med_l - 50.0, denom)))

    return SkinToneResult(
        median_l=round(med_l, 2),
        median_a=round(med_a, 2),
        median_b=round(med_b, 2),
        median_r=med_r,
        median_g=med_g,
        median_b_rgb=med_blue,
        ita_degrees=round(ita, 2),
        depth=_depth_from_ita(ita),
        undertone=_undertone(med_a, med_b),
        fitzpatrick_estimate=_fitzpatrick_estimate(ita),
        hex_color=f"#{med_r:02x}{med_g:02x}{med_blue:02x}",
        sample_pixel_count=n,
    )
