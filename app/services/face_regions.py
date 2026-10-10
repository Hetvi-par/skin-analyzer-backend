"""Exact facial skin regions via MediaPipe geometry ∩ ResNet18 parsing mask.

Same approach as virtual-tryon / research doc: landmark shapes define anatomy,
face-parsing keeps only real skin (and nose class for the nose region).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# CelebAMask class id for nose (from face_masking.FACE_PARSING_LABELS)
PARSING_NOSE_ID = 10

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

HAIRLINE = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
BROW_CLOSE = [300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70]
FOREHEAD = HAIRLINE + BROW_CLOSE

# Nose landmarks for convex hull (bridge + tip + both alae).
# Parser "nose" class is tip-heavy; sides are often labeled skin — hull ∩ skin fixes that.
NOSE = [
    8, 9, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
    45, 51, 44, 220, 115, 48, 64, 98, 97,
    275, 281, 274, 440, 344, 278, 294, 327, 326,
    236, 3, 196, 419, 248, 456,
]

# Cheek mid-face hull (under-eye ribbon comes from lower-lid arcs)
LEFT_CHEEK = [
    129, 203, 205, 207, 187, 147, 123, 116, 143, 227, 234,
    137, 177, 215, 138, 135, 214, 192, 213, 212, 216, 206,
]
RIGHT_CHEEK = [
    358, 423, 425, 427, 411, 376, 352, 345, 372, 447, 454,
    366, 401, 435, 367, 364, 434, 416, 433, 432, 436, 426,
]

# Lower eyelid arcs (outer → inner) — cheek top should sit just under these
LOWER_LID_L = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LOWER_LID_R = [263, 249, 390, 373, 374, 380, 381, 382, 362]

REGION_ORDER = ("forehead", "left_cheek", "right_cheek", "nose", "chin")

REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "forehead": (40, 160, 220),
    "left_cheek": (60, 200, 120),
    "right_cheek": (90, 190, 80),
    "nose": (220, 160, 60),
    "chin": (180, 100, 200),
}


@dataclass
class RegionCrop:
    name: str
    bbox: dict[str, int]
    skin_pixel_count: int
    skin_coverage_percent: float
    preview_b64: str
    mask_preview_b64: str
    mask: np.ndarray | None = None


@dataclass
class FaceRegionsResult:
    regions: list[RegionCrop]
    regions_overlay_b64: str


def _landmarks_to_px(
    landmarks: list[dict[str, float]],
    width: int,
    height: int,
) -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    for lm in landmarks:
        idx = int(lm["index"])
        if 0 <= idx < 478:
            pts[idx, 0] = float(lm["x"]) * width
            pts[idx, 1] = float(lm["y"]) * height
    return pts


def _pt(points: np.ndarray, idx: int) -> tuple[float, float]:
    return float(points[idx, 0]), float(points[idx, 1])


def _unique_ordered(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if 0 <= idx < 478 and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _fill_polygon(
    shape_hw: tuple[int, int],
    points_px: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    ordered = _unique_ordered(indices)
    if len(ordered) < 3:
        return mask
    poly = np.round(points_px[ordered]).astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _convex_hull_mask(
    shape_hw: tuple[int, int],
    points_px: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    idxs = [i for i in indices if 0 <= i < 478]
    if len(idxs) < 3:
        return mask
    pts = np.round(points_px[idxs]).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _expand_hairline_up(
    points: np.ndarray,
    indices: list[int],
    lift: float,
) -> np.ndarray:
    out = points.copy()
    for idx in indices:
        if 0 <= idx < 478:
            out[idx, 1] = max(0.0, float(out[idx, 1]) - lift)
    return out


def _lower_lid_y(points: np.ndarray, indices: list[int]) -> float:
    ys = [float(points[i, 1]) for i in indices if 0 <= i < 478]
    return max(ys) if ys else 0.0


def _under_eye_ribbon(
    shape_hw: tuple[int, int],
    points: np.ndarray,
    lid_indices: list[int],
    drop_px: float,
    extend_outer_px: float = 0.0,
) -> np.ndarray:
    """Filled band from lower-lid arc straight down — infraorbital shelf.

    Optionally extends past the outer canthus so the ribbon doesn't leave a
    sharp V where it meets the mid-cheek hull.
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    idxs = [i for i in lid_indices if 0 <= i < 478]
    if len(idxs) < 2:
        return mask
    top = np.round(points[idxs]).astype(np.float32)
    # Order outer → inner by distance from face midline later; keep mesh order.
    # Extend outermost point further outward + slightly down to cover temple gap.
    if extend_outer_px > 0 and len(top) >= 2:
        outer = top[0].copy()
        nxt = top[1]
        direction = outer - nxt
        norm = float(np.hypot(direction[0], direction[1])) + 1e-6
        direction = direction / norm
        tip = outer + direction * extend_outer_px
        tip[1] += drop_px * 0.15
        top = np.vstack([tip[None, :], top])
    top_i = np.round(top).astype(np.int32)
    bot = top_i.copy()
    bot[:, 1] = np.clip(bot[:, 1] + int(round(drop_px)), 0, h - 1)
    poly = np.vstack([top_i, bot[::-1]])
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _ellipse_mask(
    shape_hw: tuple[int, int],
    center: tuple[float, float],
    axes: tuple[float, float],
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(1, int(round(axes[0]))), max(1, int(round(axes[1])))),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask


def build_region_masks(
    points: np.ndarray,
    shape_hw: tuple[int, int],
    parsing_map: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Landmark anatomy masks (skin ∩ applied later for exact edges)."""
    h, w = shape_hw
    le = _pt(points, 33)
    re = _pt(points, 263)
    nose = _pt(points, 1)
    tip = _pt(points, 4)
    ml = _pt(points, 61)
    mr = _pt(points, 291)
    chin = _pt(points, 152)
    brow_l = _pt(points, 70)
    brow_r = _pt(points, 300)
    jaw_l = _pt(points, 234)
    jaw_r = _pt(points, 454)
    lower_lip = _pt(points, 17)

    eye_l = le if le[0] <= re[0] else re
    eye_r = re if le[0] <= re[0] else le
    eye_dist = max(20.0, float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])))
    face_w = max(eye_dist, float(np.hypot(jaw_r[0] - jaw_l[0], jaw_r[1] - jaw_l[1])))
    mouth_mid = ((ml[0] + mr[0]) * 0.5, (ml[1] + mr[1]) * 0.5)
    brow_y = min(brow_l[1], brow_r[1])
    nose_x = int(nose[0])
    # Image-left / image-right lower lids (mesh L/R can flip with camera view)
    if le[0] <= re[0]:
        lid_l_y = _lower_lid_y(points, LOWER_LID_L)
        lid_r_y = _lower_lid_y(points, LOWER_LID_R)
    else:
        lid_l_y = _lower_lid_y(points, LOWER_LID_R)
        lid_r_y = _lower_lid_y(points, LOWER_LID_L)

    face = _fill_polygon(shape_hw, points, FACE_OVAL)

    # --- Forehead (unchanged — already accurate) ---
    lifted = _expand_hairline_up(points, HAIRLINE, eye_dist * 0.35)
    forehead = _fill_polygon(shape_hw, lifted, FOREHEAD)
    above_brows = face.copy()
    above_brows[int(brow_y - eye_dist * 0.05) :, :] = 0
    forehead = cv2.bitwise_or(forehead, above_brows)
    forehead[int(brow_y) :, :] = 0

    # --- Nose: landmark hull (full bridge+alae); ∪ parser nose class ---
    # CelebAMask often labels nose sides as "skin", so hull ∩ skin is required.
    nose_poly = _convex_hull_mask(shape_hw, points, NOSE)
    nose_poly = cv2.dilate(
        nose_poly,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    # Include upper bridge near glabella (landmark 168 alone sits mid-bridge)
    nose_band = np.zeros((h, w), dtype=np.uint8)
    glabella = _pt(points, 9)
    ny0 = int(min(glabella[1], _pt(points, 168)[1]) - eye_dist * 0.02)
    ny1 = int(tip[1] + eye_dist * 0.10)
    nose_band[max(0, ny0) : min(h, max(ny0 + 1, ny1)), :] = 255
    nose_mask = cv2.bitwise_and(nose_poly, nose_band)
    if parsing_map is not None and parsing_map.shape[:2] == (h, w):
        parser_nose = (parsing_map == PARSING_NOSE_ID).astype(np.uint8) * 255
        envelope = cv2.dilate(
            nose_poly,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(5, int(face_w * 0.05)) | 1, max(5, int(face_w * 0.06)) | 1),
            ),
        )
        nose_mask = cv2.bitwise_or(
            nose_mask, cv2.bitwise_and(parser_nose, envelope)
        )
    nose_mask = cv2.bitwise_and(nose_mask, face)

    # --- Chin: centered oval on chin tip (landmark 152) — stays mid-face ---
    mouth_half = abs(mr[0] - ml[0]) * 0.5
    chin_cx = float(chin[0])  # tip is the true center; jaw hulls were shifting sideways
    chin_cy = 0.40 * lower_lip[1] + 0.60 * chin[1]
    chin_rx = mouth_half + eye_dist * 0.32
    chin_ry = max(12.0, abs(chin[1] - lower_lip[1]) * 0.78)
    chin_mask = _ellipse_mask(shape_hw, (chin_cx, chin_cy), (chin_rx, chin_ry))
    chin_mask[: int(lower_lip[1] + 1), :] = 0
    chin_mask = cv2.bitwise_and(chin_mask, face)

    # --- Cheeks: mid-face hull + lid-following under-eye ribbon ---
    nose_exclude = nose_mask.copy()
    mid_nose_y = int((_pt(points, 168)[1] + tip[1]) * 0.55)
    nose_exclude[:mid_nose_y, :] = 0
    nose_exclude = cv2.dilate(
        nose_exclude,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    exclude = nose_exclude.copy()
    cv2.ellipse(
        exclude,
        (int(mouth_mid[0]), int(mouth_mid[1])),
        (max(8, int(face_w * 0.22)), max(5, int(face_w * 0.10))),
        0,
        0,
        360,
        255,
        -1,
    )
    exclude = cv2.bitwise_or(exclude, chin_mask)

    hull_l = _convex_hull_mask(shape_hw, points, LEFT_CHEEK)
    hull_r = _convex_hull_mask(shape_hw, points, RIGHT_CHEEK)
    ksz = max(3, int(face_w * 0.05)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    hull_l = cv2.dilate(hull_l, k)
    hull_r = cv2.dilate(hull_r, k)

    apple_l_x = _pt(points, 205)[0]
    apple_r_x = _pt(points, 425)[0]
    if apple_l_x > apple_r_x:
        hull_l, hull_r = hull_r, hull_l

    # Lid-following ribbons (mesh L/R mapped to image L/R), extended past
    # outer canthus so they don't leave a sharp V vs the mid-cheek hull.
    if le[0] <= re[0]:
        lid_idxs_l, lid_idxs_r = LOWER_LID_L, LOWER_LID_R
    else:
        lid_idxs_l, lid_idxs_r = LOWER_LID_R, LOWER_LID_L
    drop = eye_dist * 0.48
    extend = eye_dist * 0.22
    ribbon_l = _under_eye_ribbon(shape_hw, points, lid_idxs_l, drop, extend)
    ribbon_r = _under_eye_ribbon(shape_hw, points, lid_idxs_r, drop, extend)
    hull_l = cv2.bitwise_or(hull_l, ribbon_l)
    hull_r = cv2.bitwise_or(hull_r, ribbon_r)

    # Soft close fills the V-notch where ribbon meets hull (sharp edge cause)
    smooth_ksz = max(7, int(face_w * 0.06)) | 1
    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_ksz, smooth_ksz))
    hull_l = cv2.morphologyEx(hull_l, cv2.MORPH_CLOSE, smooth_k)
    hull_r = cv2.morphologyEx(hull_r, cv2.MORPH_CLOSE, smooth_k)

    cheek_bot = int(min(ml[1], mr[1]) - eye_dist * 0.02)
    # Soft vertical window: no hard flat top (that also made a sharp shelf edge)
    y_left = np.zeros((h, w), dtype=np.uint8)
    y_right = np.zeros((h, w), dtype=np.uint8)
    top_l = int(min(lid_l_y, float(points[lid_idxs_l[0], 1])) - eye_dist * 0.05)
    top_r = int(min(lid_r_y, float(points[lid_idxs_r[0], 1])) - eye_dist * 0.05)
    if cheek_bot > top_l:
        y_left[max(0, top_l) : min(h, cheek_bot), :] = 255
    if cheek_bot > top_r:
        y_right[max(0, top_r) : min(h, cheek_bot), :] = 255

    left_cheek = cv2.bitwise_and(cv2.bitwise_and(hull_l, face), y_left)
    right_cheek = cv2.bitwise_and(cv2.bitwise_and(hull_r, face), y_right)
    left_cheek = cv2.bitwise_and(left_cheek, cv2.bitwise_not(exclude))
    right_cheek = cv2.bitwise_and(right_cheek, cv2.bitwise_not(exclude))
    left_cheek[:, nose_x:] = 0
    right_cheek[:, : nose_x + 1] = 0

    if int((left_cheek > 0).sum()) < 80:
        mid = cv2.bitwise_and(cv2.bitwise_and(face, y_left), cv2.bitwise_not(exclude))
        left_cheek = mid.copy()
        left_cheek[:, nose_x:] = 0
    if int((right_cheek > 0).sum()) < 80:
        mid = cv2.bitwise_and(cv2.bitwise_and(face, y_right), cv2.bitwise_not(exclude))
        right_cheek = mid.copy()
        right_cheek[:, : nose_x + 1] = 0

    left_cheek = cv2.bitwise_and(left_cheek, cv2.bitwise_not(nose_mask))
    right_cheek = cv2.bitwise_and(right_cheek, cv2.bitwise_not(nose_mask))
    left_cheek = cv2.bitwise_and(left_cheek, cv2.bitwise_not(chin_mask))
    right_cheek = cv2.bitwise_and(right_cheek, cv2.bitwise_not(chin_mask))

    # Final soften on the clipped cheek (removes residual jags from ∩)
    left_cheek = cv2.morphologyEx(left_cheek, cv2.MORPH_CLOSE, smooth_k)
    right_cheek = cv2.morphologyEx(right_cheek, cv2.MORPH_CLOSE, smooth_k)
    left_cheek = cv2.bitwise_and(left_cheek, face)
    right_cheek = cv2.bitwise_and(right_cheek, face)
    left_cheek[:, nose_x:] = 0
    right_cheek[:, : nose_x + 1] = 0

    return {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "nose": nose_mask,
        "chin": chin_mask,
    }


def _crop_to_mask(
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
    pad: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return None
    h, w = image_rgb.shape[:2]
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    crop = image_rgb[y0:y1, x0:x1].copy()
    mask_crop = region_mask[y0:y1, x0:x1]
    crop[mask_crop == 0] = (128, 128, 128)
    return crop, mask_crop, {
        "x": x0,
        "y": y0,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def extract_face_regions(
    image_rgb: np.ndarray,
    landmarks: list[dict[str, float]],
    skin_mask: np.ndarray,
    encode_image_base64: Callable[[np.ndarray], str],
    overlay_mask: Callable[..., np.ndarray],
    parsing_map: np.ndarray | None = None,
) -> FaceRegionsResult:
    """Exact regions = landmark geometry ∩ ResNet18 skin/nose parsing."""
    h, w = image_rgb.shape[:2]
    points = _landmarks_to_px(landmarks, w, h)
    raw_masks = build_region_masks(points, (h, w), parsing_map=parsing_map)

    overlay = image_rgb.copy().astype(np.float32)
    regions: list[RegionCrop] = []
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Light open only — heavy open was fragmenting chin / under-eye shelf
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for name in REGION_ORDER:
        geo = raw_masks[name]
        region_mask = cv2.bitwise_and(geo, skin_mask)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, close_k)
        if name not in ("chin", "left_cheek", "right_cheek"):
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, open_k)
        else:
            # Fill small holes without eroding the under-eye / chin pad
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, open_k)

        skin_px = int((region_mask > 0).sum())
        geo_area = max(int((geo > 0).sum()), 1)
        coverage = round(min(100.0, skin_px / geo_area * 100.0), 2)
        if skin_px < 40:
            continue

        color = REGION_COLORS[name]
        tint = np.zeros_like(overlay)
        tint[region_mask > 0] = color
        overlay = overlay * (1 - 0.50 * (region_mask[..., None] > 0)) + tint * 0.50

        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay_u8, contours, -1, color, 2, cv2.LINE_AA)

        ys, xs = np.where(region_mask > 0)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(
            overlay_u8,
            name.replace("_", " "),
            (max(4, cx - 28), max(14, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        overlay = overlay_u8.astype(np.float32)

        cropped = _crop_to_mask(image_rgb, region_mask)
        if cropped is None:
            continue
        crop_rgb, mask_crop, bbox = cropped

        regions.append(
            RegionCrop(
                name=name,
                bbox=bbox,
                skin_pixel_count=skin_px,
                skin_coverage_percent=coverage,
                preview_b64=encode_image_base64(crop_rgb),
                mask_preview_b64=encode_image_base64(
                    overlay_mask(crop_rgb, mask_crop, color=color, alpha=0.55)
                ),
                mask=region_mask.copy(),
            )
        )

    return FaceRegionsResult(
        regions=regions,
        regions_overlay_b64=encode_image_base64(
            np.clip(overlay, 0, 255).astype(np.uint8)
        ),
    )
