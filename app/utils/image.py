import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image. Upload a valid image file.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_image_base64(image_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encode an RGB image as a base64 data URL."""
    pil = Image.fromarray(image_rgb.astype(np.uint8))
    buffer = BytesIO()
    pil.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{encoded}"


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 120),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask onto the image for visualization."""
    overlay = image_rgb.copy().astype(np.float32)
    tint = np.zeros_like(overlay)
    tint[mask > 0] = color
    blended = overlay * (1 - alpha * (mask[..., None] > 0)) + tint * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_landmarks(
    image_rgb: np.ndarray,
    landmarks: list[dict[str, float]],
    color: tuple[int, int, int] = (40, 120, 255),
    radius: int | None = None,
) -> np.ndarray:
    """Draw normalized landmarks onto an RGB image."""
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    if radius is None:
        radius = max(1, int(round(min(h, w) * 0.003)))
    for point in landmarks:
        x = int(point["x"] * w)
        y = int(point["y"] * h)
        cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas
