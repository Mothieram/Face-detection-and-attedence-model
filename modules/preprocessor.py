"""
modules/preprocessor.py
────────────────────────
Image Preprocessing Pipeline

Prepares raw images before face detection.
All operations are non-destructive (works on copies).

Pipeline:
  Raw BGR image
      ↓
  [1] Upscale (if too small for reliable detection)
      ↓
  [2] CLAHE contrast enhancement (LAB color space)
      ↓
  [3] Bilateral denoising — SKIPPED in fast mode
      ↓
  [4] Auto gamma correction (brightness normalization)
      ↓
  [5] Unsharp masking — SKIPPED in fast mode
      ↓
  Preprocessed BGR image

fast=True skips bilateral denoise and sharpening for live camera use.
"""

import cv2
import numpy as np
from config import (
    INPUT_SIZE,
    PREPROCESS_MIN_DIM,
    PREPROCESS_ENABLE_CLAHE,
    PREPROCESS_ENABLE_DENOISE,
    PREPROCESS_ENABLE_GAMMA,
    PREPROCESS_ENABLE_SHARPEN,
)


def preprocess_image(image: np.ndarray, fast: bool = False) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a BGR image.

    Args:
        image : BGR numpy array
        fast  : If True, skips bilateral denoise and sharpening.
                Use fast=True for live camera frames to avoid stalling.

    Returns:
        Preprocessed BGR image ready for RetinaFace detection.
    """
    image = image.copy()
    image = _upscale_if_small(image, min_dim=PREPROCESS_MIN_DIM)

    if PREPROCESS_ENABLE_CLAHE:
        image = _clahe_enhancement(image)

    if not fast and PREPROCESS_ENABLE_DENOISE:
        image = _bilateral_denoise(image)

    if PREPROCESS_ENABLE_GAMMA:
        image = _auto_gamma(image)

    if not fast and PREPROCESS_ENABLE_SHARPEN:
        image = _sharpen(image)

    return image


def preprocess_face_crop(face_bgr: np.ndarray) -> np.ndarray:
    """
    Lightweight preprocessing for a face crop before embedding.
    Returns float32 BGR in [0, 1].
    """
    face = cv2.resize(face_bgr, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    return face.astype(np.float32) / 255.0


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Backward-compatible alias used by embedder.
    Returns RGB float32 in [-1, 1] at INPUT_SIZE.
    """
    face = cv2.resize(face_bgr, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - 0.5) / 0.5


# ─────────────────────────────────────────────
# Module-level singletons — allocated once, reused every frame
# ─────────────────────────────────────────────

# OPT: CLAHE object created once at import time instead of per frame
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# OPT: Gamma LUTs precomputed for both correction directions.
# Only 3 possible outcomes (dark / bright / unchanged) so we
# precompute both and pick at runtime with a single np.where call.
_LUT_DARK   = np.array(
    [int(((i / 255.0) ** (1.0 / 0.7)) * 255) for i in range(256)], dtype=np.uint8
)
_LUT_BRIGHT = np.array(
    [int(((i / 255.0) ** (1.0 / 1.5)) * 255) for i in range(256)], dtype=np.uint8
)


# ─────────────────────────────────────────────
# Internal pipeline steps
# ─────────────────────────────────────────────

def _upscale_if_small(image: np.ndarray, min_dim: int = 300) -> np.ndarray:
    h, w = image.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    return image


def _clahe_enhancement(image: np.ndarray) -> np.ndarray:
    # OPT: reuse module-level _CLAHE singleton — no allocation per frame
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _bilateral_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)


def _auto_gamma(image: np.ndarray) -> np.ndarray:
    # OPT: use precomputed LUTs instead of rebuilding 256-element array per frame
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
    if brightness < 0.4:
        return cv2.LUT(image, _LUT_DARK)
    if brightness > 0.7:
        return cv2.LUT(image, _LUT_BRIGHT)
    return image


def _sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)