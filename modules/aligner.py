"""
modules/aligner.py
───────────────────
STEP 3 — Face Alignment

Aligns a detected face to a standard 112x112 canonical template
using a 5-point affine transform.

Primary method : estimateAffinePartial2D (RANSAC) — 5-point similarity transform
Fallback method : simple eye-axis rotation crop
"""

import cv2
import numpy as np
from config import INPUT_SIZE, LANDMARK_TEMPLATE, MIN_FACE_SIZE, MIN_BLUR_SCORE, MAX_YAW_DEGREES


def align_face(image: np.ndarray, landmarks: dict, bbox: list | None = None) -> np.ndarray:
    """
    Affine-align a face to the 112x112 canonical landmark template.

    Args:
        image     : full BGR image
        landmarks : dict — {right_eye, left_eye, nose, mouth_right, mouth_left}
        bbox      : optional [x1,y1,x2,y2] used as fallback crop region

    Returns:
        Aligned BGR face crop at INPUT_SIZE (112x112).
    """
    src_pts = _landmarks_to_array(landmarks)

    M, _ = cv2.estimateAffinePartial2D(
        src_pts, LANDMARK_TEMPLATE,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if M is None:
        return _fallback_align(image, landmarks, bbox=bbox)

    return cv2.warpAffine(
        image, M, INPUT_SIZE,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def check_face_quality(image: np.ndarray, face: dict) -> dict:
    """
    Quality gate — checks face size, blur, and yaw.
    Returns: {passed, reason, face_size, blur_score, yaw_deg}
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(face["bbox"], h, w)

    fw, fh    = max(0, x2 - x1), max(0, y2 - y1)
    face_size = min(fw, fh)

    if face_size < MIN_FACE_SIZE:
        return _fail(f"Face too small ({fw}x{fh})", face_size, 0.0, 0.0)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return _fail("Invalid face crop", face_size, 0.0, 0.0)

    gray       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < MIN_BLUR_SCORE:
        return _fail(f"Blur too high ({blur_score:.1f} < {MIN_BLUR_SCORE:.1f})",
                     face_size, blur_score, 0.0)

    lm      = face.get("landmarks", {})
    le      = np.array(lm.get("left_eye",  [0.0, 0.0]), dtype=np.float32)
    re      = np.array(lm.get("right_eye", [0.0, 0.0]), dtype=np.float32)
    nose    = np.array(lm.get("nose",      [0.0, 0.0]), dtype=np.float32)
    eye_mid = (le + re) / 2.0
    eye_dist = max(float(np.linalg.norm(le - re)), 1e-6)
    yaw_deg  = float(abs(nose[0] - eye_mid[0]) / eye_dist * 90.0)

    if yaw_deg > MAX_YAW_DEGREES:
        return _fail(f"Yaw too large ({yaw_deg:.1f} > {MAX_YAW_DEGREES:.1f})",
                     face_size, blur_score, yaw_deg)

    return {"passed": True, "reason": "OK",
            "face_size": face_size, "blur_score": blur_score, "yaw_deg": yaw_deg}


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _clamp_bbox(bbox: list, h: int, w: int) -> tuple[int, int, int, int]:
    """Clamp bbox coordinates to image bounds."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _fail(reason: str, face_size: int, blur_score: float, yaw_deg: float) -> dict:
    return {"passed": False, "reason": reason,
            "face_size": face_size, "blur_score": blur_score, "yaw_deg": yaw_deg}


def _landmarks_to_array(landmarks: dict) -> np.ndarray:
    return np.array([
        landmarks["right_eye"],
        landmarks["left_eye"],
        landmarks["nose"],
        landmarks["mouth_right"],
        landmarks["mouth_left"],
    ], dtype=np.float32)


def _fallback_align(image: np.ndarray, landmarks: dict,
                    bbox: list | None = None) -> np.ndarray:
    """
    Fallback: rotate so eyes are level, then crop and resize to INPUT_SIZE.
    """
    left_eye  = np.array(landmarks["left_eye"])
    right_eye = np.array(landmarks["right_eye"])
    dy, dx    = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle     = np.degrees(np.arctan2(dy, dx))
    center    = (int((left_eye[0] + right_eye[0]) // 2),
                 int((left_eye[1] + right_eye[1]) // 2))

    M       = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # If bbox provided use it as the crop region (with padding)
    if bbox is not None:
        h, w = rotated.shape[:2]
        bx1, by1, bx2, by2 = _clamp_bbox(bbox, h, w)
        pad  = int(0.15 * max(bx2 - bx1, by2 - by1))
        bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
        bx2, by2 = min(w, bx2 + pad), min(h, by2 + pad)
        if bx2 > bx1 and by2 > by1:
            return cv2.resize(rotated[by1:by2, bx1:bx2], INPUT_SIZE,
                              interpolation=cv2.INTER_CUBIC)

    # Derive crop from rotated landmark positions
    pts     = _landmarks_to_array(landmarks)
    pts_h   = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    pts_rot = (M @ pts_h.T).T
    min_xy  = np.min(pts_rot, axis=0)
    max_xy  = np.max(pts_rot, axis=0)
    cx, cy  = (min_xy + max_xy) / 2.0
    size    = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]) * 2.2
    half    = size / 2.0

    h, w = rotated.shape[:2]
    x1 = int(max(0, cx - half));  y1 = int(max(0, cy - half))
    x2 = int(min(w,  cx + half)); y2 = int(min(h,  cy + half))

    if x2 <= x1 or y2 <= y1:
        return cv2.resize(rotated, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    return cv2.resize(rotated[y1:y2, x1:x2], INPUT_SIZE, interpolation=cv2.INTER_CUBIC)