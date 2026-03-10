"""
utils/visualizer.py
────────────────────
Drawing and display utilities.

Handles all OpenCV rendering — bounding boxes, landmarks,
confidence labels, match labels, and window management.
"""

import cv2
import numpy as np
from config import (
    BBOX_COLOR, LANDMARK_COLORS,
    SHOW_LANDMARKS, SHOW_CONFIDENCE,
    DISPLAY_MAX_W, DISPLAY_MAX_H,
    CONFIDENCE_THRESHOLD,
)


def draw_detections(image: np.ndarray,
                    faces: list[dict],
                    labels: dict = None) -> np.ndarray:
    """
    Annotate an image with bounding boxes, landmarks, and optional match labels.

    Args:
        image  : BGR np.ndarray to draw on (will be modified in-place)
        faces  : list of face dicts from detector.detect_faces()
        labels : optional dict {face_id: "Name (score)"} for match overlay

    Returns:
        Annotated BGR image.
    """
    for face in faces:
        if face["score"] < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = face["bbox"]

        # ── Bounding box ─────────────────────────
        cv2.rectangle(image, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        # ── Confidence label above box ───────────
        if SHOW_CONFIDENCE:
            _draw_confidence_label(image, face["score"], x1, y1)

        # ── Match label below box ────────────────
        if labels and face["face_id"] in labels:
            _draw_match_label(image, labels[face["face_id"]], x1, y2)

        # ── 5-point landmarks ────────────────────
        if SHOW_LANDMARKS and "landmarks" in face:
            _draw_landmarks(image, face["landmarks"])

    return image


def draw_face_count(image: np.ndarray, count: int) -> np.ndarray:
    """Overlay face count in the top-left corner."""
    cv2.putText(image, f"Faces: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, BBOX_COLOR, 2, cv2.LINE_AA)
    return image


def draw_status_label(image: np.ndarray, text: str,
                      color: tuple = (0, 255, 255)) -> np.ndarray:
    """Overlay a single status string (e.g. 'Registered: Alice') top-left."""
    cv2.putText(image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return image


def show_window(title: str, image: np.ndarray):
    """
    Display an image in a resized OpenCV window.
    Caps display at DISPLAY_MAX_W × DISPLAY_MAX_H (never upscales).
    Press 's' to save the full-resolution image.
    Press any other key to close.
    """
    display = _fit_to_screen(image)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, display.shape[1], display.shape[0])
    cv2.imshow(title, display)

    print("[DISPLAY] Press 's' to save | Any key to close.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            save_path = "result_pipeline.jpg"
            cv2.imwrite(save_path, image)   # save full resolution
            print(f"[DISPLAY] Saved full-resolution image → {save_path}")
        else:
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Internal drawing helpers
# ─────────────────────────────────────────────

def _draw_confidence_label(image: np.ndarray, score: float, x1: int, y1: int):
    label         = f"{score:.2f}"
    (tw, th), _   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), BBOX_COLOR, -1)
    cv2.putText(image, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_match_label(image: np.ndarray, label: str, x1: int, y2: int):
    color = (0, 255, 0) if "No Match" not in label else (0, 0, 255)
    cv2.putText(image, label, (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _draw_landmarks(image: np.ndarray, landmarks: dict):
    for point_name, coords in landmarks.items():
        px, py = int(coords[0]), int(coords[1])
        color  = LANDMARK_COLORS.get(point_name, (255, 255, 0))
        cv2.circle(image, (px, py), 4, color, -1)
        cv2.circle(image, (px, py), 4, (255, 255, 255), 1)   # white ring


def _fit_to_screen(image: np.ndarray) -> np.ndarray:
    """Scale image down to fit within DISPLAY_MAX_W × DISPLAY_MAX_H. Never upscales."""
    h, w  = image.shape[:2]
    scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
    if scale < 1.0:
        return cv2.resize(image, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return image
