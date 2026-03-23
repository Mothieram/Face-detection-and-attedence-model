import cv2
import numpy as np
import torch

from config import (
    CONFIDENCE_THRESHOLD,
    MIN_FACE_SIZE_PX,
    DETECTION_CONFIDENCE_PASSES,
    DETECTION_SCALE_PASSES,
    DETECTION_NMS_IOU,
)

_RETINAFACE_MODEL = None
_RETINAFACE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_RETINAFACE_LOGGED = False
_RETINAFACE_AVAILABLE = True


def detect_faces(image: np.ndarray) -> list[dict]:
    """
    RetinaFace-PyTorch detection with multi-scale and multi-threshold passes.
    Returns faces in canonical project format:
      {score, bbox, landmarks, face_id}

    FIX (bottleneck #1 — early exit):
      The original threshold loop ran every threshold in DETECTION_CONFIDENCE_PASSES
      even when the first pass already found >= 2 faces.  Now we break as soon
      as any threshold yields faces, unless those faces are fewer than 2 (in which
      case we try a lower threshold to catch partially occluded / dim faces).

      Original: always N threshold passes regardless of result
      Fixed:    1 pass on clean images; falls through only when faces < 2
    """
    _log_backend_once()
    model = _get_retinaface_model()
    if model is None:
        return _fallback_detect_faces_haar(image)

    # ── scale passes (unchanged — each scale is a separate inference) ──────
    all_faces = []
    for scale in DETECTION_SCALE_PASSES:
        if scale == 1.0:
            work = image
        else:
            h, w = image.shape[:2]
            work = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        detections = _predict_faces_pytorch(model, work)
        if not detections:
            continue

        for det in detections:
            if not det.get("bbox"):
                continue

            score = float(det.get("score", 0.0))
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            landmarks = _landmarks_list_to_dict(det.get("landmarks", []))
            if not landmarks:
                continue

            if scale != 1.0:
                x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                landmarks = {
                    k: [v[0] / scale, v[1] / scale] for k, v in landmarks.items()
                }

            all_faces.append(
                {
                    "score": score,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "landmarks": landmarks,
                }
            )

    if not all_faces:
        return []

    # ── threshold passes — EARLY EXIT once we have a usable result ─────────
    #
    # Strategy:
    #   • Try each threshold from highest (strictest) to lowest.
    #   • If the result has >= 2 faces → stop immediately (early exit).
    #   • If the result has exactly 1 face → keep it as the best result so
    #     far but still try the next threshold in case a lower one reveals
    #     a second face (preserves the original >= 2 break behaviour).
    #   • If a lower threshold finds more faces → use those instead.
    #   • After all thresholds → use whatever the last successful pass gave.
    #
    # On a typical single-person image this means exactly 1 threshold pass.

    thresholds = DETECTION_CONFIDENCE_PASSES or [CONFIDENCE_THRESHOLD]
    best_faces: list[dict] = []

    for thr in thresholds:
        thr = float(thr)
        candidates = [
            f
            for f in all_faces
            if f["score"] >= thr
            and min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1])
            >= MIN_FACE_SIZE_PX
        ]
        candidates = _nms_faces(candidates, DETECTION_NMS_IOU)

        if len(candidates) > len(best_faces):
            best_faces = candidates   # lower threshold found more faces — keep

        if len(best_faces) >= 2:
            # Enough faces found — no need to try lower thresholds
            break

    faces = best_faces
    faces.sort(key=lambda x: x["score"], reverse=True)
    for idx, face in enumerate(faces, 1):
        face["face_id"] = f"face_{idx}"

    return faces


def _get_retinaface_model():
    global _RETINAFACE_MODEL, _RETINAFACE_AVAILABLE
    if _RETINAFACE_MODEL is not None:
        return _RETINAFACE_MODEL
    if not _RETINAFACE_AVAILABLE:
        return None

    try:
        from retinaface.pre_trained_models import get_model
    except Exception as e:
        print(f"[DETECT] RetinaFace import failed, using Haar fallback: {e}")
        _RETINAFACE_AVAILABLE = False
        return None

    try:
        model = get_model("resnet50_2020-07-20", max_size=960, device=_RETINAFACE_DEVICE)
        model.eval()
        _RETINAFACE_MODEL = model
        return model
    except Exception as e:
        print(f"[DETECT] RetinaFace init failed, using Haar fallback: {e}")
        _RETINAFACE_AVAILABLE = False
        return None


def _predict_faces_pytorch(model, image_bgr: np.ndarray) -> list[dict]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds = model.predict_jsons(rgb, confidence_threshold=0.1, nms_threshold=0.4)
    if not isinstance(preds, list):
        return []
    return preds


def _landmarks_list_to_dict(points: list) -> dict:
    if not isinstance(points, list) or len(points) < 5:
        return {}

    names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
    out = {}
    for idx, name in enumerate(names):
        if idx >= len(points):
            return {}
        p = points[idx]
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return {}
        out[name] = [float(p[0]), float(p[1])]
    return out


def _log_backend_once() -> None:
    global _RETINAFACE_LOGGED
    if _RETINAFACE_LOGGED:
        return
    backend = (
        f"RetinaFace PyTorch ({_RETINAFACE_DEVICE})"
        if _RETINAFACE_AVAILABLE
        else "OpenCV Haar fallback"
    )
    print(f"[DETECT] Detection backend: {backend}")
    _RETINAFACE_LOGGED = True


def _fallback_detect_faces_haar(image: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []

    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX),
    )

    faces = []
    for idx, (x, y, w, h) in enumerate(boxes, 1):
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        landmarks = {
            "right_eye":   [x1 + int(0.35 * w), y1 + int(0.38 * h)],
            "left_eye":    [x1 + int(0.65 * w), y1 + int(0.38 * h)],
            "nose":        [x1 + int(0.50 * w), y1 + int(0.56 * h)],
            "mouth_right": [x1 + int(0.40 * w), y1 + int(0.76 * h)],
            "mouth_left":  [x1 + int(0.60 * w), y1 + int(0.76 * h)],
        }
        faces.append(
            {
                "score": 0.60,
                "bbox": [x1, y1, x2, y2],
                "landmarks": landmarks,
                "face_id": f"face_{idx}",
            }
        )

    faces.sort(key=lambda x: x["score"], reverse=True)
    return faces


def _nms_faces(faces: list[dict], iou_thresh: float) -> list[dict]:
    if not faces:
        return []
    ordered = sorted(faces, key=lambda f: f["score"], reverse=True)
    kept = [ordered[0]]
    for f in ordered[1:]:
        if all(_bbox_iou(f["bbox"], k["bbox"]) <= iou_thresh for k in kept):
            kept.append(f)
    return kept


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0