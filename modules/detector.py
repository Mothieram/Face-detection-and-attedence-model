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


def detect_faces(image: np.ndarray) -> list[dict]:
    """
    RetinaFace-PyTorch detection with multi-scale and multi-threshold passes.
    Returns faces in canonical project format:
      {score, bbox, landmarks, face_id}
    """
    _log_backend_once()
    model = _get_retinaface_model()

    all_faces = []
    for scale in DETECTION_SCALE_PASSES:
        if scale == 1.0:
            work = image
        else:
            h, w = image.shape[:2]
            work = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

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
                landmarks = {k: [v[0] / scale, v[1] / scale] for k, v in landmarks.items()}

            all_faces.append(
                {
                    "score": score,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "landmarks": landmarks,
                }
            )

    if not all_faces:
        return []

    thresholds = DETECTION_CONFIDENCE_PASSES or [CONFIDENCE_THRESHOLD]
    faces = []
    for thr in thresholds:
        thr = float(thr)
        candidates = [
            f
            for f in all_faces
            if f["score"] >= thr
            and min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= MIN_FACE_SIZE_PX
        ]
        candidates = _nms_faces(candidates, DETECTION_NMS_IOU)
        faces = candidates
        if len(faces) >= 2:
            break

    faces.sort(key=lambda x: x["score"], reverse=True)
    for idx, face in enumerate(faces, 1):
        face["face_id"] = f"face_{idx}"

    return faces


def _get_retinaface_model():
    global _RETINAFACE_MODEL
    if _RETINAFACE_MODEL is not None:
        return _RETINAFACE_MODEL

    try:
        from retinaface.pre_trained_models import get_model
    except Exception as e:
        raise RuntimeError(f"[DETECT] retinaface-pytorch import failed: {e}") from e

    # The package downloads pretrained weights on first run if missing.
    model = get_model("resnet50_2020-07-20", max_size=960, device=_RETINAFACE_DEVICE)
    model.eval()
    _RETINAFACE_MODEL = model
    return model


def _predict_faces_pytorch(model, image_bgr: np.ndarray) -> list[dict]:
    # retinaface-pytorch expects RGB image
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds = model.predict_jsons(rgb, confidence_threshold=0.1, nms_threshold=0.4)
    if not isinstance(preds, list):
        return []
    return preds


def _landmarks_list_to_dict(points: list) -> dict:
    if not isinstance(points, list) or len(points) < 5:
        return {}

    # RetinaFace landmark order convention
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
    print(f"[DETECT] RetinaFace backend: PyTorch ({_RETINAFACE_DEVICE})")
    _RETINAFACE_LOGGED = True


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