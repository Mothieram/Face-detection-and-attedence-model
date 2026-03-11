"""
modules/liveness.py
────────────────────────────────────────────────────────

Liveness Detection Pipeline

Stage 1 — Passive Anti-Spoofing
    Model: SilentFace (MiniFASNetV2 ONNX)
    Input size: 80×80
    Detects:
        • printed photos
        • phone screen replay
        • masks / flat surfaces

Stage 2 — Active Challenge (Head Pose)
    Headless yaw-motion validation from camera frames.
    Uses yaw estimation from RetinaFace landmarks.

IMPORTANT
─────────
The camera is NOT reopened here.
The existing camera handle from main.py is reused.
This prevents the “camera busy” bug.

Public API
──────────
result = check_liveness(frame, face, cap)

Returns:
{
    passed : bool
    reason : str
    method : passive | active | passive+active
    score  : float
}
"""

import os
import time
import threading
import urllib.request
import tempfile

import cv2
import numpy as np

from config import (
    LIVENESS_PASSIVE_REAL_THRESHOLD,
    LIVENESS_PASSIVE_SPOOF_THRESHOLD,
    LIVENESS_PASSIVE_MODEL_PATH,
    LIVENESS_ACTIVE_ENABLED,
    LIVENESS_HEADPOSE_TIMEOUT,
    LIVENESS_HEADPOSE_YAW_THRESHOLD,
)

# ─────────────────────────────────────────────
# ONNX Runtime session
# ─────────────────────────────────────────────

_ort_session = None
# Backward-compat for callers that expect multi-session loading APIs.
_ort_sessions = {}
_ort_lock = threading.Lock()
_passive_download_attempted = False

# ─────────────────────────────────────────────
# Micro-motion detection (anti replay)
# ─────────────────────────────────────────────
_MOTION_WINDOW = 5
_MOTION_THRESHOLD = 2.5
_motion_buffer = []
_passive_frame_buffer = []


# SilentFace model URLs
_MODEL_URLS = [
    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx",
    "https://huggingface.co/spaces/aaavvvrrr/face-anti-spoofing/resolve/main/weights/modelrgb.onnx",
]


# ═══════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════

def check_liveness(image: np.ndarray, face: dict, camera_index: int = 0, cap=None) -> dict:

    passive = _passive_check(image, face)

    if not passive.get("available", True):
        _reset_liveness_buffers()
        return {"passed": False, "reason": "Passive unavailable", "method": "passive", "score": 0.0}

    passive_score = float(passive.get("score", 0.5))

    if passive_score >= LIVENESS_PASSIVE_REAL_THRESHOLD:
        _reset_liveness_buffers()
        return {
            "passed": True,
            "reason": "Passive real",
            "method": "passive",
            "score": passive_score,
        }

    if passive_score < LIVENESS_PASSIVE_SPOOF_THRESHOLD:
        _reset_liveness_buffers()
        return {
            "passed": False,
            "reason": "Passive spoof detected",
            "method": "passive",
            "score": passive_score,
        }

    if not LIVENESS_ACTIVE_ENABLED:
        _reset_liveness_buffers()
        return {
            "passed": False,
            "reason": "Passive uncertain",
            "method": "passive",
            "score": passive_score,
        }

    # Keep a stable camera session: active liveness must reuse caller-provided cap.
    # Opening a second capture can trigger camera-busy/freeze issues.
    if cap is None or not cap.isOpened():
        _reset_liveness_buffers()
        return {
            "passed": False,
            "reason": "Active challenge camera unavailable",
            "method": "passive+active",
            "score": passive_score,
        }

    active = _active_headpose_challenge(cap)

    if active["passed"]:
        _reset_liveness_buffers()
        return {
            "passed": True,
            "reason": "Active challenge passed",
            "method": "passive+active",
            "score": passive_score,
        }

    _reset_liveness_buffers()
    return {
        "passed": False,
        "reason": active.get("reason", "Active challenge failed"),
        "method": "passive+active",
        "score": passive_score,
    }


# ═══════════════════════════════════════════════
# PASSIVE — SilentFace
# ═══════════════════════════════════════════════

def _passive_check(image: np.ndarray, face: dict):

    session = _load_ort_session()

    if session is None:
        return {"available": False, "score": 0.5}

    crop = _crop_face(image, face["bbox"])

    if crop is None:
        return {"score": 0.5}

    if not _check_micro_motion(crop):
        return {"score": 0.0}

    blob = _preprocess(crop)

    input_name = session.get_inputs()[0].name

    output = session.run(None, {input_name: blob})[0]

    probs = _softmax(output[0])

    real_score = float(probs[1])

    return {"score": real_score}


def _load_ort_session():

    global _ort_session
    global _ort_sessions
    global _passive_download_attempted

    if _ort_session is not None:
        if "passive" not in _ort_sessions:
            _ort_sessions["passive"] = _ort_session
        return _ort_session

    with _ort_lock:

        if _ort_session is not None:
            if "passive" not in _ort_sessions:
                _ort_sessions["passive"] = _ort_session
            return _ort_session

        model_path = os.path.expanduser(LIVENESS_PASSIVE_MODEL_PATH)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if not os.path.exists(model_path):

            if _passive_download_attempted:
                return None

            _passive_download_attempted = True

            print("[LIVENESS] Downloading SilentFace model...")

            for url in _MODEL_URLS:

                try:
                    _download_model(url, model_path)
                    break
                except Exception:
                    continue

        try:

            import onnxruntime as ort

            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3

            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

            _ort_session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=providers,
            )
            _ort_sessions["passive"] = _ort_session

            print("[LIVENESS] SilentFace loaded")

            return _ort_session

        except Exception as e:
            print("[LIVENESS] Model load failed:", e)
            return None


def _load_all_sessions():
    """
    Compatibility API used by api.py startup.
    Returns a dict of available liveness sessions.
    """
    session = _load_ort_session()
    if session is None:
        _ort_sessions.clear()
    return _ort_sessions


# ═══════════════════════════════════════════════
# ACTIVE — HEAD POSE CHALLENGE
# ═══════════════════════════════════════════════

def _active_headpose_challenge(cap):

    if cap is None or not cap.isOpened():
        print("[LIVENESS] Camera unavailable")
        return {"passed": False, "reason": "Camera unavailable"}

    from modules.detector import detect_faces

    # Phase 1: neutral yaw calibration
    print("[LIVENESS] Active challenge: calibrating neutral head pose...")
    calib_target = 12
    neutral_samples = []
    calib_deadline = time.time() + 5.0
    while len(neutral_samples) < calib_target and time.time() < calib_deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        faces = detect_faces(frame)
        if faces:
            face = max(faces, key=lambda f: f["score"])
            neutral_samples.append(_estimate_yaw(face["landmarks"]))

    if len(neutral_samples) < 6:
        print("[LIVENESS] Calibration failed")
        return {"passed": False, "reason": "Calibration failed"}

    neutral_yaw = float(np.median(neutral_samples))

    # Phase 2: directional challenge (user may turn either side)
    print("[LIVENESS] Active challenge: waiting for LEFT/RIGHT head turn...")
    deadline = time.time() + LIVENESS_HEADPOSE_TIMEOUT

    passed = False

    while time.time() < deadline:

        ok, frame = cap.read()

        if not ok:
            time.sleep(0.01)
            continue

        faces = detect_faces(frame)

        if faces:

            face = max(faces, key=lambda f: f["score"])

            yaw = _estimate_yaw(face["landmarks"]) - neutral_yaw

            if abs(yaw) > LIVENESS_HEADPOSE_YAW_THRESHOLD:
                passed = True
                break

    if not passed:
        return {"passed": False, "reason": "Head pose challenge failed"}

    return {"passed": True}


# ═══════════════════════════════════════════════
# YAW ESTIMATION
# ═══════════════════════════════════════════════

def _estimate_yaw(landmarks):

    le = np.array(landmarks["left_eye"])
    re = np.array(landmarks["right_eye"])
    nose = np.array(landmarks["nose"])

    mid = (le + re) / 2

    eye_dist = np.linalg.norm(le - re)

    if eye_dist < 1e-6:
        return 0

    yaw = (nose[0] - mid[0]) / eye_dist

    return yaw


# ═══════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════

def _draw_progress_bar(
    frame: np.ndarray,
    current: int,
    total: int,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str = "",
):

    total = max(1, int(total))
    progress = min(1.0, max(0.0, float(current) / float(total)))
    fill_w = int(width * progress)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (180, 180, 180), 2)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), (0, 200, 80), -1)

    if label:
        cv2.putText(
            frame,
            f"{label}: {int(progress * 100)}%",
            (x, y + height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def _reset_liveness_buffers():

    _motion_buffer.clear()
    _passive_frame_buffer.clear()


def _check_micro_motion(crop: np.ndarray) -> bool:
    """
    Detect small motion between face crops.
    Helps block phone/video replay attacks.
    """
    global _motion_buffer

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    _motion_buffer.append(gray)

    if len(_motion_buffer) < _MOTION_WINDOW:
        return True

    if len(_motion_buffer) > _MOTION_WINDOW:
        _motion_buffer.pop(0)

    diffs = []
    for i in range(len(_motion_buffer) - 1):
        diff = cv2.absdiff(_motion_buffer[i], _motion_buffer[i + 1])
        diffs.append(np.mean(diff))

    avg_motion = float(np.mean(diffs))

    return avg_motion > _MOTION_THRESHOLD

def _crop_face(image, bbox, scale=2.7):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    fw = x2 - x1
    fh = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    half_w = fw * scale / 2
    half_h = fh * scale / 2

    nx1 = int(max(0, cx - half_w))
    ny1 = int(max(0, cy - half_h))
    nx2 = int(min(w, cx + half_w))
    ny2 = int(min(h, cy + half_h))

    crop = image[ny1:ny2, nx1:nx2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (112, 112))


def _preprocess(img):

    img = img.astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    img = (img - mean) / std

    img = img.transpose(2, 0, 1)

    blob = np.expand_dims(img, 0)

    return blob.astype(np.float32)


def _softmax(x):

    e = np.exp(x - np.max(x))

    return e / e.sum()


def _download_model(url, destination):

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    fd, tmp = tempfile.mkstemp(suffix=".onnx")

    os.close(fd)

    with open(tmp, "wb") as f:
        f.write(data)

    os.replace(tmp, destination)
