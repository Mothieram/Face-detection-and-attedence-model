"""
Public API
──────────
result = check_liveness(frame, face, cap)

Returns:
{
    passed       : bool
    reason       : str
    method       : "passive" | "passive+active"
    score        : float        # fused real score (higher = more real)
    model_scores : dict         # per-model spoof probabilities
}
"""

import os
import time
import threading
import tempfile
import urllib.request
import concurrent.futures

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from config import (
    LIVENESS_PASSIVE_REAL_THRESHOLD,
    LIVENESS_PASSIVE_SPOOF_THRESHOLD,
    LIVENESS_ACTIVE_ENABLED,
    LIVENESS_HEADPOSE_TIMEOUT,
    LIVENESS_HEADPOSE_YAW_THRESHOLD,
)

# ─────────────────────────────────────────────
# Weights directory
# ─────────────────────────────────────────────
_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")

def _W(filename: str) -> str:
    """Resolve a weight filename to its full path."""
    return os.path.join(_WEIGHTS_DIR, filename)


# ─────────────────────────────────────────────
# Auto-download URLs (HuggingFace repo)
# Each entry: filename → list of fallback URLs
# ─────────────────────────────────────────────
_HF_BASE = "https://huggingface.co/spaces/mothieram/face-anti-spoofing/resolve/main/weights"

_WEIGHT_URLS: dict[str, list[str]] = {
    "ICM2O.pth.tar": [
        f"{_HF_BASE}/ICM2O.pth.tar",
        f"{_HF_BASE}/ICM2O.pth.tar?download=true",
    ],
    "IOM2C.pth.tar": [
        f"{_HF_BASE}/IOM2C.pth.tar",
        f"{_HF_BASE}/IOM2C.pth.tar?download=true",
    ],
    "modelrgb.onnx": [
        f"{_HF_BASE}/modelrgb.onnx",
        f"{_HF_BASE}/modelrgb.onnx?download=true",
    ],
    "2.7_80x80_MiniFASNetV2.pth": [
        f"{_HF_BASE}/2.7_80x80_MiniFASNetV2.pth",
        f"{_HF_BASE}/2.7_80x80_MiniFASNetV2.pth?download=true",
    ],
    "4_0_0_80x80_MiniFASNetV1SE.pth": [
        f"{_HF_BASE}/4_0_0_80x80_MiniFASNetV1SE.pth",
        f"{_HF_BASE}/4_0_0_80x80_MiniFASNetV1SE.pth?download=true",
    ],
}

# Track which files have already had a download attempt this session
_download_attempted: set[str] = set()
_download_lock = threading.Lock()


def _ensure_weight(filename: str) -> bool:
    """
    Download weight file from HuggingFace if not already present.
    Returns True if the file exists (or was downloaded successfully).
    """
    path = _W(filename)
    if os.path.exists(path):
        return True

    with _download_lock:
        # Re-check inside lock
        if os.path.exists(path):
            return True
        if filename in _download_attempted:
            return False
        _download_attempted.add(filename)

        os.makedirs(_WEIGHTS_DIR, exist_ok=True)
        urls = _WEIGHT_URLS.get(filename, [])

        print(f"[LIVENESS] Downloading {filename}...")
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()

                # Write atomically via temp file
                fd, tmp = tempfile.mkstemp(dir=_WEIGHTS_DIR, suffix=".tmp")
                os.close(fd)
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)

                print(f"[LIVENESS] Downloaded {filename} ({len(data) // 1024} KB)")
                return True

            except Exception as e:
                print(f"[LIVENESS] Download failed ({url}): {e}")
                continue

        print(f"[LIVENESS] Could not download {filename} — model will be skipped")
        return False


# ─────────────────────────────────────────────
# Ensemble config
# ─────────────────────────────────────────────
_ENSEMBLE = [
    # (model_id, weight, cache_key)
    ("ICM2O",    0.25, "icm2o"),
    ("IOM2C",    0.25, "iom2c"),
    ("modelrgb", 0.25, "modelrgb"),
    ("SASF",     0.25, "sasf"),
]

_models: dict     = {}
_model_lock       = threading.Lock()

# ─────────────────────────────────────────────
# Micro-motion anti-replay buffer
# ─────────────────────────────────────────────
_MOTION_WINDOW    = 5
_MOTION_THRESHOLD = 2.5
_motion_buffer: list = []


# ═══════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════

def check_liveness(
    image: np.ndarray,
    face:  dict,
    camera_index: int = 0,
    cap=None,
    active_enabled_override: bool | None = None,
) -> dict:
    """
    Parameters
    ----------
    image : BGR frame  (H×W×3, uint8)
    face  : {bbox: [x1,y1,x2,y2], landmarks: (5,2) array or 5-pt dict}
    cap   : already-open cv2.VideoCapture (needed only for active challenge)
    """
    passive = _ensemble_passive_check(image, face)

    if not passive.get("available", True):
        _reset_buffers()
        return {
            "passed": False,
            "reason": "All passive models unavailable",
            "method": "passive",
            "score":  0.0,
            "model_scores": passive.get("model_scores", {}),
        }

    fused_spoof  = float(passive["fused_spoof"])
    real_score   = 1.0 - fused_spoof
    model_scores = passive.get("model_scores", {})

    if real_score >= LIVENESS_PASSIVE_REAL_THRESHOLD:
        _reset_buffers()
        return {"passed": True,  "reason": "Passive ensemble: real",
                "method": "passive", "score": real_score, "model_scores": model_scores}

    if real_score < LIVENESS_PASSIVE_SPOOF_THRESHOLD:
        _reset_buffers()
        return {"passed": False, "reason": "Passive ensemble: spoof detected",
                "method": "passive", "score": real_score, "model_scores": model_scores}

    # Uncertain zone → active challenge
    active_on = LIVENESS_ACTIVE_ENABLED if active_enabled_override is None else bool(active_enabled_override)

    if not active_on:
        _reset_buffers()
        return {"passed": False, "reason": "Passive uncertain (active disabled)",
                "method": "passive", "score": real_score, "model_scores": model_scores}

    if cap is None or not cap.isOpened():
        _reset_buffers()
        return {"passed": False, "reason": "Active challenge camera unavailable",
                "method": "passive+active", "score": real_score, "model_scores": model_scores}

    active = _active_headpose_challenge(cap)
    _reset_buffers()
    return {
        "passed": active["passed"],
        "reason": "Active challenge passed" if active["passed"] else active.get("reason", "Active challenge failed"),
        "method": "passive+active",
        "score":  real_score,
        "model_scores": model_scores,
    }


# ═══════════════════════════════════════════════
# PASSIVE ENSEMBLE
# ═══════════════════════════════════════════════

def _ensemble_passive_check(image: np.ndarray, face: dict) -> dict:
    bbox      = face.get("bbox", [0, 0, image.shape[1], image.shape[0]])
    landmarks = face.get("landmarks")

    # Micro-motion guard
    crop = _motion_crop(image, bbox)
    if crop is not None and not _check_micro_motion(crop):
        return {"fused_spoof": 1.0, "model_scores": {"micro_motion": 1.0}}

    _load_all_models()

    def _run(model_id, weight, key):
        model = _models.get(key)
        if model is None:
            return model_id, weight, None
        try:
            _spoof_flag, spoof_prob, _crop = model(image, bbox, landmarks)
            return model_id, weight, float(spoof_prob)
        except Exception as exc:
            print(f"[LIVENESS] {model_id} inference error: {exc}")
            return model_id, weight, None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_run, mid, w, key) for mid, w, key in _ENSEMBLE]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    model_scores: dict = {}
    total_weight = 0.0
    weighted_sum = 0.0
    for model_id, weight, score in results:
        if score is not None:
            model_scores[model_id] = score
            weighted_sum += weight * score
            total_weight += weight

    if total_weight == 0.0:
        return {"available": False, "model_scores": model_scores}
    fused_spoof = weighted_sum / total_weight
    fused_real = 1.0 - fused_spoof

    # Debug visibility: spoof probability + real confidence per model.
    print("[LIVENESS] Passive model scores:")
    for model_id, _, _ in _ENSEMBLE:
        score = model_scores.get(model_id)
        if score is None:
            print(f"  {model_id}: unavailable")
        else:
            print(f"  {model_id}: spoof={score:.4f}, real_conf={1.0 - score:.4f}")
    print(f"  FUSED: spoof={fused_spoof:.4f}, real_conf={fused_real:.4f}")

    return {"fused_spoof": fused_spoof, "model_scores": model_scores}


# ═══════════════════════════════════════════════
# MODEL LOADING  (with auto-download)
# ═══════════════════════════════════════════════

def _load_all_models():
    """
    Lazy-load all four spoof models once into _models cache.
    Each weight file is auto-downloaded from HuggingFace if missing.
    """
    with _model_lock:

        # ── ICM2O ──────────────────────────────────────────────────────
        if "icm2o" not in _models:
            if _ensure_weight("ICM2O.pth.tar"):
                try:
                    _models["icm2o"] = _SpoofModel("ICM2O", threshold=0.90)
                    print("[LIVENESS] Loaded: ICM2O (PyTorch)")
                except Exception as e:
                    print(f"[LIVENESS] ICM2O load failed: {e}")
                    _models["icm2o"] = None
            else:
                _models["icm2o"] = None

        # ── IOM2C ──────────────────────────────────────────────────────
        if "iom2c" not in _models:
            if _ensure_weight("IOM2C.pth.tar"):
                try:
                    _models["iom2c"] = _SpoofModel("IOM2C", threshold=0.90)
                    print("[LIVENESS] Loaded: IOM2C (PyTorch)")
                except Exception as e:
                    print(f"[LIVENESS] IOM2C load failed: {e}")
                    _models["iom2c"] = None
            else:
                _models["iom2c"] = None

        # ── modelrgb ───────────────────────────────────────────────────
        if "modelrgb" not in _models:
            if _ensure_weight("modelrgb.onnx"):
                try:
                    _models["modelrgb"] = _SpoofONNX("modelrgb", threshold=0.28)
                    print("[LIVENESS] Loaded: modelrgb (ONNX)")
                except Exception as e:
                    print(f"[LIVENESS] modelrgb load failed: {e}")
                    _models["modelrgb"] = None
            else:
                _models["modelrgb"] = None

        # ── SASF (MiniFASNetV2 + MiniFASNetV1SE) ───────────────────────
        if "sasf" not in _models:
            v2_ok  = _ensure_weight("2.7_80x80_MiniFASNetV2.pth")
            v1_ok  = _ensure_weight("4_0_0_80x80_MiniFASNetV1SE.pth")
            if v2_ok and v1_ok:
                try:
                    from modules.SASF import aSASF
                    _models["sasf"] = aSASF(threshold=0.085)
                    print("[LIVENESS] Loaded: SASF (MiniFASNetV2 + V1SE)")
                except Exception as e:
                    print(f"[LIVENESS] SASF load failed: {e}")
                    _models["sasf"] = None
            else:
                _models["sasf"] = None


def _load_all_sessions() -> dict:
    """Compatibility shim called by api.py at startup."""
    _load_all_models()
    return {k: v for k, v in _models.items() if v is not None}


# ═══════════════════════════════════════════════
# INLINED MODEL CLASSES  (extracted from IADG.py)
# Loads weights directly — no import of IADG.py
# ═══════════════════════════════════════════════

def _load_checkpoint(path: str) -> dict:
    """Load a PyTorch checkpoint, compatible with torch >= 2.6."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _crop_from_5landmarks(img: np.ndarray, lm: np.ndarray, margin: float) -> np.ndarray:
    """Landmark-based face crop with margin."""
    if margin == 0:
        return img
    x, y   = round(float(lm[:, 0].min())), round(float(lm[:, 1].min()))
    w, h   = round(float(lm[:, 0].max())) - x, round(float(lm[:, 1].max())) - y
    side   = max(w, h)
    m      = int(side * margin / 2)
    x1 = max(0, x - m);            x2 = min(img.shape[1], x + side + m)
    y1 = max(0, y - m);            y2 = min(img.shape[0], y + side + m)
    return img[y1:y2, x1:x2]


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert landmark dict or array to (5,2) float32 array."""
    if isinstance(landmarks, dict):
        order = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
        return np.array([landmarks[k] for k in order], dtype=np.float32)
    return np.array(landmarks, dtype=np.float32)[:, :2]


# ── PyTorch architecture ────────────────────────────────────────────────────

def _conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)


class _DKGModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        super().__init__()
        self.k       = k
        self.group   = inplanes // 2
        self.pad     = padding
        self.stride  = stride
        self.conv_k      = nn.Conv2d(1, 1, 1, bias=True)
        self.conv_kernel = nn.Conv2d(1, k * k, 1, bias=True)
        self.conv_static = nn.Conv2d(inplanes // 2, inplanes // 2, 3, padding=1, bias=True)
        self.fuse        = nn.Conv2d(inplanes, planes, 1, bias=True)
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C = x.shape[:2]
        x1, x2 = x[:, :C//2], x[:, C//2:]
        g      = self.avg_pool(x1).permute(0, 2, 1, 3)
        kernel = self.conv_kernel(g).permute(0, 3, 2, 1)
        out = []
        for f_one, g_one in zip(torch.split(x1, 1, 0), torch.split(kernel, 1, 0)):
            g_k = self.conv_k(g_one).reshape(g_one.size(2), 1, self.k, self.k)
            pad = ((self.k-1)//2,)*4 if self.pad is None else (self.pad,)*4
            out.append(F.conv2d(F.pad(f_one, pad), g_k, stride=self.stride, groups=self.group))
        return self.fuse(torch.cat([torch.cat(out, 0), self.conv_static(x2)], 1))


class _ConvBlockGate(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dkg_flag):
        super().__init__()
        self.conv = (_DKGModule(3, in_ch, out_ch, m=16, padding=1, stride=1)
                     if dkg_flag else _conv3x3(in_ch, out_ch))
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class _BasicBlockGate(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dkg_flag):
        super().__init__()
        self.b1   = _ConvBlockGate(in_ch,  128,    0,       False)
        self.b2   = _ConvBlockGate(128,    196,    padding, dkg_flag)
        self.b3   = _ConvBlockGate(196,    out_ch, 0,       False)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.b3(self.b2(self.b1(x))))


class _FeatExtractor(nn.Module):
    def __init__(self, dkg_flag, in_channels=6):
        super().__init__()
        self.inc   = _ConvBlockGate(in_channels, 64,  0, False)
        self.down1 = _BasicBlockGate(64,  128, 1, dkg_flag)
        self.down2 = _BasicBlockGate(128, 128, 1, dkg_flag)
        self.down3 = _BasicBlockGate(128, 128, 1, dkg_flag)

    def forward(self, x):
        x  = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        cat = torch.cat([F.adaptive_avg_pool2d(x1, 32),
                         F.adaptive_avg_pool2d(x2, 32), x3], 1)
        return {"cat_feat": cat, "out": cat}


class _FeatEmbedder(nn.Module):
    def __init__(self, in_channels=384):
        super().__init__()
        self.b1   = _ConvBlockGate(in_channels, 128, False, False)
        self.b2   = _ConvBlockGate(128,         256, False, False)
        self.b3   = _ConvBlockGate(256,         512, False, False)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.gap(self.b3(x))
        return {"out": self.fc(x.view(x.size(0), -1))}


class _Framework(nn.Module):
    def __init__(self, total_dkg_flag, style_dim, base_style_num,
                 concentration_coeff, in_channels=6, mid_channels=384,
                 model_initial="kaiming", clusters=2):
        super().__init__()
        self.FeatExtractor = _FeatExtractor(dkg_flag=total_dkg_flag, in_channels=in_channels)
        self.Classifier    = _FeatEmbedder(in_channels=mid_channels)

    def forward(self, x):
        return self.Classifier(self.FeatExtractor(x)["out"])


# ── _SpoofModel: ICM2O / IOM2C .pth.tar ───────────────────────────────────

class _SpoofModel:
    def __init__(self, model_name: str, threshold: float):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.crop      = 0.7  # landmark crop margin for ICM2O / IOM2C

        ckpt       = _load_checkpoint(_W(f"{model_name}.pth.tar"))
        transform  = ckpt["args"].transform
        model_defs = ckpt["args"].model

        self.model = _Framework(**model_defs["params"])
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([transform["image_size"]] * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform["mean"], std=transform["std"]),
        ])

    def __call__(self, image: np.ndarray, bbox, landmarks):
        """image: BGR uint8"""
        # Model was trained on RGB ordering; convert from OpenCV BGR input.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lm  = _landmarks_to_array(landmarks)
        img = _crop_from_5landmarks(image, lm, self.crop)
        with torch.no_grad():
            t   = self.transform(img).unsqueeze(0).to(self.device)
            out = self.model(t)
            spoof_prob = float(torch.softmax(out["out"], dim=1)[0, 1].cpu())
        return spoof_prob > self.threshold, spoof_prob, img


# ── _SpoofONNX: modelrgb.onnx ─────────────────────────────────────────────

class _SpoofONNX:
    def __init__(self, model_name: str, threshold: float):
        providers        = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        # Silence warning-level ORT graph initialization logs.
        opts.log_severity_level = 3
        self.session     = ort.InferenceSession(_W(f"{model_name}.onnx"), sess_options=opts, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.crop        = 1.5
        self.threshold   = threshold

    def __call__(self, image: np.ndarray, bbox, landmarks):
        """image: BGR uint8"""
        # modelrgb expects RGB channel order; convert once before preprocessing.
        image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lm         = _landmarks_to_array(landmarks)
        img        = _crop_from_5landmarks(image, lm, self.crop)
        img        = cv2.resize(img, (112, 112))
        blob       = cv2.dnn.blobFromImage(img, 1.0 / 255, mean=0, swapRB=False)
        out        = self.session.run([self.output_name], {self.input_name: blob})
        spoof_prob = float(out[0][0][0])
        return spoof_prob > self.threshold, spoof_prob, img


# ═══════════════════════════════════════════════
# ACTIVE — HEAD POSE CHALLENGE
# ═══════════════════════════════════════════════

def _active_headpose_challenge(cap) -> dict:
    if cap is None or not cap.isOpened():
        return {"passed": False, "reason": "Camera unavailable"}

    from modules.detector import detect_faces

    print("[LIVENESS] Active: calibrating neutral yaw...")
    neutral_samples, deadline = [], time.time() + 5.0
    while len(neutral_samples) < 12 and time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        faces = detect_faces(frame)
        if faces:
            neutral_samples.append(_estimate_yaw(max(faces, key=lambda f: f["score"])["landmarks"]))

    if len(neutral_samples) < 6:
        return {"passed": False, "reason": "Calibration failed"}

    neutral_yaw = float(np.median(neutral_samples))

    print("[LIVENESS] Active: waiting for LEFT/RIGHT head turn...")
    deadline = time.time() + LIVENESS_HEADPOSE_TIMEOUT
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        faces = detect_faces(frame)
        if faces:
            yaw = _estimate_yaw(max(faces, key=lambda f: f["score"])["landmarks"]) - neutral_yaw
            if abs(yaw) > LIVENESS_HEADPOSE_YAW_THRESHOLD:
                return {"passed": True}

    return {"passed": False, "reason": "Head pose challenge timed out"}


# ═══════════════════════════════════════════════
# YAW ESTIMATION
# ═══════════════════════════════════════════════

def _estimate_yaw(landmarks) -> float:
    if isinstance(landmarks, dict):
        le   = np.array(landmarks["left_eye"])
        re   = np.array(landmarks["right_eye"])
        nose = np.array(landmarks["nose"])
    else:
        lm   = np.array(landmarks)
        re, le, nose = lm[0], lm[1], lm[2]
    mid      = (le + re) / 2.0
    eye_dist = np.linalg.norm(le - re)
    return float((nose[0] - mid[0]) / eye_dist) if eye_dist > 1e-6 else 0.0


# ═══════════════════════════════════════════════
# MICRO-MOTION  (anti-replay guard)
# ═══════════════════════════════════════════════

def _motion_crop(image: np.ndarray, bbox) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    return cv2.resize(crop, (64, 64)) if crop.size else None


def _check_micro_motion(crop: np.ndarray) -> bool:
    global _motion_buffer
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _motion_buffer.append(gray)
    if len(_motion_buffer) > _MOTION_WINDOW:
        _motion_buffer.pop(0)
    if len(_motion_buffer) < _MOTION_WINDOW:
        return True
    diffs = [np.mean(cv2.absdiff(_motion_buffer[i], _motion_buffer[i+1]))
             for i in range(len(_motion_buffer) - 1)]
    return float(np.mean(diffs)) > _MOTION_THRESHOLD


def _reset_buffers():
    _motion_buffer.clear()
