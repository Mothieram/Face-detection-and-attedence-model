"""
config.py — Central configuration for the face recognition pipeline.
"""

import numpy as np

# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80
DETECTION_CONFIDENCE_PASSES = [0.80, 0.60, 0.45]
DETECTION_SCALE_PASSES = [1.0]
DETECTION_NMS_IOU = 0.35

# Live camera performance tuning
LIVE_CAPTURE_WIDTH = 640
LIVE_CAPTURE_HEIGHT = 480
LIVE_DETECT_EVERY_N_FRAMES = 3
LIVE_MAX_DETECT_WIDTH = 640


# ─────────────────────────────────────────────
# CVLFace Model
# ─────────────────────────────────────────────
CVLFACE_REPO_ID = "minchul/cvlface_adaface_ir101_webface12m"
CVLFACE_CACHE_DIR = "~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m"


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────
EMBEDDING_DIM = 512
INPUT_SIZE = (112, 112)


# ─────────────────────────────────────────────
# Match thresholds
# ─────────────────────────────────────────────
MATCH_THRESHOLDS = {
    "cvlface": 0.55,
    "fallback": 0.94,
}


# ─────────────────────────────────────────────
# Auto-update
# ─────────────────────────────────────────────
AUTO_UPDATE_ENABLED = True

AUTO_UPDATE_MIN_CONFIDENCE = {
    "cvlface": 0.65,
    "fallback": 0.96,
}

AUTO_UPDATE_HIGH_CONFIDENCE = {
    "cvlface": 0.90,
    "fallback": 0.99,
}

AUTO_UPDATE_MAX_SIM_TO_EXISTING = 0.98


# ─────────────────────────────────────────────
# Dynamic scaling tiers
# ─────────────────────────────────────────────
SCALE_TIERS = [
    (0, 99, "single", 1, 8, 100, 64),
    (100, 999, "multi", 5, 16, 200, 128),
    (1000, 9999, "hybrid", 3, 32, 300, 256),
    (10000, None, "mean", 1, 64, 400, 512),
]


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
QDRANT_PATH = "./qdrant_storage"
QDRANT_COLLECTION = "faces"


# ─────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────
SHOW_LANDMARKS = True
SHOW_CONFIDENCE = True
DISPLAY_MAX_W = 900
DISPLAY_MAX_H = 700
BBOX_COLOR = (0, 255, 0)

LANDMARK_COLORS = {
    "right_eye": (0, 255, 0),
    "left_eye": (0, 255, 0),
    "nose": (0, 0, 255),
    "mouth_right": (255, 0, 0),
    "mouth_left": (255, 0, 0),
}


# ─────────────────────────────────────────────
# Alignment template
# ─────────────────────────────────────────────
LANDMARK_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ─────────────────────────────────────────────
# Face quality gate
# ─────────────────────────────────────────────
MIN_FACE_SIZE = 40
MIN_BLUR_SCORE = 20.0
MAX_YAW_DEGREES = 45.0

MIN_FACE_SIZE_PX = MIN_FACE_SIZE
BLUR_THRESHOLD = MIN_BLUR_SCORE


# ─────────────────────────────────────────────
# Liveness Detection
# ─────────────────────────────────────────────

# Passive spoof detection (SilentFace)
LIVENESS_PASSIVE_REAL_THRESHOLD = 0.70
LIVENESS_PASSIVE_SPOOF_THRESHOLD = 0.40
# For SilentFace ONNX exports with 3 classes, class index 2 is typically "real".
LIVENESS_PASSIVE_REAL_CLASS_INDEX = 2

LIVENESS_PASSIVE_MODEL_PATH = "~/.liveness_cache/2.7_80x80_MiniFASNetV2.onnx"


# ─────────────────────────────────────────────
# Active Liveness (Challenge-Response)
# ─────────────────────────────────────────────

# Master switch
LIVENESS_ENABLED = True
LIVENESS_ACTIVE_ENABLED = True
# Security mode: require active challenge even when passive score is high.
LIVENESS_REQUIRE_ACTIVE = False


# Head pose challenge
LIVENESS_HEADPOSE_TIMEOUT = 8
LIVENESS_HEADPOSE_YAW_THRESHOLD = 0.25


# Blink detection (optional future extension)
LIVENESS_BLINK_TIMEOUT_SEC = 6
LIVENESS_BLINK_EAR_CLOSED = 0.23
LIVENESS_BLINK_EAR_OPEN = 0.30
LIVENESS_BLINK_CONSEC_FRAMES = 2


# ─────────────────────────────────────────────
# Preprocessing tuning
# ─────────────────────────────────────────────
PREPROCESS_MIN_DIM = 400
PREPROCESS_ENABLE_CLAHE = True
PREPROCESS_ENABLE_DENOISE = False
PREPROCESS_ENABLE_GAMMA = True
PREPROCESS_ENABLE_SHARPEN = False
