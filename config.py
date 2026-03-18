"""
config.py — Central configuration for the face recognition pipeline.
"""

import os
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
MIN_FACE_SIZE = 90
MIN_BLUR_SCORE = 20.0
MAX_YAW_DEGREES = 60.0

MIN_FACE_SIZE_PX = MIN_FACE_SIZE
BLUR_THRESHOLD = MIN_BLUR_SCORE


# ─────────────────────────────────────────────
# Liveness Detection
# ─────────────────────────────────────────────

# Passive spoof detection (SilentFace)
LIVENESS_PASSIVE_REAL_THRESHOLD = 0.60
LIVENESS_PASSIVE_SPOOF_THRESHOLD = 0.50
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

# ─────────────────────────────────────────────
# Preprocessing tuning
# ─────────────────────────────────────────────
PREPROCESS_MIN_DIM = 400
PREPROCESS_ENABLE_CLAHE = True
PREPROCESS_ENABLE_DENOISE = False
PREPROCESS_ENABLE_GAMMA = True
PREPROCESS_ENABLE_SHARPEN = False


# ─────────────────────────────────────────────
# API Authentication
# ─────────────────────────────────────────────
# Comma-separated keys via env, e.g.:
#   API_KEYS=dev_key_1,dev_key_2
API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
API_KEYS = [
    k.strip()
    for k in os.getenv("API_KEYS", "developer_key").split(",")
    if k.strip()
]

# ─────────────────────────────────────────────
# Geotagging
# ─────────────────────────────────────────────
 
# Master switch — set False to disable all geotagging with zero overhead
GEOTAGGING_ENABLED: bool = True
 
# Reverse geocoding backend: "nominatim" | "google" | "here"
# - nominatim : free, OSM-backed, 1 req/s rate limit (handled by cache)
# - google    : requires GEOCODER_API_KEY, higher quota, better accuracy
# - here      : requires GEOCODER_API_KEY, enterprise-grade
GEOCODER_BACKEND: str = "nominatim"
 
# API key — only used when GEOCODER_BACKEND is "google" or "here"
# Set via environment variable in production:
#   GEOCODER_API_KEY=your_key_here
GEOCODER_API_KEY: str = os.getenv("GEOCODER_API_KEY", "")
 
# Nominatim requires a unique user-agent string per OSM policy
NOMINATIM_USER_AGENT: str = "face_attendance_pipeline/1.0"
 
# How long to cache a reverse-geocode result (seconds).
# Fixed cameras see the same coordinates repeatedly — cache aggressively.
GEOCODE_CACHE_TTL: int = 3600           # 1 hour
 
# Maximum number of cached geocode entries (one per ~11 m grid cell)
GEOCODE_CACHE_MAX: int = 512
 
# HTTP timeout for geocoding requests (seconds)
GEOCODE_TIMEOUT: int = 5
 
# Coordinate grid resolution for cache key bucketing (degrees).
# 0.0001° ≈ 11 metres — fine for fixed-camera setups.
CACHE_GRID_DEG: float = 0.0001
 
# ── Geofence zones ───────────────────────────────────────────────────────
# List of allowed zones. Each zone must be either:
#
#   Circular zone:
#     { "id": "office", "name": "Main Office",
#       "type": "circle", "center": [LAT, LON], "radius_m": 200 }
#
#   Polygon zone (arbitrary boundary):
#     { "id": "campus", "name": "Campus",
#       "type": "polygon",
#       "polygon": [[LAT1,LON1], [LAT2,LON2], [LAT3,LON3], ...] }
#
# Leave empty [] to disable geofencing entirely.
GEOFENCE_ZONES: list = [{ "id": "office", 
                         "name": "Main Office",
                         "type": "circle", 
                         "center": [11.019751378519258, 76.97866478828647], 
                         "radius_m": 50},
                        ]
 
# True  → reject /match and /attendance (HTTP 403) when outside all zones
# False → log geofence status in response but still allow the request
GEOFENCE_ENFORCE: bool = False
