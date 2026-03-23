"""
config.py — Loads all configuration from application.properties.

Priority order (highest wins):
  1. Environment variables   — for secrets in production
  2. application.properties  — for everything else
  3. Hardcoded defaults here — fallback if key is missing from properties file

The properties file is looked for in this order:
  1. Path in env var APP_PROPERTIES (e.g. APP_PROPERTIES=/etc/face/app.properties)
  2. ./application.properties  (same directory as this file)
"""

import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# Properties file loader
# ─────────────────────────────────────────────

def _load_properties(path: str) -> dict[str, str]:
    """
    Parse a .properties file into a flat dict.
    Ignores blank lines and lines starting with # .
    Strips inline comments (anything after unquoted #).
    """
    props: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.split("#")[0].strip()  # strip inline comment
                props[key.strip()] = value
    except FileNotFoundError:
        pass  # No properties file — all values fall back to defaults
    return props


def _find_properties_file() -> str:
    env_path = os.getenv("APP_PROPERTIES")
    if env_path:
        return env_path
    return str(Path(__file__).parent / "application.properties")


_PROPS = _load_properties(_find_properties_file())


# ─────────────────────────────────────────────
# Typed getter helpers
# ─────────────────────────────────────────────

def _str(key: str, default: str = "") -> str:
    """env var → properties file → default."""
    env_key = key.upper().replace(".", "_")
    return os.getenv(env_key, _PROPS.get(key, default)).strip()

def _bool(key: str, default: bool = False) -> bool:
    return _str(key, str(default)).lower() in {"1", "true", "yes", "on"}

def _int(key: str, default: int = 0) -> int:
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default

def _float(key: str, default: float = 0.0) -> float:
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default

def _floats(key: str, default: list) -> list:
    raw = _str(key, "")
    if not raw:
        return default
    try:
        return [float(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError:
        return default

def _ints(key: str, default: list) -> list:
    raw = _str(key, "")
    if not raw:
        return default
    try:
        return [int(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError:
        return default

def _color(key: str, default: tuple) -> tuple:
    """Parse 'B, G, R' → (B, G, R) int tuple."""
    vals = _ints(key, list(default))
    return tuple(vals[:3]) if len(vals) >= 3 else default


# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD        = _float("detection.confidence_threshold", 0.80)
DETECTION_CONFIDENCE_PASSES = _floats("detection.confidence_passes", [0.80, 0.60, 0.45])
DETECTION_SCALE_PASSES      = _floats("detection.scale_passes", [1.0])
DETECTION_NMS_IOU           = _float("detection.nms_iou", 0.35)

LIVE_CAPTURE_WIDTH         = _int("camera.capture_width", 640)
LIVE_CAPTURE_HEIGHT        = _int("camera.capture_height", 480)
LIVE_DETECT_EVERY_N_FRAMES = _int("camera.detect_every_n_frames", 3)
LIVE_MAX_DETECT_WIDTH      = _int("camera.max_detect_width", 640)


# ─────────────────────────────────────────────
# CVLFace model
# ─────────────────────────────────────────────

CVLFACE_REPO_ID   = _str("cvlface.repo_id",  "minchul/cvlface_adaface_ir101_webface12m")
CVLFACE_CACHE_DIR = _str("cvlface.cache_dir", "~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m")


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────

EMBEDDING_DIM = _int("embedding.dim", 512)
_input        = _ints("embedding.input_size", [112, 112])
INPUT_SIZE    = (int(_input[0]), int(_input[1]))


# ─────────────────────────────────────────────
# Match thresholds
# ─────────────────────────────────────────────

MATCH_THRESHOLDS = {
    "cvlface": _float("match.threshold.cvlface", 0.55),
    "fallback": _float("match.threshold.fallback", 0.94),
}


# ─────────────────────────────────────────────
# Auto-update
# ─────────────────────────────────────────────

AUTO_UPDATE_ENABLED = _bool("auto_update.enabled", True)

AUTO_UPDATE_MIN_CONFIDENCE = {
    "cvlface": _float("auto_update.min_confidence.cvlface", 0.65),
    "fallback": _float("auto_update.min_confidence.fallback", 0.96),
}

AUTO_UPDATE_HIGH_CONFIDENCE = {
    "cvlface": _float("auto_update.high_confidence.cvlface", 0.90),
    "fallback": _float("auto_update.high_confidence.fallback", 0.99),
}

AUTO_UPDATE_MAX_SIM_TO_EXISTING = _float("auto_update.max_sim_to_existing", 0.98)


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────

QDRANT_PATH       = _str("qdrant.path",       "./qdrant_storage")
QDRANT_COLLECTION = _str("qdrant.collection", "faces")

# SQL person registry (PostgreSQL preferred, SQLite fallback)
SQL_BACKEND      = _str("sql.backend", "sqlite").lower()
SQL_DATABASE_URL = _str("sql.database_url", "")
SQL_SQLITE_PATH  = _str("sql.sqlite_path", "./persons.sqlite3")


# ─────────────────────────────────────────────
# Dynamic scaling tiers
# ─────────────────────────────────────────────

def _parse_scale_tiers() -> list:
    """
    Read scale_tier.0 .. scale_tier.N from properties.
    Each line: min, max, strategy, cap, hnsw_m, hnsw_ef_construct, search_ef
    Leave max blank on the last tier to mean unlimited (→ None).
    """
    _DEFAULTS = [
        (0,     99,   "single", 1,  8,  100, 64),
        (100,   999,  "multi",  5,  16, 200, 128),
        (1000,  9999, "hybrid", 3,  32, 300, 256),
        (10000, None, "mean",   1,  64, 400, 512),
    ]
    tiers, i = [], 0
    while True:
        raw = _str(f"scale_tier.{i}", "")
        if not raw:
            break
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 7:
            break
        tiers.append((
            int(parts[0]),
            int(parts[1]) if parts[1] else None,
            parts[2],
            int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]),
        ))
        i += 1
    return tiers or _DEFAULTS

SCALE_TIERS = _parse_scale_tiers()


# ─────────────────────────────────────────────
# Face quality gate
# ─────────────────────────────────────────────

MIN_FACE_SIZE   = _int("quality.min_face_size", 90)
MIN_BLUR_SCORE  = _float("quality.min_blur_score", 20.0)
MAX_YAW_DEGREES = _float("quality.max_yaw_degrees", 60.0)

# Aliases kept for backward compatibility with existing module imports
MIN_FACE_SIZE_PX = MIN_FACE_SIZE
BLUR_THRESHOLD   = MIN_BLUR_SCORE


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

PREPROCESS_MIN_DIM        = _int("preprocess.min_dim", 400)
PREPROCESS_ENABLE_CLAHE   = _bool("preprocess.enable_clahe", True)
PREPROCESS_ENABLE_DENOISE = _bool("preprocess.enable_denoise", False)
PREPROCESS_ENABLE_GAMMA   = _bool("preprocess.enable_gamma", True)
PREPROCESS_ENABLE_SHARPEN = _bool("preprocess.enable_sharpen", False)


# ─────────────────────────────────────────────
# Liveness detection
# ─────────────────────────────────────────────

LIVENESS_ENABLED                 = _bool("liveness.enabled", True)
LIVENESS_PASSIVE_REAL_THRESHOLD  = _float("liveness.passive.real_threshold", 0.55)
LIVENESS_PASSIVE_SPOOF_THRESHOLD = _float("liveness.passive.spoof_threshold", 0.50)

LIVENESS_ACTIVE_ENABLED          = _bool("liveness.active.enabled", True)
LIVENESS_REQUIRE_ACTIVE          = _bool("liveness.active.require_always", False)
LIVENESS_HEADPOSE_TIMEOUT        = _int("liveness.active.headpose_timeout_sec", 8)
LIVENESS_HEADPOSE_YAW_THRESHOLD  = _float("liveness.active.headpose_yaw_threshold", 0.25)


# ─────────────────────────────────────────────
# Display (CLI only)
# ─────────────────────────────────────────────

SHOW_LANDMARKS  = _bool("display.show_landmarks", True)
SHOW_CONFIDENCE = _bool("display.show_confidence", True)
DISPLAY_MAX_W   = _int("display.max_width", 900)
DISPLAY_MAX_H   = _int("display.max_height", 700)
BBOX_COLOR      = _color("display.bbox_color", (0, 255, 0))

LANDMARK_COLORS = {
    "right_eye":   _color("display.landmark_color.right_eye",   (0, 255, 0)),
    "left_eye":    _color("display.landmark_color.left_eye",    (0, 255, 0)),
    "nose":        _color("display.landmark_color.nose",        (0, 0, 255)),
    "mouth_right": _color("display.landmark_color.mouth_right", (255, 0, 0)),
    "mouth_left":  _color("display.landmark_color.mouth_left",  (255, 0, 0)),
}


# ─────────────────────────────────────────────
# Alignment template  (fixed — not user-configurable)
# ─────────────────────────────────────────────

LANDMARK_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ─────────────────────────────────────────────
# API authentication
# Env vars API_AUTH_ENABLED and API_KEYS always override properties values.
# ─────────────────────────────────────────────

API_AUTH_ENABLED = (
    os.getenv("API_AUTH_ENABLED", _str("api.auth.enabled", "true"))
    .strip().lower() in {"1", "true", "yes", "on"}
)

API_KEYS = [
    k.strip()
    for k in os.getenv("API_KEYS", _str("api.auth.keys", "")).split(",")
    if k.strip()
]

# Read-only keys — can call match/status/detect/list but NOT register or delete.
# Set via env var API_KEYS_READONLY or application.properties api.auth.keys_readonly
API_KEYS_READONLY = [
    k.strip()
    for k in os.getenv("API_KEYS_READONLY", _str("api.auth.keys_readonly", "")).split(",")
    if k.strip()
]


# ─────────────────────────────────────────────
# Geotagging
# ─────────────────────────────────────────────

GEOTAGGING_ENABLED: bool   = _bool("geo.enabled", True)
GEOCODER_BACKEND: str      = _str("geo.geocoder_backend", "nominatim")
GEOCODER_API_KEY: str      = os.getenv("GEOCODER_API_KEY", _str("geo.geocoder_api_key", ""))
NOMINATIM_USER_AGENT: str  = _str("geo.nominatim_user_agent", "face_attendance_pipeline/1.0")
GEOCODE_CACHE_TTL: int     = _int("geo.cache_ttl_sec", 3600)
GEOCODE_CACHE_MAX: int     = _int("geo.cache_max_entries", 512)
GEOCODE_TIMEOUT: int       = _int("geo.geocode_timeout_sec", 5)
CACHE_GRID_DEG: float      = _float("geo.cache_grid_deg", 0.0001)


# ─────────────────────────────────────────────
# Geofence zones  (built from properties)
# ─────────────────────────────────────────────

GEOFENCE_ENFORCE: bool = _bool("geofence.enforce", False)


def _parse_geofence_zones() -> list:
    """
    Collect all geofence.zone.<id>.* keys and build
    the GEOFENCE_ZONES list that geotagging.py expects.
    """
    prefix = "geofence.zone."
    zone_ids: set = set()
    for key in _PROPS:
        if key.startswith(prefix):
            parts = key[len(prefix):].split(".")
            if parts:
                zone_ids.add(parts[0])

    zones = []
    for zone_id in sorted(zone_ids):
        def zstr(field: str, default: str = "", _zid: str = zone_id) -> str:
            return _str(f"{prefix}{_zid}.{field}", default)

        ztype = zstr("type", "circle")

        if ztype == "circle":
            try:
                zones.append({
                    "id":       zone_id,
                    "name":     zstr("name", zone_id),
                    "type":     "circle",
                    "center":   [float(zstr("lat")), float(zstr("lon"))],
                    "radius_m": float(zstr("radius", "100")),
                })
            except ValueError:
                pass

        elif ztype == "polygon":
            raw_points = zstr("points", "")
            try:
                points = [
                    [float(c.split(",")[0].strip()), float(c.split(",")[1].strip())]
                    for c in raw_points.split("|")
                    if "," in c
                ]
                if len(points) >= 3:
                    zones.append({
                        "id":      zone_id,
                        "name":    zstr("name", zone_id),
                        "type":    "polygon",
                        "polygon": points,
                    })
            except (ValueError, IndexError):
                pass

    return zones


GEOFENCE_ZONES: list = _parse_geofence_zones()


# ─────────────────────────────────────────────
# Attendance cooldown store
# ─────────────────────────────────────────────

ATTENDANCE_COOLDOWN_SEC = _int("attendance.cooldown_sec", 1800)
ATTENDANCE_REDIS_URL    = _str("attendance.redis_url", "redis://localhost:6379/0")