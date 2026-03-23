"""
api.py — FastAPI server for the Face Recognition Attendance Pipeline.

All endpoints are under /v1/ prefix.

Route map (noun-based, REST-compliant):
  GET    /v1/status                  → system health
  POST   /v1/faces/detections        → detect faces in image
  POST   /v1/faces/liveness          → liveness check on largest face
  POST   /v1/persons                 → register a new person  (201 Created)
  POST   /v1/faces/matches           → match faces against DB
  POST   /v1/attendance/records      → log attendance
  GET    /v1/persons                 → list registered persons (paginated)
  GET    /v1/persons/{name}          → check if person exists
  DELETE /v1/persons/{name}          → delete person          (204 No Content)
  GET    /v1/db/tier                 → DB scaling tier info

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import sys
import time
import os
import uuid
import contextlib
import logging
import logging.handlers
from contextlib import asynccontextmanager

# Keep Torch model cache inside project when TORCH_HOME is not predefined.
# This avoids permission issues on restricted Windows environments.
os.environ.setdefault(
    "TORCH_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache"),
)

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends, Security, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import (
    MATCH_THRESHOLDS,
    AUTO_UPDATE_ENABLED,
    AUTO_UPDATE_MIN_CONFIDENCE,
    AUTO_UPDATE_HIGH_CONFIDENCE,
    LIVENESS_ENABLED,
    LIVENESS_ACTIVE_ENABLED,
    API_AUTH_ENABLED,
    API_KEYS,
    API_KEYS_READONLY,
    GEOFENCE_ENFORCE,
)
from modules.preprocessor import preprocess_image
from modules.detector     import detect_faces, _get_retinaface_model
from modules.aligner      import align_face, check_face_quality
from modules.liveness     import check_liveness, _load_all_sessions
from modules.embedder     import generate_embedding, is_cvlface_loaded, _load_cvlface
from modules.database     import (
    save_record,
    auto_update_add_template,
    current_tier_info,
    delete_record_by_person_id,
    load_db,
)
from modules.matcher      import match_face, is_match
from modules.geotagging   import geotag_event
from modules.persons_db   import (
    create_person,
    delete_person_by_name,
    get_person_by_id,
    get_person_by_name,
    list_persons,
)

# ─────────────────────────────────────────────
# Logging — stdout + rotating file (10 MB × 5 backups)
# ─────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("face_api")
    log.setLevel(logging.INFO)
    if not log.handlers:
        # Console
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
        # Rotating file — sits next to api.py
        _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_api.log")
        fh = logging.handlers.RotatingFileHandler(
            _log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log

logger = _setup_logging()

# ─────────────────────────────────────────────
# Attendance duplicate prevention
# Tracks last-seen timestamp per person per session.
# Same person cannot be counted twice within ATTENDANCE_COOLDOWN_SEC.
# ─────────────────────────────────────────────
import threading as _threading

_attendance_lock  = _threading.Lock()
_attendance_log: dict[str, float] = {}   # person_name → last_logged unix timestamp
ATTENDANCE_COOLDOWN_SEC = 1800            # 30 minutes — configurable here

def _can_log_attendance(name: str) -> bool:
    """Returns True if enough time has passed since this person was last logged."""
    now = time.time()
    with _attendance_lock:
        last = _attendance_log.get(name, 0.0)
        if now - last >= ATTENDANCE_COOLDOWN_SEC:
            _attendance_log[name] = now
            return True
        return False

def _clear_attendance_log() -> None:
    """Call at the start of a new attendance session to reset all cooldowns."""
    with _attendance_lock:
        _attendance_log.clear()


# ─────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# RBAC — two key tiers
#
# API_KEYS       → admin keys: full access (register, delete, match, attend)
# API_KEYS_READONLY → read-only keys: status, detect, liveness, list, match only
#                    Cannot register or delete persons.
#
# Set via application.properties or environment variables:
#   api.auth.keys          = admin_key_1, admin_key_2
#   api.auth.keys_readonly = readonly_key_1
# ─────────────────────────────────────────────

_API_KEYS_ADMIN    = set(API_KEYS)
_API_KEYS_READONLY = set(API_KEYS_READONLY)
_API_KEYS_ALL      = _API_KEYS_ADMIN | _API_KEYS_READONLY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key(api_key: Optional[str] = Security(_api_key_header)) -> Optional[str]:
    """Extract and validate the API key — returns the key string or None."""
    return api_key


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """Dependency: any valid key (admin or readonly) is accepted."""
    if not API_AUTH_ENABLED:
        return
    if not api_key:
        raise HTTPException(status_code=401, detail={"error":"MISSING_API_KEY","message":"No API key provided. Add header: X-API-Key: <your_key>","field":"X-API-Key","code":"MISSING_API_KEY"})
    if api_key not in _API_KEYS_ALL:
        raise HTTPException(status_code=403, detail={"error":"INVALID_API_KEY","message":"The provided API key is not recognised. Check your X-API-Key header value.","field":"X-API-Key","code":"INVALID_API_KEY"})


async def require_admin_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """Dependency: only admin keys are accepted. Readonly keys get 403."""
    if not API_AUTH_ENABLED:
        return
    if not api_key:
        raise HTTPException(status_code=401, detail={"error":"MISSING_API_KEY","message":"No API key provided. Add header: X-API-Key: <your_key>","field":"X-API-Key","code":"MISSING_API_KEY"})
    if api_key not in _API_KEYS_ALL:
        raise HTTPException(status_code=403, detail={"error":"INVALID_API_KEY","message":"The provided API key is not recognised. Check your X-API-Key header value.","field":"X-API-Key","code":"INVALID_API_KEY"})
    if api_key not in _API_KEYS_ADMIN:
        raise HTTPException(status_code=403, detail={"error":"INSUFFICIENT_PERMISSIONS","message":"This action requires an admin API key. Your key has read-only access.","field":"X-API-Key","code":"INSUFFICIENT_PERMISSIONS"})


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_load_models()
    yield


def _startup_load_models() -> None:
    print("[STARTUP] Loading RetinaFace detector...", flush=True)
    try:
        model = _get_retinaface_model()
        if model is None:
            print("[STARTUP] [WARN] RetinaFace unavailable - using Haar fallback", flush=True)
        else:
            print("[STARTUP] [OK] RetinaFace ready", flush=True)
    except Exception as exc:
        msg = f"RetinaFace failed to load: {exc}"
        print(f"[STARTUP] [FAIL] {msg}", flush=True)
        print("[STARTUP] [WARN] Continuing startup in degraded mode", flush=True)

    print("[STARTUP] Loading CVLFace embedder (AdaFace IR-101)...", flush=True)
    try:
        model = _load_cvlface()
        if model is None:
            print("[STARTUP] [WARN] CVLFace unavailable - using fallback embeddings", flush=True)
        else:
            print("[STARTUP] [OK] CVLFace ready", flush=True)
    except Exception as exc:
        msg = f"CVLFace failed to load: {exc}"
        print(f"[STARTUP] [FAIL] {msg}", flush=True)
        print("[STARTUP] [WARN] Continuing startup with fallback embeddings", flush=True)

    print("[STARTUP] Loading liveness ensemble (ICM2O · IOM2C · modelrgb · SASF)...", flush=True)
    try:
        _load_all_sessions()
        from modules.liveness import _models
        loaded = [k for k, v in _models.items() if v is not None]
        failed = [k for k, v in _models.items() if v is None]
        if loaded:
            print(f"[STARTUP] [OK] Liveness ready - {loaded}", flush=True)
        if failed:
            print(f"[STARTUP] [WARN] Some liveness models failed - {failed}", flush=True)
        if not loaded:
            print("[STARTUP] [WARN] All liveness models unavailable", flush=True)
    except Exception as exc:
        print(f"[STARTUP] [WARN] Liveness load error: {exc}", flush=True)

    print("[STARTUP] All models loaded — server ready.\n", flush=True)


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Face Recognition Attendance API",
    description=(
        "REST interface for the face recognition pipeline.\n\n"
        "All endpoints are under `/v1/`. "
        "Image endpoints accept **multipart/form-data** with an `image` field (JPEG / PNG).\n\n"
        "Authenticate with header: `X-API-Key: <your_key>`"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/styles", StaticFiles(directory=os.path.join(_BASE_DIR, "styles")), name="styles")


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(os.path.join(_BASE_DIR, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Inline SVG favicon — stops the 404 log noise on every browser tab open.
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<circle cx="16" cy="16" r="16" fill="#1a1a2e"/>'
        '<circle cx="16" cy="13" r="6" fill="#4ade80"/>'
        '<ellipse cx="16" cy="26" rx="9" ry="5" fill="#4ade80"/>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=(
        r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
        r"|https://[a-zA-Z0-9\-]+\.ngrok(-free)?\.app"
        r"|https://[a-zA-Z0-9\-]+\.ngrok\.io"
        r"|https://[a-zA-Z0-9\-]+\.loca\.lt"
        r"|https://[a-zA-Z0-9\-]+\.trycloudflare\.com"
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Upload size limit ─────────────────────────────────────────────────────
# Rejects requests whose Content-Length header exceeds MAX_UPLOAD_BYTES.
# Protects ML inference endpoints from oversized payloads.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        return _error_response(
            status_code=413,
            error="FILE_TOO_LARGE",
            message=f"Upload exceeds the 10 MB limit. Compress or resize the image and try again.",
            field="image",
        )
    return await call_next(request)


# ── ngrok interstitial bypass ─────────────────────────────────────────────
# Adds ngrok-skip-browser-warning to every response so the initial HTML
# page load also bypasses the ngrok warning page (JS api.js handles fetch calls).
@app.middleware("http")
async def ngrok_skip_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ─────────────────────────────────────────────
# Error response helpers
#
# Every error returns the same envelope:
# {
#   "error"  : machine-readable error type  (e.g. "FACE_NOT_DETECTED")
#   "message": human-readable explanation   (actionable, not generic)
#   "field"  : which field caused the error (null when not field-specific)
#   "code"   : stable error code string     (same as "error", for clients)
# }
# ─────────────────────────────────────────────

def _error_response(
    status_code: int,
    error: str,
    message: str,
    field: str | None = None,
) -> JSONResponse:
    """Build the standard error envelope used by all exception handlers."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error":   error,
            "message": message,
            "field":   field,
            "code":    error,          # duplicate of error — keeps parity with the article schema
        },
    )


# ── HTTP exceptions raised by our own code ───────────────────────────────
# Every raise HTTPException(...) goes through here.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # exc.detail is either a plain string or a dict we built with {error, message, field}
    if isinstance(exc.detail, dict):
        return _error_response(
            status_code=exc.status_code,
            error=exc.detail.get("error", "ERROR"),
            message=exc.detail.get("message", ""),
            field=exc.detail.get("field"),
        )
    # Plain string detail — derive error code from status code
    _STATUS_CODES = {
        400: "BAD_REQUEST",   401: "UNAUTHORIZED",  403: "FORBIDDEN",
        404: "NOT_FOUND",     409: "CONFLICT",       422: "UNPROCESSABLE",
        429: "RATE_LIMITED",  500: "INTERNAL_ERROR",
    }
    return _error_response(
        status_code=exc.status_code,
        error=_STATUS_CODES.get(exc.status_code, "ERROR"),
        message=str(exc.detail),
    )


# ── Pydantic / FastAPI validation errors (wrong type, missing field) ──────
# These are NOT StarletteHTTPExceptions — they need their own handler.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    # Pull the first error's field and reason to surface in the top-level message
    first = errors[0] if errors else {}
    loc   = first.get("loc", [])
    # loc looks like ("body", "name") — skip the "body" prefix
    field = str(loc[-1]) if len(loc) > 1 else (str(loc[0]) if loc else None)
    msg   = first.get("msg", "Validation failed")
    return _error_response(
        status_code=422,
        error="VALIDATION_ERROR",
        message=f"Invalid value for '{field}': {msg}" if field else msg,
        field=field,
    )


# ── Rate limit exceeded ───────────────────────────────────────────────────
# slowapi's default handler returns plain text — we override with our envelope.
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    retry = getattr(exc, "retry_after", None)
    msg = (
        f"Rate limit exceeded. You can retry after {retry} seconds."
        if retry else
        "Rate limit exceeded. Please slow down and retry shortly."
    )
    return _error_response(status_code=429, error="RATE_LIMITED", message=msg)


# ── Unhandled Python exceptions → clean 500 ───────────────────────────────
# Prevents raw tracebacks leaking to the client if something unexpected crashes.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return _error_response(
        status_code=500,
        error="INTERNAL_ERROR",
        message="An unexpected server error occurred. The issue has been logged.",
    )


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class FaceInfo(BaseModel):
    face_id: str
    score: float
    bbox: list[int]
    landmarks: dict
    quality: Optional[dict] = None

class LivenessResult(BaseModel):
    passed: bool
    method: str
    score: float
    reason: str
    model_scores: Optional[dict] = None

class RegisterResponse(BaseModel):
    success: bool
    name: str
    face_id: str
    liveness: Optional[LivenessResult] = None
    geo: Optional[dict] = None
    embedding_mode: str
    message: str

class MatchResult(BaseModel):
    face_id: str
    matched: bool
    name: str
    score: float
    liveness: Optional[LivenessResult] = None
    geo: Optional[dict] = None
    embedding_mode: str
    auto_updated: bool = False
    bbox: Optional[list[int]] = None
    landmarks: Optional[dict] = None
    quality: Optional[dict] = None

class MatchResponse(BaseModel):
    results: list[MatchResult]
    total_faces: int
    matched_count: int
    geo: Optional[dict] = None

class DetectResponse(BaseModel):
    faces: list[FaceInfo]
    total: int

class DeleteResponse(BaseModel):
    success: bool
    name: str
    message: str

class PersonRecord(BaseModel):
    name: str
    display_name: str
    embedding_mode: str
    template_type: str
    bbox: list
    landmarks: dict

class PersonsListResponse(BaseModel):
    records: list[PersonRecord]
    total: int
    limit: int
    offset: int
    has_more: bool

class AttendanceResult(BaseModel):
    timestamp: int
    present: list[str]
    present_count: int
    unknown_count: int
    spoofed_count: int
    total_faces: int
    geo: Optional[dict] = None
    detail: list[dict]

class StatusResponse(BaseModel):
    cvlface_loaded: bool
    liveness_enabled: bool
    active_liveness_enabled: bool
    db_tier: dict
    match_thresholds: dict


# ─────────────────────────────────────────────
# Shared pipeline helpers
# ─────────────────────────────────────────────

def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail={"error":"INVALID_IMAGE","message":"Cannot decode the uploaded file. Send a valid JPEG or PNG image in the 'image' field.","field":"image","code":"INVALID_IMAGE"})
    return img


def _silent_detect(image: np.ndarray) -> list:
    with contextlib.redirect_stdout(io.StringIO()):
        return detect_faces(image)


def _run_pipeline(image_bgr: np.ndarray) -> tuple[np.ndarray, list]:
    preprocessed = preprocess_image(image_bgr, fast=False)
    faces = _silent_detect(preprocessed)
    return preprocessed, faces


def _quality_filter(raw: np.ndarray, faces: list) -> tuple[list, list]:
    passed, rejected = [], []
    for face in faces:
        q = check_face_quality(raw, face)
        face["quality"] = q
        if q["passed"]:
            passed.append(face)
        else:
            rejected.append({"face_id": face["face_id"], "reason": q["reason"]})
    return passed, rejected


def _liveness_check(raw: np.ndarray, face: dict,
                    camera_index: int = 0, skip_active: bool = False) -> dict:
    return check_liveness(
        raw,
        face,
        camera_index=camera_index,
        active_enabled_override=False if skip_active else None,
    )


def _resolve_identity_name(person_ref: str) -> str:
    if not person_ref or person_ref in ("No Match", "No DB records"):
        return person_ref
    try:
        uuid.UUID(person_ref)
    except ValueError:
        return person_ref
    person = get_person_by_id(person_ref)
    if person:
        return person["display_name"]
    return person_ref


def _do_auto_update(person_ref: str, score: float, mode: str,
                    face: dict, embedding: list, display_name: str | None = None) -> bool:
    if not AUTO_UPDATE_ENABLED:
        return False
    threshold = MATCH_THRESHOLDS.get(mode, MATCH_THRESHOLDS.get("cvlface", 0.40))
    min_auto  = AUTO_UPDATE_MIN_CONFIDENCE.get(mode, threshold)
    high_conf = AUTO_UPDATE_HIGH_CONFIDENCE.get(mode, max(min_auto, threshold))
    if score < min_auto or score >= high_conf:
        return False
    name_for_payload = display_name or person_ref
    added, _ = auto_update_add_template(
        name=name_for_payload,
        mode=mode,
        embedding=embedding,
        bbox=face["bbox"],
        landmarks=face["landmarks"],
        person_id=person_ref,
    )
    return added


async def _resolve_geo(
    image_bytes: bytes,
    lat: Optional[float],
    lon: Optional[float],
    accuracy_m: Optional[float],
    zone_id: Optional[str],
    skip_geocode: bool,
) -> dict:
    """
    Full geotag — geofence check + reverse geocode in one call.
    Called in parallel with _run_pipeline() via asyncio.gather so the
    Nominatim HTTP call (200-800 ms) overlaps with face detection and
    embedding instead of adding to total latency.
    Address info is present in the response — no more 'Unknown'.
    """
    geo = await geotag_event(
        lat=lat,
        lon=lon,
        source="browser",
        accuracy_m=accuracy_m,
        image_bytes=image_bytes,
        zone_id=zone_id,
        skip_geocode=skip_geocode,
    )
    geo_dict = geo.to_dict()
    if GEOFENCE_ENFORCE and not geo.geofence_passed:
        raise HTTPException(status_code=403, detail={"error":"GEOFENCE_DENIED","message":f"Your location is outside the allowed zone: {geo.geofence_reason}. Move within the designated area and try again.","field":"lat","code":"GEOFENCE_DENIED"})
    return geo_dict


# ─────────────────────────────────────────────
# v1 router prefix
# ─────────────────────────────────────────────

from fastapi import APIRouter
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
# Admin-only router — register and delete require admin key
v1_admin = APIRouter(prefix="/v1", dependencies=[Depends(require_admin_key)])


# ═══════════════════════════════════════════════
# GET /v1/status
# ═══════════════════════════════════════════════

@v1.get("/status", response_model=StatusResponse, tags=["System"])
@limiter.limit("60/minute")
def get_status(request: Request):
    """Health check and system status."""
    return {
        "cvlface_loaded":          is_cvlface_loaded(),
        "liveness_enabled":        LIVENESS_ENABLED,
        "active_liveness_enabled": LIVENESS_ACTIVE_ENABLED,
        "db_tier":                 current_tier_info(),
        "match_thresholds":        MATCH_THRESHOLDS,
    }


# ═══════════════════════════════════════════════
# POST /v1/faces/detections
# ═══════════════════════════════════════════════

@v1.post("/faces/detections", response_model=DetectResponse, tags=["Detection"])
@limiter.limit("120/minute")
async def detect(
    request: Request,
    image: UploadFile = File(..., description="JPEG or PNG image"),
):
    """
    Detect all faces in an image.
    Returns bounding boxes, landmarks, confidence scores, and quality results.
    """
    raw          = _decode_image(await image.read())
    preprocessed = preprocess_image(raw, fast=False)
    faces        = _silent_detect(preprocessed)

    for face in faces:
        face["quality"] = check_face_quality(raw, face)

    return {
        "faces": [
            {
                "face_id":   face["face_id"],
                "score":     round(face["score"], 4),
                "bbox":      face["bbox"],
                "landmarks": face["landmarks"],
                "quality":   face["quality"],
            }
            for face in faces
        ],
        "total": len(faces),
    }


# Backward-compatible alias for older clients still calling /v1/detect.
@v1.post("/detect", response_model=DetectResponse, tags=["Detection"], include_in_schema=False)
@limiter.limit("60/minute")
async def detect_legacy(
    request: Request,
    image: UploadFile = File(..., description="JPEG or PNG image"),
):
    return await detect(request=request, image=image)


# ═══════════════════════════════════════════════
# POST /v1/faces/liveness
# ═══════════════════════════════════════════════

@v1.post("/faces/liveness", response_model=LivenessResult, tags=["Liveness"])
@limiter.limit("20/minute")
async def liveness_check(
    request: Request,
    image:        UploadFile = File(...,  description="JPEG or PNG image"),
    passive_only: bool       = Form(True, description="Passive only — no camera needed"),
    camera_index: int        = Form(0,    description="Camera index for active challenge"),
):
    """
    Run liveness detection on the largest detected face.
    Returns pass/fail, score, and method used (passive | active).
    """
    raw = _decode_image(await image.read())
    _, faces = _run_pipeline(raw)

    if not faces:
        raise HTTPException(status_code=422, detail={"error":"NO_FACE_DETECTED","message":"No face was detected in the image. Ensure the face is clearly visible, well-lit, and at least 90×90 px in the frame.","field":"image","code":"NO_FACE_DETECTED"})

    face = faces[0]
    lv   = _liveness_check(raw, face, camera_index=camera_index, skip_active=passive_only)
    return {
        "passed":       lv.get("passed", False),
        "method":       lv.get("method", "passive"),
        "score":        float(lv.get("score", 0.0)),
        "reason":       lv.get("reason", ""),
        "model_scores": lv.get("model_scores"),
    }


# ═══════════════════════════════════════════════
# POST /v1/persons  — register (201 Created)
# ═══════════════════════════════════════════════

@v1_admin.post("/persons", response_model=RegisterResponse, status_code=201, tags=["Persons"])
@limiter.limit("10/minute")
async def register(
    request: Request,
    image:         UploadFile        = File(...,   description="JPEG or PNG image of the person"),
    name:          str               = Form(...,   description="Person's name (letters, digits, spaces, _- allowed)"),
    face_index:    int               = Form(0,     description="Which face to register when multiple detected (0 = largest)"),
    skip_liveness: bool              = Form(False, description="Bypass liveness check (not recommended for production)"),
    passive_only:  bool              = Form(True,  description="Passive liveness only — no camera challenge needed"),
    camera_index:  int               = Form(0,     description="Camera index for active liveness challenge"),
    lat:           Optional[float]   = Form(None,  description="Latitude from client geolocation"),
    lon:           Optional[float]   = Form(None,  description="Longitude from client geolocation"),
    accuracy_m:    Optional[float]   = Form(None,  description="GPS accuracy in metres"),
    zone_id:       Optional[str]     = Form(None,  description="Optional geofence zone id"),
    skip_geocode:  bool              = Form(False, description="Skip reverse geocoding lookup"),
):
    """
    Register a new person into the database. Returns **201 Created** on success.

    **Flow:** decode → preprocess → detect → quality check → liveness → embed → save.
    """
    existing_person = get_person_by_name(name)
    if existing_person:
        raise HTTPException(status_code=409, detail={
            "error":   "PERSON_ALREADY_EXISTS",
            "message": f"'{name}' is already registered. To replace them, call DELETE /v1/persons/{name} first, then register again.",
            "field":   "name",
            "code":    "PERSON_ALREADY_EXISTS",
        })
    try:
        person = create_person(name)
    except ValueError:
        raise HTTPException(status_code=422, detail={"error":"INVALID_NAME","message":"Name contains invalid characters. Use only letters, digits, spaces, hyphens, underscores, and apostrophes (max 64 characters).","field":"name","code":"INVALID_NAME"})
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"error":"DATABASE_ERROR","message":"Failed to initialise the person record in the SQL store. Check your SQL_DATABASE_URL configuration.","field":None,"code":"DATABASE_ERROR"})
    if not person:
        raise HTTPException(status_code=500, detail={"error":"DATABASE_ERROR","message":"Failed to create the person record. The SQL store may be unavailable — check server logs for details.","field":None,"code":"DATABASE_ERROR"})

    image_bytes = await image.read()
    raw = _decode_image(image_bytes)
    geo = await _resolve_geo(
        image_bytes=image_bytes,
        lat=lat, lon=lon,
        accuracy_m=accuracy_m,
        zone_id=zone_id,
        skip_geocode=skip_geocode,
    )
    preprocessed, faces = _run_pipeline(raw)

    if not faces:
        raise HTTPException(status_code=422, detail={"error":"NO_FACE_DETECTED","message":"No face was detected in the uploaded image. Ensure the face is clearly visible, well-lit, not blurred, and occupies a reasonable portion of the frame.","field":"image","code":"NO_FACE_DETECTED"})

    passed_faces, rejected = _quality_filter(raw, faces)
    if not passed_faces:
        detail = "; ".join(r["reason"] for r in rejected)
        raise HTTPException(status_code=422, detail={"error":"FACE_QUALITY_FAILED","message":f"All detected faces failed the quality check: {detail}. Ensure the face is at least 90×90 px, not blurry, and looking roughly forward.","field":"image","code":"FACE_QUALITY_FAILED"})

    if face_index >= len(passed_faces):
        face_index = 0
    face = passed_faces[face_index]

    liveness_result = None
    if LIVENESS_ENABLED and not skip_liveness:
        liveness_raw = _liveness_check(
            raw, face,
            camera_index=camera_index,
            skip_active=passive_only,
        )
        liveness_result = LivenessResult(
            passed=liveness_raw.get("passed", False),
            method=liveness_raw.get("method", "passive"),
            score=float(liveness_raw.get("score", 0.0)),
            reason=liveness_raw.get("reason", ""),
            model_scores=liveness_raw.get("model_scores"),
        )
        if not liveness_result.passed:
            # Liveness failed — return 403 with the standard error envelope.
            # Per-model scores are included so the developer can see which
            # model triggered the rejection without digging through server logs.
            return JSONResponse(
                status_code=403,
                content={
                    "error":   "LIVENESS_FAILED",
                    "message": (
                        f"Liveness check failed: {liveness_result.reason}. "
                        "Ensure the face is real, well-lit, and looking at the camera."
                    ),
                    "field":   "image",
                    "code":    "LIVENESS_FAILED",
                    "detail":  {
                        "face_id":  face["face_id"],
                        "liveness": liveness_result.dict(),
                        "geo":      geo,
                    },
                },
            )

    aligned        = align_face(raw, face["landmarks"], bbox=face["bbox"])
    embedding, mode = generate_embedding(aligned)

    ok = save_record(
        person["display_name"],
        face["bbox"],
        face["landmarks"],
        embedding,
        embedding_mode=mode,
        person_id=person["person_id"],
    )
    if not ok:
        delete_person_by_name(name)
        raise HTTPException(status_code=500, detail={"error":"EMBEDDING_SAVE_FAILED","message":"Failed to save the face embedding to the vector database. The person record has been rolled back. Check Qdrant storage availability.","field":None,"code":"EMBEDDING_SAVE_FAILED"})

    return {
        "success":        True,
        "name":           person["display_name"],
        "face_id":        face["face_id"],
        "liveness":       liveness_result,
        "geo":            geo,
        "embedding_mode": mode,
        "message":        f"'{name}' registered successfully.",
    }


# ═══════════════════════════════════════════════
# POST /v1/faces/matches
# ═══════════════════════════════════════════════

@v1.post("/faces/matches", response_model=MatchResponse, tags=["Matching"])
@limiter.limit("30/minute")
async def match(
    request: Request,
    image:         UploadFile      = File(...,   description="JPEG or PNG image"),
    skip_liveness: bool            = Form(False, description="Bypass liveness check"),
    passive_only:  bool            = Form(True,  description="Passive liveness only"),
    camera_index:  int             = Form(0,     description="Camera index for active liveness"),
    auto_update:   bool            = Form(True,  description="Auto-update templates on high-confidence match"),
    lat:           Optional[float] = Form(None,  description="Latitude from client geolocation"),
    lon:           Optional[float] = Form(None,  description="Longitude from client geolocation"),
    accuracy_m:    Optional[float] = Form(None,  description="GPS accuracy in metres"),
    zone_id:       Optional[str]   = Form(None,  description="Optional geofence zone id"),
    skip_geocode:  bool            = Form(False, description="Skip reverse geocoding lookup"),
):
    """
    Identify all faces in an image against the database.
    Returns one result per detected face.
    """
    import asyncio as _asyncio

    image_bytes = await image.read()
    raw = _decode_image(image_bytes)

    # Run geocode AND face pipeline in parallel.
    # Nominatim (~500 ms) overlaps with RetinaFace + preprocess (~500 ms)
    # so total wait = max(geocode, pipeline) instead of geocode + pipeline.
    # Address info is fully present in the response — no more "Unknown".
    async def _run_pipeline_async():
        return _run_pipeline(raw)

    geo, (preprocessed, faces) = await _asyncio.gather(
        _resolve_geo(
            image_bytes=image_bytes,
            lat=lat, lon=lon,
            accuracy_m=accuracy_m,
            zone_id=zone_id,
            skip_geocode=skip_geocode,
        ),
        _run_pipeline_async(),
    )

    if not faces:
        return {"results": [], "total_faces": 0, "matched_count": 0, "geo": geo}

    passed_faces, _ = _quality_filter(raw, faces)

    results = []
    matched_count = 0

    # Rejected faces
    for face in faces:
        if not face.get("quality", {}).get("passed", False):
            results.append(MatchResult(
                face_id=face["face_id"],
                matched=False,
                name="Rejected",
                score=0.0,
                liveness=None,
                geo=geo,
                embedding_mode="",
                auto_updated=False,
                bbox=face.get("bbox"),
                landmarks=face.get("landmarks"),
                quality=face.get("quality"),
            ))

    for face in passed_faces:
        liveness_result = None

        if LIVENESS_ENABLED and not skip_liveness:
            liveness_raw = _liveness_check(
                raw, face,
                camera_index=camera_index,
                skip_active=passive_only,
            )
            liveness_result = LivenessResult(
                passed=liveness_raw.get("passed", False),
                method=liveness_raw.get("method", "passive"),
                score=float(liveness_raw.get("score", 0.0)),
                reason=liveness_raw.get("reason", ""),
                model_scores=liveness_raw.get("model_scores"),
            )
            if not liveness_result.passed:
                reason_text = (liveness_result.reason or "").strip().lower()
                name = "Uncertain" if ("uncertain" in reason_text or "unavailable" in reason_text) else "Spoof"
                results.append(MatchResult(
                    face_id=face["face_id"],
                    matched=False,
                    name=name,
                    score=0.0,
                    liveness=liveness_result,
                    geo=geo,
                    embedding_mode="",
                    auto_updated=False,
                    bbox=face.get("bbox"),
                    landmarks=face.get("landmarks"),
                    quality=face.get("quality"),
                ))
                continue

        aligned          = align_face(raw, face["landmarks"], bbox=face["bbox"])
        embedding, mode  = generate_embedding(aligned)
        person_ref, score = match_face(embedding, query_mode=mode)
        matched           = is_match(person_ref)
        name              = _resolve_identity_name(person_ref)

        if matched:
            matched_count += 1

        updated = False
        if matched and auto_update:
            updated = _do_auto_update(person_ref, score, mode, face, embedding, display_name=name)

        results.append(MatchResult(
            face_id=face["face_id"],
            matched=matched,
            name=name,
            score=score,
            liveness=liveness_result,
            geo=geo,
            embedding_mode=mode,
            auto_updated=updated,
            bbox=face.get("bbox"),
            landmarks=face.get("landmarks"),
            quality=face.get("quality"),
        ))

    return {
        "results":       results,
        "total_faces":   len(faces),
        "matched_count": matched_count,
        "geo":           geo,
    }


# ═══════════════════════════════════════════════
# POST /v1/attendance/records
# ═══════════════════════════════════════════════

@v1.post("/attendance/records", response_model=AttendanceResult, tags=["Attendance"])
@limiter.limit("30/minute")
async def log_attendance(
    request: Request,
    image:         UploadFile      = File(...),
    skip_liveness: bool            = Form(False),
    passive_only:  bool            = Form(True),
    camera_index:  int             = Form(0),
    lat:           Optional[float] = Form(None),
    lon:           Optional[float] = Form(None),
    accuracy_m:    Optional[float] = Form(None),
    zone_id:       Optional[str]   = Form(None),
    skip_geocode:  bool            = Form(False),
):
    """
    Convenience endpoint: match all faces and return an attendance-ready payload.

    - `present` — deduplicated list of matched names
    - `unknown_count` — faces that didn't match
    - `spoofed_count` — faces that failed liveness
    - `detail` — per-face breakdown
    """
    import asyncio as _asyncio

    image_bytes = await image.read()
    raw = _decode_image(image_bytes)

    # Run geocode AND face pipeline in parallel.
    # Nominatim (~500 ms) overlaps with RetinaFace + preprocess (~500 ms)
    # so total wait = max(geocode, pipeline) instead of geocode + pipeline.
    # Address info is fully present in the response — no more "Unknown".
    async def _run_pipeline_async():
        return _run_pipeline(raw)

    geo, (_, faces) = await _asyncio.gather(
        _resolve_geo(
            image_bytes=image_bytes,
            lat=lat, lon=lon,
            accuracy_m=accuracy_m,
            zone_id=zone_id,
            skip_geocode=skip_geocode,
        ),
        _run_pipeline_async(),
    )
    passed_faces, _ = _quality_filter(raw, faces)

    present, unknown, spoofed, detail = [], 0, 0, []

    for face in passed_faces:
        liveness_result = None

        if LIVENESS_ENABLED and not skip_liveness:
            lv = _liveness_check(raw, face,
                                 camera_index=camera_index, skip_active=passive_only)
            liveness_result = lv
            if not lv.get("passed", False):
                spoofed += 1
                detail.append({
                    "face_id":        face["face_id"],
                    "status":         "spoofed",
                    "liveness_score": lv.get("score", 0.0),
                })
                continue

        aligned          = align_face(raw, face["landmarks"], bbox=face["bbox"])
        embedding, mode  = generate_embedding(aligned)
        person_ref, score = match_face(embedding, query_mode=mode)
        name              = _resolve_identity_name(person_ref)

        if is_match(person_ref):
            _do_auto_update(person_ref, score, mode, face, embedding, display_name=name)
            if _can_log_attendance(name):
                # First time this person is seen in the cooldown window — count them
                if name not in present:
                    present.append(name)
                detail.append({
                    "face_id":        face["face_id"],
                    "status":         "matched",
                    "name":           name,
                    "score":          score,
                    "mode":           mode,
                    "liveness_score": liveness_result.get("score") if liveness_result else None,
                })
            else:
                # Within the cooldown window — acknowledge but don't double-count
                detail.append({
                    "face_id": face["face_id"],
                    "status":  "already_logged",
                    "name":    name,
                    "score":   score,
                })
        else:
            unknown += 1
            detail.append({
                "face_id": face["face_id"],
                "status":  "unknown",
                "score":   score,
            })

    return {
        "timestamp":     int(time.time()),
        "present":       present,
        "present_count": len(present),
        "unknown_count": unknown,
        "spoofed_count": spoofed,
        "total_faces":   len(passed_faces),
        "geo":           geo,
        "detail":        detail,
    }


# ═══════════════════════════════════════════════
# GET /v1/persons  — list (paginated)
# ═══════════════════════════════════════════════

@v1.get("/persons", response_model=PersonsListResponse, tags=["Persons"])
@limiter.limit("30/minute")
def persons_list(
    request: Request,
    limit:            int           = Query(50,   ge=1, le=500, description="Max records to return"),
    after_created_at: Optional[str] = Query(None, description="Cursor: created_at from previous next_cursor"),
    after_person_id:  Optional[str] = Query(None, description="Cursor: person_id from previous next_cursor"),
):
    """
    List registered persons. Cursor-paginated — O(log n) on every page.
    Pass next_cursor values from the previous response to get the next page.
    """
    return list_persons(
        limit=limit,
        after_created_at=after_created_at,
        after_person_id=after_person_id,
    )


# ═══════════════════════════════════════════════
# GET /v1/persons/{name}
# ═══════════════════════════════════════════════

@v1.get("/persons/{name}", tags=["Persons"])
@limiter.limit("60/minute")
def person_get(request: Request, name: str):
    """
    Check whether a person is registered.
    Returns 200 with exists=true/false. Use 404 absence to gate re-registration.
    """
    person = get_person_by_name(name)
    if not person:
        raise HTTPException(status_code=404, detail={"error":"PERSON_NOT_FOUND","message":f"'{name}' is not registered. Check the spelling or call GET /v1/persons to list all registered names.","field":"name","code":"PERSON_NOT_FOUND"})
    return {"name": person["display_name"], "exists": True}


# ═══════════════════════════════════════════════
# DELETE /v1/persons/{name}  — 204 No Content
# ═══════════════════════════════════════════════

@v1_admin.delete("/persons/{name}", status_code=204, tags=["Persons"])
@limiter.limit("10/minute")
def person_delete(request: Request, name: str):
    """
    Delete all templates for a person. Returns **204 No Content** on success.
    Raises 404 if the person is not registered.
    """
    person = get_person_by_name(name)
    if not person:
        raise HTTPException(status_code=404, detail={"error":"PERSON_NOT_FOUND","message":f"'{name}' is not registered. Check the spelling or call GET /v1/persons to list all registered names.","field":"name","code":"PERSON_NOT_FOUND"})
    delete_record_by_person_id(person["person_id"])
    delete_person_by_name(name)
    return Response(status_code=204)


# ═══════════════════════════════════════════════
# GET /v1/db/tier
# ═══════════════════════════════════════════════

@v1.get("/db/tier", tags=["System"])
@limiter.limit("30/minute")
def db_tier(request: Request):
    """Return current HNSW tier info — strategy, cap, HNSW params, search ef."""
    return current_tier_info()


# ─────────────────────────────────────────────
# Mount the versioned router
# ─────────────────────────────────────────────

app.include_router(v1)
app.include_router(v1_admin)