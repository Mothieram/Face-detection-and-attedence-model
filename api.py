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
import contextlib
import logging
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
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Optional

from slowapi import Limiter, _rate_limit_exceeded_handler
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
    delete_record,
    record_exists,
    load_db,
)
from modules.matcher      import match_face, is_match, match_result_label
from modules.geotagging   import geotag_event

logger = logging.getLogger("face_api")

# ─────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

_API_KEYS_SET = set(API_KEYS)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    if not API_AUTH_ENABLED:
        return
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key. Send header: X-API-Key")
    if api_key not in _API_KEYS_SET:
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_load_models()
    yield


def _startup_load_models() -> None:
    failures = []

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

    if failures:
        print("\n[STARTUP] [WARN] Some components failed; server running in degraded mode:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)

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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/styles", StaticFiles(directory=os.path.join(_BASE_DIR, "styles")), name="styles")


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(os.path.join(_BASE_DIR, "index.html"))


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
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Consistent error handler
# All errors return: {"error": CODE, "message": "...", "detail": ...}
# ─────────────────────────────────────────────

ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "UNPROCESSABLE",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
}

@app.exception_handler(StarletteHTTPException)
async def unified_http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = ERROR_CODES.get(exc.status_code, "ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":   code,
            "message": str(exc.detail),
            "detail":  None,
        },
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
        raise HTTPException(
            status_code=400,
            detail="Cannot decode image. Send a valid JPEG or PNG.",
        )
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


def _do_auto_update(name: str, score: float, mode: str,
                    face: dict, embedding: list) -> bool:
    if not AUTO_UPDATE_ENABLED:
        return False
    threshold = MATCH_THRESHOLDS.get(mode, MATCH_THRESHOLDS.get("cvlface", 0.40))
    min_auto  = AUTO_UPDATE_MIN_CONFIDENCE.get(mode, threshold)
    high_conf = AUTO_UPDATE_HIGH_CONFIDENCE.get(mode, max(min_auto, threshold))
    if score < min_auto or score >= high_conf:
        return False
    added, _ = auto_update_add_template(
        name=name,
        mode=mode,
        embedding=embedding,
        bbox=face["bbox"],
        landmarks=face["landmarks"],
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
        raise HTTPException(
            status_code=403,
            detail=f"Geofence failed: {geo.geofence_reason}",
        )
    return geo_dict


# ─────────────────────────────────────────────
# v1 router prefix
# ─────────────────────────────────────────────

from fastapi import APIRouter
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


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
@limiter.limit("60/minute")
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
        raise HTTPException(status_code=422, detail="No face detected.")

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

@v1.post("/persons", response_model=RegisterResponse, status_code=201, tags=["Persons"])
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
    if record_exists(name):
        raise HTTPException(
            status_code=409,
            detail=f"'{name}' is already registered. Use DELETE /v1/persons/{name} first to re-register.",
        )

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
        raise HTTPException(status_code=422, detail="No face detected in image.")

    passed_faces, rejected = _quality_filter(raw, faces)
    if not passed_faces:
        detail = "; ".join(r["reason"] for r in rejected)
        raise HTTPException(status_code=422, detail=f"All faces failed quality check: {detail}")

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
            # Return 403 but with consistent error envelope
            return JSONResponse(
                status_code=403,
                content={
                    "error":   "LIVENESS_FAILED",
                    "message": f"Liveness failed: {liveness_result.reason}",
                    "detail":  {
                        "success":        False,
                        "name":           name,
                        "face_id":        face["face_id"],
                        "liveness":       liveness_result.dict(),
                        "geo":            geo,
                        "embedding_mode": "",
                    },
                },
            )

    aligned        = align_face(raw, face["landmarks"], bbox=face["bbox"])
    embedding, mode = generate_embedding(aligned)

    ok = save_record(
        name,
        face["bbox"],
        face["landmarks"],
        embedding,
        embedding_mode=mode,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Database write failed. Check name format.")

    return {
        "success":        True,
        "name":           name,
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
        name, score      = match_face(embedding, query_mode=mode)
        matched          = is_match(name)

        if matched:
            matched_count += 1

        updated = False
        if matched and auto_update:
            updated = _do_auto_update(name, score, mode, face, embedding)

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
    image_bytes = await image.read()
    raw = _decode_image(image_bytes)
    geo = await _resolve_geo(
        image_bytes=image_bytes,
        lat=lat, lon=lon,
        accuracy_m=accuracy_m,
        zone_id=zone_id,
        skip_geocode=skip_geocode,
    )
    _, faces = _run_pipeline(raw)
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
        name, score      = match_face(embedding, query_mode=mode)

        if is_match(name):
            if name not in present:
                present.append(name)
            _do_auto_update(name, score, mode, face, embedding)
            detail.append({
                "face_id":        face["face_id"],
                "status":         "matched",
                "name":           name,
                "score":          score,
                "mode":           mode,
                "liveness_score": liveness_result.get("score") if liveness_result else None,
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
    limit:              int  = Query(50,    ge=1, le=500, description="Max records to return"),
    offset:             int  = Query(0,     ge=0,         description="Number of records to skip"),
    include_embeddings: bool = Query(False,               description="Include raw 512-d embeddings (large payload)"),
):
    """
    List all registered persons. Paginated — use `limit` and `offset`.
    """
    all_records = list(load_db())
    page        = all_records[offset : offset + limit]

    cleaned = []
    for r in page:
        entry = {
            "name":           r.get("name", ""),
            "display_name":   r.get("display_name", r.get("name", "")),
            "embedding_mode": r.get("embedding_mode", ""),
            "template_type":  r.get("template_type", ""),
            "bbox":           r.get("bbox", []),
            "landmarks":      r.get("landmarks", {}),
        }
        if include_embeddings:
            entry["embedding"] = r.get("embedding", [])
        cleaned.append(entry)

    return {
        "records":  cleaned,
        "total":    len(all_records),
        "limit":    limit,
        "offset":   offset,
        "has_more": (offset + limit) < len(all_records),
    }


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
    exists = record_exists(name)
    if not exists:
        raise HTTPException(status_code=404, detail=f"'{name}' not found in database.")
    return {"name": name, "exists": True}


# ═══════════════════════════════════════════════
# DELETE /v1/persons/{name}  — 204 No Content
# ═══════════════════════════════════════════════

@v1.delete("/persons/{name}", status_code=204, tags=["Persons"])
@limiter.limit("10/minute")
def person_delete(request: Request, name: str):
    """
    Delete all templates for a person. Returns **204 No Content** on success.
    Raises 404 if the person is not registered.
    """
    if not record_exists(name):
        raise HTTPException(status_code=404, detail=f"'{name}' not found in database.")
    ok = delete_record(name)
    if not ok:
        raise HTTPException(status_code=500, detail="Delete failed.")
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
