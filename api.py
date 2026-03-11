"""
api.py — FastAPI server for the Face Recognition Attendance Pipeline.

Mirrors every capability from main.py as REST endpoints.
Active liveness (head-pose challenge) requires a connected camera on the server.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Dependencies (add to requirements.txt):
    fastapi
    uvicorn[standard]
    python-multipart        # for UploadFile / Form
"""

import io
import sys
import time
import contextlib
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from config import (
    MATCH_THRESHOLDS,
    AUTO_UPDATE_ENABLED,
    AUTO_UPDATE_MIN_CONFIDENCE,
    AUTO_UPDATE_HIGH_CONFIDENCE,
    LIVENESS_ENABLED,
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

logger = logging.getLogger("face_api")

# ═══════════════════════════════════════════════
# Startup — pre-load all models before accepting requests
# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load all three models at startup so the first HTTP request
    is never blocked by a multi-second model download/init.

    Startup order:
      1. RetinaFace  (face detector   — PyTorch, ~100 MB)
      2. CVLFace     (face embedder   — AdaFace IR-101, ~250 MB)
      3. SilentFace  (liveness ONNX   — V2 + V1SE, ~1 MB each)

    Server will NOT start (process exits) if RetinaFace or CVLFace fail to load,
    since the pipeline is non-functional without them.
    Liveness failure is a warning — the server starts but liveness will be skipped.
    """
    _startup_load_models()
    yield
    # Shutdown: nothing to tear down — atexit handles Qdrant client close


def _startup_load_models() -> None:
    failures = []

    # ── RetinaFace ────────────────────────────────────────────────────────
    print("[STARTUP] Loading RetinaFace detector...", flush=True)
    try:
        _get_retinaface_model()
        print("[STARTUP] ✓ RetinaFace ready", flush=True)
    except Exception as exc:
        msg = f"RetinaFace failed to load: {exc}"
        print(f"[STARTUP] ✗ {msg}", flush=True)
        failures.append(msg)

    # ── CVLFace / AdaFace embedder ────────────────────────────────────────
    print("[STARTUP] Loading CVLFace embedder (AdaFace IR-101)...", flush=True)
    try:
        model = _load_cvlface()
        if model is None:
            raise RuntimeError("_load_cvlface returned None — check dependencies")
        print("[STARTUP] ✓ CVLFace ready", flush=True)
    except Exception as exc:
        msg = f"CVLFace failed to load: {exc}"
        print(f"[STARTUP] ✗ {msg}", flush=True)
        failures.append(msg)

    # ── SilentFace liveness models ────────────────────────────────────────
    print("[STARTUP] Loading SilentFace liveness models (V2 + V1SE)...", flush=True)
    try:
        _load_all_sessions()
        from modules.liveness import _ort_sessions
        if _ort_sessions:
            loaded = list(_ort_sessions.keys())
            print(f"[STARTUP] ✓ Liveness ready — {loaded}", flush=True)
        else:
            print("[STARTUP] ⚠ Liveness models unavailable — passive check will be skipped", flush=True)
    except Exception as exc:
        print(f"[STARTUP] ⚠ Liveness load error: {exc}", flush=True)

    # ── Abort if critical models failed ──────────────────────────────────
    if failures:
        print("\n[STARTUP] FATAL — critical models failed to load:", flush=True)
        for f in failures:
            print(f"  • {f}", flush=True)
        print("[STARTUP] Fix the errors above and restart the server.\n", flush=True)
        sys.exit(1)

    print("[STARTUP] All models loaded — server ready.\n", flush=True)


app = FastAPI(
    title="Face Recognition Attendance API",
    description=(
        "REST interface for the face recognition pipeline.\n\n"
        "All image endpoints accept **multipart/form-data** with an `image` field "
        "(JPEG / PNG).\n\n"
        "Active liveness requires a physical camera attached to the server "
        "(pass `camera_index`). "
        "Passive-only liveness works without a camera."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════
# Pydantic response models
# ═══════════════════════════════════════════════

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

class RegisterResponse(BaseModel):
    success: bool
    name: str
    face_id: str
    liveness: Optional[LivenessResult] = None
    embedding_mode: str
    message: str

class MatchResult(BaseModel):
    face_id: str
    matched: bool
    name: str
    score: float
    liveness: Optional[LivenessResult] = None
    embedding_mode: str
    auto_updated: bool = False

class MatchResponse(BaseModel):
    results: list[MatchResult]
    total_faces: int
    matched_count: int

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

class DBListResponse(BaseModel):
    records: list[PersonRecord]
    total: int

class StatusResponse(BaseModel):
    cvlface_loaded: bool
    liveness_enabled: bool
    active_liveness_enabled: bool
    db_tier: dict
    match_thresholds: dict


# ═══════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════

def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes → BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Send a valid JPEG or PNG.")
    return img


def _silent_detect(image: np.ndarray) -> list:
    with contextlib.redirect_stdout(io.StringIO()):
        return detect_faces(image)


def _run_pipeline(image_bgr: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Preprocess → detect → quality filter.
    Returns (preprocessed_image, quality_passed_faces).
    """
    preprocessed = preprocess_image(image_bgr, fast=False)
    faces = _silent_detect(preprocessed)
    return preprocessed, faces


def _quality_filter(raw: np.ndarray, faces: list) -> tuple[list, list]:
    """Returns (passed_faces, rejected_reasons)."""
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
    """Run liveness. skip_active=True forces passive-only (no camera needed)."""
    if skip_active:
        # Temporarily patch active flag to False for this call
        import config as _cfg
        orig = _cfg.LIVENESS_ACTIVE_ENABLED
        _cfg.LIVENESS_ACTIVE_ENABLED = False
        try:
            result = check_liveness(raw, face, camera_index=camera_index)
        finally:
            _cfg.LIVENESS_ACTIVE_ENABLED = orig
        return result
    return check_liveness(raw, face, camera_index=camera_index)


def _do_auto_update(name: str, score: float, mode: str,
                    face: dict, embedding: list) -> bool:
    """Mirror of main.py _auto_update. Returns True if template was added."""
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


# ═══════════════════════════════════════════════
# ── /status ──────────────────────────────────
# ═══════════════════════════════════════════════

@app.get("/status", response_model=StatusResponse, tags=["System"])
def get_status():
    """Health check and system status."""
    import config as _cfg
    return {
        "cvlface_loaded":         is_cvlface_loaded(),
        "liveness_enabled":       LIVENESS_ENABLED,
        "active_liveness_enabled": _cfg.LIVENESS_ACTIVE_ENABLED,
        "db_tier":                current_tier_info(),
        "match_thresholds":       MATCH_THRESHOLDS,
    }


# ═══════════════════════════════════════════════
# ── /detect ──────────────────────────────────
# ═══════════════════════════════════════════════

@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect(
    image: UploadFile = File(..., description="JPEG or PNG image"),
):
    """
    Detect all faces in an image.
    Returns face bounding boxes, landmarks, confidence scores, and quality results
    for every detected face (including those that fail the quality gate).
    """
    raw          = _decode_image(await image.read())
    preprocessed = preprocess_image(raw, fast=False)
    faces        = _silent_detect(preprocessed)

    # Run quality check and attach result to each face dict in-place
    for face in faces:
        face["quality"] = check_face_quality(preprocessed, face)

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


# ═══════════════════════════════════════════════
# ── /register ────────────────────────────────
# ═══════════════════════════════════════════════

@app.post("/register", response_model=RegisterResponse, tags=["Registration"])
async def register(
    image:        UploadFile = File(...,   description="JPEG or PNG image of the person"),
    name:         str        = Form(...,   description="Person's name (letters, digits, spaces, _- allowed)"),
    face_index:   int        = Form(0,     description="Which face to register when multiple detected (0 = largest)"),
    skip_liveness: bool      = Form(False, description="Bypass liveness check (not recommended for production)"),
    passive_only: bool       = Form(True,  description="Passive liveness only — no camera challenge needed"),
    camera_index: int        = Form(0,     description="Camera index for active liveness challenge"),
):
    """
    Register a new person into the database.

    **Flow:** decode → preprocess → detect → quality check → liveness → embed → save.

    - `passive_only=true` (default): uses the SilentFace passive model only — no camera needed.
    - `passive_only=false`: may trigger active head-pose challenge using server camera.
    - `skip_liveness=true`: skips liveness entirely (use only in controlled environments).
    """
    raw = _decode_image(await image.read())
    preprocessed, faces = _run_pipeline(raw)

    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in image.")

    passed_faces, rejected = _quality_filter(preprocessed, faces)
    if not passed_faces:
        detail = "; ".join(r["reason"] for r in rejected)
        raise HTTPException(status_code=422, detail=f"All faces failed quality check: {detail}")

    # Select face by index
    if face_index >= len(passed_faces):
        face_index = 0
    face = passed_faces[face_index]

    liveness_result = None
    if LIVENESS_ENABLED and not skip_liveness:
        liveness_raw = _liveness_check(
            preprocessed, face,
            camera_index=camera_index,
            skip_active=passive_only,
        )
        liveness_result = LivenessResult(
            passed=liveness_raw.get("passed", False),
            method=liveness_raw.get("method", "passive"),
            score=float(liveness_raw.get("score", 0.0)),
            reason=liveness_raw.get("reason", ""),
        )
        if not liveness_result.passed:
            return JSONResponse(
                status_code=403,
                content={
                    "success":  False,
                    "name":     name,
                    "face_id":  face["face_id"],
                    "liveness": liveness_result.dict(),
                    "embedding_mode": "",
                    "message":  f"Liveness failed: {liveness_result.reason}",
                },
            )

    aligned        = align_face(preprocessed, face["landmarks"], bbox=face["bbox"])
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
        "embedding_mode": mode,
        "message":        f"'{name}' registered successfully.",
    }


# ═══════════════════════════════════════════════
# ── /match ───────────────────────────────────
# ═══════════════════════════════════════════════

@app.post("/match", response_model=MatchResponse, tags=["Matching"])
async def match(
    image:        UploadFile = File(...,   description="JPEG or PNG image"),
    skip_liveness: bool      = Form(False, description="Bypass liveness check"),
    passive_only: bool       = Form(True,  description="Passive liveness only"),
    camera_index: int        = Form(0,     description="Camera index for active liveness"),
    auto_update:  bool       = Form(True,  description="Auto-update templates on high-confidence match"),
):
    """
    Identify all faces in an image against the database.

    Returns a match result per face including name, score, liveness status, and
    whether the face template was auto-updated.

    **Attendance use-case:** call this endpoint with each captured frame/image.
    """
    raw = _decode_image(await image.read())
    preprocessed, faces = _run_pipeline(raw)

    if not faces:
        return {"results": [], "total_faces": 0, "matched_count": 0}

    passed_faces, _ = _quality_filter(preprocessed, faces)

    results = []
    matched_count = 0

    for face in passed_faces:
        liveness_result = None

        if LIVENESS_ENABLED and not skip_liveness:
            liveness_raw = _liveness_check(
                preprocessed, face,
                camera_index=camera_index,
                skip_active=passive_only,
            )
            liveness_result = LivenessResult(
                passed=liveness_raw.get("passed", False),
                method=liveness_raw.get("method", "passive"),
                score=float(liveness_raw.get("score", 0.0)),
                reason=liveness_raw.get("reason", ""),
            )
            if not liveness_result.passed:
                results.append(MatchResult(
                    face_id=face["face_id"],
                    matched=False,
                    name="Spoof",
                    score=0.0,
                    liveness=liveness_result,
                    embedding_mode="",
                    auto_updated=False,
                ))
                continue

        aligned         = align_face(preprocessed, face["landmarks"], bbox=face["bbox"])
        embedding, mode = generate_embedding(aligned)
        name, score     = match_face(embedding, query_mode=mode)
        matched         = is_match(name)

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
            embedding_mode=mode,
            auto_updated=updated,
        ))

    return {
        "results":       results,
        "total_faces":   len(passed_faces),
        "matched_count": matched_count,
    }


# ═══════════════════════════════════════════════
# ── /attendance ──────────────────────────────
# ═══════════════════════════════════════════════

@app.post("/attendance", tags=["Attendance"])
async def log_attendance(
    image:        UploadFile = File(...),
    skip_liveness: bool      = Form(False),
    passive_only: bool       = Form(True),
    camera_index: int        = Form(0),
):
    """
    Convenience endpoint: match all faces and return an attendance-ready payload.

    Response includes:
    - `present`: list of matched names (deduplicated)
    - `unknown`: count of faces that didn't match
    - `spoofed`: count of faces that failed liveness
    - `timestamp`: UTC epoch seconds
    - Full per-face detail
    """
    raw = _decode_image(await image.read())
    preprocessed, faces = _run_pipeline(raw)
    passed_faces, _ = _quality_filter(preprocessed, faces)

    present, unknown, spoofed, detail = [], 0, 0, []

    for face in passed_faces:
        liveness_result = None

        if LIVENESS_ENABLED and not skip_liveness:
            lv = _liveness_check(preprocessed, face,
                                 camera_index=camera_index, skip_active=passive_only)
            liveness_result = lv
            if not lv.get("passed", False):
                spoofed += 1
                detail.append({
                    "face_id": face["face_id"],
                    "status":  "spoofed",
                    "liveness_score": lv.get("score", 0.0),
                })
                continue

        aligned         = align_face(preprocessed, face["landmarks"], bbox=face["bbox"])
        embedding, mode = generate_embedding(aligned)
        name, score     = match_face(embedding, query_mode=mode)

        if is_match(name):
            if name not in present:
                present.append(name)
            _do_auto_update(name, score, mode, face, embedding)
            detail.append({
                "face_id":  face["face_id"],
                "status":   "matched",
                "name":     name,
                "score":    score,
                "mode":     mode,
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
        "detail":        detail,
    }


# ═══════════════════════════════════════════════
# ── /liveness ────────────────────────────────
# ═══════════════════════════════════════════════

@app.post("/liveness", response_model=LivenessResult, tags=["Liveness"])
async def liveness_check(
    image:        UploadFile = File(...,   description="JPEG or PNG image"),
    passive_only: bool       = Form(True,  description="Passive only — no camera needed"),
    camera_index: int        = Form(0,     description="Camera index for active challenge"),
):
    """
    Run liveness detection on the largest detected face.
    Returns pass/fail, score, and method used (passive | active).
    """
    raw = _decode_image(await image.read())
    preprocessed, faces = _run_pipeline(raw)

    if not faces:
        raise HTTPException(status_code=422, detail="No face detected.")

    face = faces[0]  # largest / most confident
    lv   = _liveness_check(preprocessed, face,
                            camera_index=camera_index, skip_active=passive_only)
    return {
        "passed": lv.get("passed", False),
        "method": lv.get("method", "passive"),
        "score":  float(lv.get("score", 0.0)),
        "reason": lv.get("reason", ""),
    }


# ═══════════════════════════════════════════════
# ── /db/list ─────────────────────────────────
# ═══════════════════════════════════════════════

@app.get("/db/list", response_model=DBListResponse, tags=["Database"])
def db_list(
    include_embeddings: bool = Query(False, description="Include raw 512-d embeddings in response"),
):
    """
    List all registered persons and their templates.
    Set `include_embeddings=true` to return the raw vectors (large payload).
    """
    records = load_db()
    cleaned = []
    for r in records:
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

    return {"records": cleaned, "total": len(cleaned)}


# ═══════════════════════════════════════════════
# ── /db/exists ───────────────────────────────
# ═══════════════════════════════════════════════

@app.get("/db/exists", tags=["Database"])
def db_exists(name: str = Query(..., description="Person name to check")):
    """Check whether a name is already registered in the database."""
    exists = record_exists(name)
    return {"name": name, "exists": exists}


# ═══════════════════════════════════════════════
# ── /db/delete ───────────────────────────────
# ═══════════════════════════════════════════════

@app.delete("/db/delete", response_model=DeleteResponse, tags=["Database"])
def db_delete(name: str = Query(..., description="Name to delete")):
    """
    Delete all templates for a person from the database.
    This is irreversible — re-register to add them back.
    """
    ok = delete_record(name)
    return {
        "success": ok,
        "name":    name,
        "message": f"Deleted '{name}'." if ok else f"'{name}' not found in database.",
    }


# ═══════════════════════════════════════════════
# ── /db/tier ─────────────────────────────────
# ═══════════════════════════════════════════════

@app.get("/db/tier", tags=["Database"])
def db_tier():
    """Return current HNSW tier info — strategy, cap, HNSW params, search ef."""
    return current_tier_info()