"""
modules/database.py
Qdrant Vector Database with dynamic scaling, generator-based loading, 
and deterministic ID consolidation for production hardening.
"""

import json
import os
import re
import threading
import time
import uuid
import atexit
from typing import Optional, Generator

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import (
    AUTO_UPDATE_MAX_SIM_TO_EXISTING,
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    SCALE_TIERS,
)

def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two float lists."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    d  = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0

# Singleton client + runtime state
_client: Optional[QdrantClient] = None
_current_tier: int = -1

# Runtime metadata persisted on disk
_META_FILE = "_runtime_meta.json"
_meta_lock = threading.RLock()

# Per-person advisory locks
_person_locks: dict[str, threading.RLock] = {}
_person_locks_mutex = threading.Lock()
_LOCK_CACHE_MAX = 10_000

_NAME_RE = re.compile(r"^[a-zA-Z0-9 '_-]+$")
_MAX_NAME_LEN = 64

# ── Scaling & Tier Logic ──────────────────────────────────────────────────

def get_tier(people_count: int) -> dict:
    for i, (lo, hi, strategy, cap, hnsw_m, hnsw_ef, search_ef) in enumerate(SCALE_TIERS):
        if hi is None or people_count <= hi:
            return {
                "tier_index": i, "strategy": strategy, "cap": cap,
                "hnsw_m": hnsw_m, "hnsw_ef": hnsw_ef, "search_ef": search_ef,
                "people_min": lo, "people_max": hi,
            }
    return get_tier(0) # Fallback

def current_tier_info() -> dict:
    client = _get_client()
    people_count = _get_people_count(client)
    tier = get_tier(people_count)
    _maybe_update_hnsw(client, tier)
    return tier

def _get_client() -> QdrantClient:
    global _client, _current_tier
    if _client is not None: return _client
    _client = QdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in _client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        t0 = get_tier(0)
        _client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    meta = _ensure_runtime_meta(_client)
    _current_tier = int(meta.get("current_tier", -1))
    return _client

# ── Data Operations ──────────────────────────────────────────────────────

def save_record(name: str, bbox: list, landmarks: dict, embedding: list, 
                embedding_source: str = "unknown", embedding_mode: str | None = None, 
                template_type: str = "template") -> bool:
    mode = embedding_mode if embedding_mode is not None else embedding_source
    try:
        display_name, name_key = _validate_name(name)
    except ValueError as e:
        print(f"[DB] {e}")
        return False

    with _get_person_lock(name_key):
        return _save_record_locked(display_name, name_key, bbox, landmarks, 
                                   embedding, mode, template_type, True)

def load_db() -> Generator[dict, None, None]:
    """
    FIX: Now a Generator to prevent OOM.
    Yields records one-by-one from the Qdrant scroll.
    """
    client, offset = _get_client(), None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION, with_vectors=True, 
            with_payload=True, limit=512, offset=offset
        )
        for point in points:
            rec = dict(point.payload)
            rec["embedding"] = list(point.vector)
            yield rec # Stream records instead of loading into a list
        if offset is None: break

# 

def _consolidate(name_key: str, mode: str, templates: list) -> None:
    """
    FIX: Uses Deterministic IDs (UUIDv5) to prevent duplicate 'mean'/'best' records.
    """
    client = _get_client()
    vecs = [np.array(t["embedding"], dtype=np.float32) for t in templates]
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec); mean_vec = mean_vec / norm if norm > 0 else mean_vec
    
    mean_list = mean_vec.tolist()
    best = max(templates, key=lambda t: _cosine_sim(t["embedding"], mean_list))
    all_points = _scroll_by_name_key(name_key)
    old_ids = [p.id for p in all_points if p.payload.get("embedding_mode") == mode]

    meta = templates[0]
    base_payload = {
        "name": name_key, "display_name": meta.get("display_name", name_key),
        "bbox": meta.get("bbox", []), "landmarks": meta.get("landmarks", {}),
        "embedding_mode": mode,
    }

    # Deterministic IDs: Overwrites existing mean/best if they exist
    def get_det_id(t_type):
        seed = f"{name_key}_{mode}_{t_type}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

    mid, bid = get_det_id("mean"), get_det_id("best")
    _upsert_with_retry(client, [
        PointStruct(id=mid, vector=mean_list, payload={**base_payload, "template_type": "mean"}),
        PointStruct(id=bid, vector=best["embedding"], payload={**base_payload, "template_type": "best"}),
    ])

    to_delete = [oid for oid in old_ids if oid not in (mid, bid)]
    if to_delete: _delete_with_retry(client, to_delete)
    print(f"[DB] Consolidated '{name_key}' -> mean + best")

# ── Internal Helpers ──────────────────────────────────────────────────────
def _update_mean(name_key: str, mode: str, new_embedding: list, existing: list) -> None:
    """
    Calculates a new running average embedding and updates the 'mean' point in DB.
    """
    client = _get_client()
    # Convert all to numpy for math
    all_vecs = [np.array(t["embedding"], dtype=np.float32) for t in existing]
    all_vecs.append(np.array(new_embedding, dtype=np.float32))

    # Compute new mean and re-normalize to unit length (L2)
    mean_vec = np.mean(all_vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    
    # Generate the deterministic ID for the 'mean' record
    seed = f"{name_key}_{mode}_mean"
    mean_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

    # Recover display_name and metadata from existing or defaults
    display_name = existing[0].get("display_name", name_key) if existing else name_key
    bbox = existing[0].get("bbox", []) if existing else []
    landmarks = existing[0].get("landmarks", {}) if existing else {}

    _upsert_with_retry(client, [
        PointStruct(
            id=mean_id,
            vector=mean_vec.tolist(),
            payload={
                "name": name_key,
                "display_name": display_name,
                "bbox": bbox,
                "landmarks": landmarks,
                "embedding_mode": mode,
                "template_type": "mean",
            },
        )
    ])
    print(f"[DB] Mean updated for '{name_key}' ({len(all_vecs)} samples)")

def _save_record_locked(display_name, name_key, bbox, landmarks, embedding, 
                        mode, template_type, warn_if_exists) -> bool:
    client = _get_client()
    existed = _record_exists_key(name_key)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=[float(v) for v in embedding],
        payload={
            "name": name_key, "display_name": display_name,
            "bbox": [int(v) for v in bbox], "landmarks": landmarks,
            "embedding_mode": mode, "template_type": template_type
        },
    )
    try:
        _upsert_with_retry(client, [point])
        if not existed: _adjust_people_count(+1)
        return True
    except Exception as e:
        print(f"[DB] Save error: {e}"); return False

def auto_update_add_template(name: str, mode: str, embedding: list, 
                             bbox: list, landmarks: dict) -> tuple[bool, str]:
    name_key = name.strip().lower()
    with _get_person_lock(name_key):
        tier = current_tier_info()
        templates = fetch_templates_for(name_key, mode)
        
        if tier["strategy"] == "mean":
            _update_mean(name_key, mode, embedding, templates)
            return True, "mean updated"
        
        if len(templates) >= tier["cap"]:
            if tier["strategy"] == "hybrid":
                _consolidate(name_key, mode, templates)
                return True, "consolidated"
            return False, "at cap"
            
        return _save_record_locked(name, name_key, bbox, landmarks, embedding, mode, "template", False), "added"

def record_exists(name: str) -> bool:
    """Public API: check whether a person has at least one stored template."""
    try:
        _, name_key = _validate_name(name)
    except ValueError:
        return False
    return _record_exists_key(name_key)


def delete_record(name: str) -> bool:
    """
    Public API: delete all templates for a person.
    Returns True when at least one point was deleted.
    """
    try:
        _, name_key = _validate_name(name)
    except ValueError:
        return False

    with _get_person_lock(name_key):
        points = _scroll_by_name_key(name_key, limit=10_000)
        if not points:
            return False

        ids = [p.id for p in points]
        _delete_with_retry(_get_client(), ids)
        _adjust_people_count(-1)
        return True

def _record_exists_key(name_key: str) -> bool:
    return len(_scroll_by_name_key(name_key, limit=1)) > 0

def _scroll_by_name_key(name_key: str, limit: int = 100) -> list:
    client = _get_client()
    res, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="name", match=MatchValue(value=name_key))]),
        limit=limit
    )
    return list(res)

def fetch_templates_for(name_key: str, mode: str) -> list[dict]:
    client = _get_client()
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="name", match=MatchValue(value=name_key)),
            FieldCondition(key="embedding_mode", match=MatchValue(value=mode)),
        ]),
        with_vectors=True, limit=20
    )
    return [{"embedding": p.vector, **p.payload} for p in points]

def _get_person_lock(name_key: str) -> threading.RLock:
    with _person_locks_mutex:
        if name_key not in _person_locks:
            if len(_person_locks) >= _LOCK_CACHE_MAX: _person_locks.clear()
            _person_locks[name_key] = threading.RLock()
        return _person_locks[name_key]

def _validate_name(name: str) -> tuple[str, str]:
    d = name.strip()[:_MAX_NAME_LEN]
    if not d or not _NAME_RE.fullmatch(d): raise ValueError("Invalid name")
    return d, d.lower()

# ── Persistence & Meta ────────────────────────────────────────────────────

def _ensure_runtime_meta(client: QdrantClient) -> dict:
    with _meta_lock:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        path = os.path.join(QDRANT_PATH, _META_FILE)
        if not os.path.exists(path):
            meta = {"people_count": 0, "current_tier": 0}
            with open(path, "w") as f: json.dump(meta, f)
        with open(path, "r") as f: return json.load(f)

def _get_people_count(client: QdrantClient) -> int:
    return _ensure_runtime_meta(client).get("people_count", 0)

def _adjust_people_count(delta: int) -> None:
    with _meta_lock:
        path = os.path.join(QDRANT_PATH, _META_FILE)
        meta = _ensure_runtime_meta(_get_client())
        meta["people_count"] = max(0, meta["people_count"] + delta)
        with open(path, "w") as f: json.dump(meta, f)

def _upsert_with_retry(client, points, attempts=3):
    for i in range(attempts):
        try: return client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        except Exception: time.sleep(0.5 * (2**i))
    raise RuntimeError("DB Upsert failed")

def _delete_with_retry(client, ids, attempts=3):
    for i in range(attempts):
        try: return client.delete(collection_name=QDRANT_COLLECTION, points_selector=ids)
        except Exception: time.sleep(0.5 * (2**i))

def _maybe_update_hnsw(client, tier):
    global _current_tier
    if tier["tier_index"] != _current_tier:
        try:
            client.update_collection(
                collection_name=QDRANT_COLLECTION,
                hnsw_config=HnswConfigDiff(m=tier["hnsw_m"], ef_construct=tier["hnsw_ef"])
            )
            _current_tier = tier["tier_index"]
        except Exception: pass

def close_client():
    global _client
    if _client: _client.close(); _client = None

atexit.register(close_client)
