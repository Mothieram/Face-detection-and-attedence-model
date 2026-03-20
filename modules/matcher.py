"""
modules/matcher.py
───────────────────
STEP 6 — Face Matching via Qdrant ANN Search

Uses HNSW approximate nearest-neighbour search.
Supports qdrant-client >= 1.7 (query_points) with
automatic fallback to legacy search() for older installs.

MATCH_THRESHOLDS must define both 'cvlface' and 'fallback' keys.
Missing keys raise a KeyError at import time (fail fast).
"""

from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import MATCH_THRESHOLDS, QDRANT_COLLECTION
from modules.database import _get_client, current_tier_info

# Fail fast if config is missing required threshold keys
for _m in ("cvlface", "fallback"):
    if _m not in MATCH_THRESHOLDS:
        raise KeyError(
            f"[MATCH] MATCH_THRESHOLDS missing key '{_m}'. "
            f"Add it to config.py  e.g. MATCH_THRESHOLDS = {{'cvlface': 0.40, 'fallback': 0.92}}"
        )


def match_face(query_embedding: list,
               query_mode: str = "cvlface") -> tuple[str, float]:
    """
    Find the closest face in Qdrant using HNSW ANN search.

    Only compares against records with the same embedding_mode.
    Uses MATCH_THRESHOLDS[query_mode] as the accept threshold.

    Args:
        query_embedding : 512-d float list (L2-normalized)
        query_mode      : 'cvlface' | 'fallback'

    Returns:
        (person_ref, score) — person_ref is person_id when available.
        Falls back to display_name/name for older records.
    """
    client    = _get_client()
    threshold = MATCH_THRESHOLDS[query_mode]

    if client.count(collection_name=QDRANT_COLLECTION).count == 0:
        return "No DB records", 0.0

    vector      = [float(v) for v in query_embedding]
    mode_filter = Filter(
        must=[
            FieldCondition(
                key="embedding_mode",
                match=MatchValue(value=query_mode),
            )
        ]
    )

    # Use tier's search_ef so recall scales with DB size automatically.
    tier      = current_tier_info()
    search_ef = tier["search_ef"]

    # query_points() is the current API (qdrant-client >= 1.7).
    # Fall back to legacy search() for older installs.
    try:
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            query_filter=mode_filter,
            limit=10,
            with_payload=True,
            search_params={"hnsw_ef": search_ef},
        )
        results = response.points
    except AttributeError:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            query_filter=mode_filter,
            limit=10,
            with_payload=True,
            search_params={"hnsw_ef": search_ef},
        )

    if not results:
        print(f"[MATCH] No DB records with mode='{query_mode}'. "
              f"Re-register faces with current embedding mode.")
        return "No DB records", 0.0

    best  = results[0]
    score = float(best.score)
    person_ref = str(
        best.payload.get("person_id")
        or best.payload.get("display_name")
        or best.payload.get("name", "Unknown")
    )

    if score < threshold:
        return "No Match", score

    return person_ref, score



def match_result_label(name: str, score: float) -> str:
    return f"{name} ({score:.2f})"


def is_match(name: str) -> bool:
    return name not in ("No Match", "No DB records")