"""
One-time migration:
- Ensure every person represented in Qdrant exists in SQL persons table.
- Backfill missing/legacy Qdrant payload fields with person_id + display_name.

Run:
    venv\\Scripts\\python migrate_qdrant_to_sql.py
"""

from __future__ import annotations

import uuid

from qdrant_client.models import PointStruct

from config import QDRANT_COLLECTION
from modules.database import _get_client
from modules.persons_db import create_person, get_person_by_name, upsert_person


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID((value or "").strip())
        return True
    except ValueError:
        return False


def _canonical_display_name(payload: dict) -> str:
    return str(payload.get("display_name") or payload.get("name") or "").strip()


def migrate() -> None:
    client = _get_client()
    total_points = 0
    total_rows_ensured = 0
    total_points_updated = 0
    skipped_points = 0

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=True,
            with_vectors=True,
            limit=256,
            offset=offset,
        )
        if not points:
            if offset is None:
                break
            continue

        updates: list[PointStruct] = []

        for p in points:
            total_points += 1
            payload = dict(p.payload or {})

            display_name = _canonical_display_name(payload)
            if not display_name:
                skipped_points += 1
                continue

            raw_person_id = str(payload.get("person_id") or "").strip()
            if _is_uuid(raw_person_id):
                person = upsert_person(raw_person_id, display_name)
            else:
                person = get_person_by_name(display_name) or create_person(display_name)

            if not person:
                skipped_points += 1
                continue

            total_rows_ensured += 1
            target_person_id = person["person_id"]
            target_display_name = person["display_name"]

            changed = False
            if str(payload.get("person_id") or "").strip() != target_person_id:
                payload["person_id"] = target_person_id
                changed = True
            if str(payload.get("display_name") or "").strip() != target_display_name:
                payload["display_name"] = target_display_name
                changed = True

            # Keep stable name key for compatibility in existing code paths.
            if not str(payload.get("name") or "").strip():
                payload["name"] = target_display_name.lower()
                changed = True

            if changed:
                updates.append(
                    PointStruct(
                        id=p.id,
                        vector=[float(v) for v in p.vector],
                        payload=payload,
                    )
                )

        if updates:
            client.upsert(collection_name=QDRANT_COLLECTION, points=updates)
            total_points_updated += len(updates)

        if offset is None:
            break

    print("Migration complete.")
    print(f"Qdrant points scanned : {total_points}")
    print(f"SQL rows ensured      : {total_rows_ensured}")
    print(f"Qdrant points updated : {total_points_updated}")
    print(f"Skipped points        : {skipped_points}")


if __name__ == "__main__":
    migrate()
