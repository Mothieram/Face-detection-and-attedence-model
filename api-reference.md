# FaceID Attendance System — API Reference

A complete developer reference for the Face Recognition Attendance Pipeline REST API. This document covers every endpoint, request format, response shape, error codes, and integration patterns so you can build your own business logic on top of the pipeline.

**Base URL:** `http://localhost:8000/v1`
**Interactive docs:** `http://localhost:8000/docs`
**Auth header:** `X-API-Key: <your_key>`

---

## Table of Contents

- [Authentication](#authentication)
- [Request format](#request-format)
- [Response envelope](#response-envelope)
- [Error codes](#error-codes)
- [Rate limits](#rate-limits)
- [Endpoints](#endpoints)
  - [GET /v1/status](#get-v1status)
  - [POST /v1/faces/detections](#post-v1facesdetections)
  - [POST /v1/faces/liveness](#post-v1facesliveness)
  - [POST /v1/persons](#post-v1persons)
  - [GET /v1/persons](#get-v1persons)
  - [GET /v1/persons/{name}](#get-v1personsname)
  - [DELETE /v1/persons/{name}](#delete-v1personsname)
  - [POST /v1/faces/matches](#post-v1facesmatches)
  - [POST /v1/attendance/records](#post-v1attendancerecords)
  - [GET /v1/db/tier](#get-v1dbtier)
- [Integration patterns](#integration-patterns)
- [Geo object reference](#geo-object-reference)

---

## Authentication

All endpoints require the `X-API-Key` header. Two key tiers are supported:

| Key type     | Access                                            |
| ------------ | ------------------------------------------------- |
| **Admin**    | Full access — register, delete, match, attendance |
| **Readonly** | Match, detect, liveness, list, status only        |

```http
X-API-Key: your_key_here
```

Generate keys:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Configure in `application.properties`:

```properties
api.auth.keys          = YOUR_ADMIN_KEY
api.auth.keys_readonly = YOUR_READONLY_KEY
```

---

## Request format

All image endpoints use `multipart/form-data`. Never send JSON for image endpoints.

```js
const fd = new FormData();
fd.append("image", blob, "frame.jpg"); // required — JPEG or PNG, max 10 MB
fd.append("lat", 11.019751); // optional — GPS latitude
fd.append("lon", 76.978664); // optional — GPS longitude
```

---

## Response envelope

All error responses return the same JSON shape:

```json
{
  "error": "ERROR_CODE",
  "message": "Human readable explanation of what went wrong.",
  "field": "image",
  "code": "ERROR_CODE"
}
```

---

## Error codes

| Code                       | HTTP | Meaning                                        |
| -------------------------- | ---- | ---------------------------------------------- |
| `MISSING_API_KEY`          | 401  | No `X-API-Key` header sent                     |
| `INVALID_API_KEY`          | 403  | Key not recognised                             |
| `INSUFFICIENT_PERMISSIONS` | 403  | Readonly key used on admin endpoint            |
| `GEOFENCE_DENIED`          | 403  | Device outside allowed zone                    |
| `LIVENESS_FAILED`          | 403  | Face failed anti-spoofing check                |
| `FILE_TOO_LARGE`           | 413  | Image exceeds 10 MB limit                      |
| `RATE_LIMITED`             | 429  | Too many requests — slow down                  |
| `INVALID_IMAGE`            | 400  | Cannot decode uploaded file                    |
| `NO_FACE_DETECTED`         | 422  | No face found in image                         |
| `FACE_QUALITY_FAILED`      | 422  | Face too small, blurry, or angled              |
| `PERSON_ALREADY_EXISTS`    | 409  | Name already registered                        |
| `PERSON_NOT_FOUND`         | 404  | Name not in database                           |
| `VALIDATION_ERROR`         | 422  | Wrong field type or missing required field     |
| `INTERNAL_ERROR`           | 500  | Unexpected server error — check `face_api.log` |

---

## Rate limits

| Endpoint                                                | Limit        |
| ------------------------------------------------------- | ------------ |
| `POST /v1/persons`, `DELETE /v1/persons/{name}`         | 10 / minute  |
| `POST /v1/faces/liveness`                               | 20 / minute  |
| `POST /v1/faces/matches`, `POST /v1/attendance/records` | 30 / minute  |
| `GET /v1/status`, `POST /v1/faces/detections`           | 60 / minute  |
| Global default                                          | 200 / minute |

---

## Endpoints

---

### GET /v1/status

Health check. Call this on page load to verify all models are ready before allowing users to scan.

**Auth:** Any key

**Request:** No body

**Response: 200**

```json
{
  "cvlface_loaded": true,
  "liveness_enabled": true,
  "active_liveness_enabled": true,
  "db_tier": {
    "tier_index": 0,
    "strategy": "single",
    "cap": 1,
    "hnsw_m": 8,
    "hnsw_ef": 100,
    "search_ef": 64,
    "people_min": 0,
    "people_max": 99
  },
  "match_thresholds": {
    "cvlface": 0.55,
    "fallback": 0.94
  }
}
```

| Field                     | Type   | Description                                 |
| ------------------------- | ------ | ------------------------------------------- |
| `cvlface_loaded`          | bool   | AdaFace IR-101 model is loaded and ready    |
| `liveness_enabled`        | bool   | Anti-spoofing is active                     |
| `active_liveness_enabled` | bool   | Head-pose challenge is enabled              |
| `db_tier`                 | object | Current Qdrant HNSW scaling tier            |
| `match_thresholds`        | object | Minimum cosine similarity to accept a match |

**Business logic tip:** Show a loading state until `cvlface_loaded: true`. If false, embedding will fall back to a lower-accuracy mode.

---

### POST /v1/faces/detections

Detect all faces in an image. Returns bounding boxes, landmarks, and quality scores. Use this for live camera preview overlay — does not run liveness or matching.

**Auth:** Any key
**Rate limit:** 60 / minute

**Request fields:**

| Field   | Type | Required | Description            |
| ------- | ---- | -------- | ---------------------- |
| `image` | file | ✅       | JPEG or PNG, max 10 MB |

**Response: 200**

```json
{
  "faces": [
    {
      "face_id": "face_1",
      "score": 0.9923,
      "bbox": [120, 45, 310, 290],
      "landmarks": {
        "right_eye": [158.3, 142.1],
        "left_eye": [252.7, 139.8],
        "nose": [205.2, 189.4],
        "mouth_right": [168.9, 234.6],
        "mouth_left": [243.1, 232.3]
      },
      "quality": {
        "passed": true,
        "reason": "OK",
        "face_size": 190,
        "blur_score": 142.8,
        "yaw_deg": 8.3
      }
    }
  ],
  "total": 1
}
```

| Field                | Type   | Description                                                     |
| -------------------- | ------ | --------------------------------------------------------------- |
| `face_id`            | string | Identifier for this face in the frame — `face_1`, `face_2` etc. |
| `score`              | float  | RetinaFace detection confidence (0–1)                           |
| `bbox`               | int[4] | Bounding box `[x1, y1, x2, y2]` in pixels                       |
| `landmarks`          | object | 5-point facial landmarks in pixel coordinates                   |
| `quality.passed`     | bool   | Whether face meets size, blur, and yaw requirements             |
| `quality.reason`     | string | `"OK"` or reason for failure                                    |
| `quality.face_size`  | int    | Short side of face bounding box in pixels                       |
| `quality.blur_score` | float  | Laplacian variance — higher = sharper                           |
| `quality.yaw_deg`    | float  | Head turn angle in degrees                                      |

**Business logic tip:** Use `quality.passed` to decide whether to enable your capture button. Only allow capture when exactly 1 face is detected with `quality.passed: true`.

---

### POST /v1/faces/liveness

Run passive anti-spoofing on the largest face in the image. Use before any registration or matching if you need an explicit liveness pre-check.

**Auth:** Any key
**Rate limit:** 20 / minute

**Request fields:**

| Field          | Type | Required | Default | Description                     |
| -------------- | ---- | -------- | ------- | ------------------------------- |
| `image`        | file | ✅       | —       | JPEG or PNG                     |
| `passive_only` | bool | ❌       | `true`  | Skip active head-pose challenge |
| `camera_index` | int  | ❌       | `0`     | Camera for active challenge     |

**Response: 200**

```json
{
  "passed": true,
  "method": "passive",
  "score": 0.8821,
  "reason": "Passive ensemble: real",
  "model_scores": {
    "ICM2O": 0.0312,
    "IOM2C": 0.0287,
    "modelrgb": 0.0421,
    "SASF": 0.0198
  }
}
```

| Field          | Type   | Description                                      |
| -------------- | ------ | ------------------------------------------------ |
| `passed`       | bool   | `true` = real face, `false` = spoof or uncertain |
| `method`       | string | `"passive"` or `"passive+active"`                |
| `score`        | float  | Real confidence (0–1). Higher = more real        |
| `reason`       | string | Human readable result explanation                |
| `model_scores` | object | Per-model spoof probability (lower = more real)  |

**Business logic tip:** `model_scores` values are spoof probabilities. A value of `0.03` means 3% chance of spoof — very real. Use these for debugging if liveness keeps failing for legitimate users.

---

### POST /v1/persons

Register a new person. Runs the full pipeline — detect → quality check → liveness → embed → save.

**Auth:** Admin key only
**Rate limit:** 10 / minute
**Status:** 201 Created on success

**Request fields:**

| Field           | Type   | Required | Default | Description                                                        |
| --------------- | ------ | -------- | ------- | ------------------------------------------------------------------ |
| `image`         | file   | ✅       | —       | JPEG or PNG of the person                                          |
| `name`          | string | ✅       | —       | Display name (letters, digits, spaces, `_-` allowed, max 64 chars) |
| `face_index`    | int    | ❌       | `0`     | Which face to register when multiple detected                      |
| `skip_liveness` | bool   | ❌       | `false` | Bypass liveness check (not recommended for production)             |
| `passive_only`  | bool   | ❌       | `true`  | Passive liveness only                                              |
| `lat`           | float  | ❌       | —       | GPS latitude                                                       |
| `lon`           | float  | ❌       | —       | GPS longitude                                                      |
| `accuracy_m`    | float  | ❌       | —       | GPS accuracy in metres                                             |
| `skip_geocode`  | bool   | ❌       | `false` | Skip reverse geocoding                                             |

**Response: 201**

```json
{
  "success": true,
  "name": "Mothieram",
  "face_id": "face_1",
  "liveness": {
    "passed": true,
    "method": "passive",
    "score": 0.9102,
    "reason": "Passive ensemble: real",
    "model_scores": {
      "ICM2O": 0.021,
      "IOM2C": 0.018,
      "modelrgb": 0.034,
      "SASF": 0.012
    }
  },
  "geo": {},
  "embedding_mode": "cvlface",
  "message": "'Mothieram' registered successfully."
}
```

**Liveness failed response: 403**

```json
{
  "error": "LIVENESS_FAILED",
  "message": "Liveness check failed: Passive ensemble: spoof detected. Ensure the face is real, well-lit, and looking at the camera.",
  "field": "image",
  "code": "LIVENESS_FAILED",
  "detail": {
    "face_id": "face_1",
    "liveness": { "passed": false, "score": 0.21, "model_scores": {} },
    "geo": {}
  }
}
```

| Field            | Type   | Description                                                               |
| ---------------- | ------ | ------------------------------------------------------------------------- |
| `success`        | bool   | Registration successful                                                   |
| `name`           | string | Registered display name                                                   |
| `embedding_mode` | string | `"cvlface"` (AdaFace IR-101) or `"fallback"` (lower accuracy)             |
| `liveness`       | object | Full liveness result — see liveness endpoint                              |
| `geo`            | object | Geo tag at time of registration — see [Geo object](#geo-object-reference) |

**Business logic tip:** Always check `embedding_mode`. If `"fallback"`, the AdaFace model was unavailable and accuracy will be lower. Prompt the user to re-register when the model is available.

---

### GET /v1/persons

List all registered persons. Cursor-paginated — O(log n) on every page.

**Auth:** Any key
**Rate limit:** 30 / minute

**Query parameters:**

| Parameter          | Type   | Default | Description                        |
| ------------------ | ------ | ------- | ---------------------------------- |
| `limit`            | int    | `50`    | Records per page (max 500)         |
| `after_created_at` | string | —       | Cursor from previous `next_cursor` |
| `after_person_id`  | string | —       | Cursor from previous `next_cursor` |

**Response: 200**

```json
{
  "records": [
    {
      "person_id": "713c1055-6f68-476f-8ef7-ae24c2196a5d",
      "name_key": "mothieram",
      "display_name": "mothieram",
      "created_at": "2026-03-19 18:12:35.564739+05:30"
    },
    {
      "person_id": "ff86ca16-fc06-42f4-9702-3dfe13924c58",
      "name_key": "anish",
      "display_name": "Anish",
      "created_at": "2026-03-19 18:12:35.907202+05:30"
    }
  ],
  "limit": 50,
  "has_more": false,
  "next_cursor": null
}
```

| Field          | Type           | Description                                        |
| -------------- | -------------- | -------------------------------------------------- |
| `records`      | array          | List of registered persons                         |
| `person_id`    | string         | UUID — stable identifier, use this for all lookups |
| `name_key`     | string         | Lowercase normalised name                          |
| `display_name` | string         | Original case display name                         |
| `has_more`     | bool           | `true` if more records exist beyond this page      |
| `next_cursor`  | object \| null | Pass these values to get the next page             |

**Pagination example:**

```js
// First page
GET /v1/persons?limit=10

// Next page — use next_cursor from previous response
GET /v1/persons?limit=10&after_created_at=2026-03-19T18:12:35&after_person_id=713c1055-...
```

---

### GET /v1/persons/{name}

Check if a specific person is registered. Use before registration to avoid duplicates.

**Auth:** Any key
**Rate limit:** 60 / minute

**Path parameter:** `name` — the display name to look up

**Response: 200 — found**

```json
{
  "name": "Mothieram",
  "exists": true
}
```

**Response: 404 — not found**

```json
{
  "error": "PERSON_NOT_FOUND",
  "message": "'Ravi' is not registered. Check the spelling or call GET /v1/persons to list all registered names.",
  "field": "name",
  "code": "PERSON_NOT_FOUND"
}
```

**Business logic tip:** Call this before showing the register form to give users a clear "already registered" message instead of a generic 409 error.

---

### DELETE /v1/persons/{name}

Delete a person and all their face embeddings from the database.

**Auth:** Admin key only
**Rate limit:** 10 / minute
**Status:** 204 No Content on success (empty body)

**Path parameter:** `name` — the display name to delete

**Response: 204** — empty body

**Response: 404** — person not found (same shape as above)

**Business logic tip:** Deletion is permanent and removes all Qdrant embeddings + SQL record. To re-register the same name, call `POST /v1/persons` again after deletion.

---

### POST /v1/faces/matches

Identify all faces in an image against the database. Returns one result per detected face. Use this for the identification-only flow (no attendance logging).

**Auth:** Any key
**Rate limit:** 30 / minute

**Request fields:**

| Field           | Type  | Required | Default | Description                              |
| --------------- | ----- | -------- | ------- | ---------------------------------------- |
| `image`         | file  | ✅       | —       | JPEG or PNG                              |
| `skip_liveness` | bool  | ❌       | `false` | Bypass liveness check                    |
| `passive_only`  | bool  | ❌       | `true`  | Passive liveness only                    |
| `auto_update`   | bool  | ❌       | `true`  | Auto-update templates on confident match |
| `lat`           | float | ❌       | —       | GPS latitude                             |
| `lon`           | float | ❌       | —       | GPS longitude                            |
| `accuracy_m`    | float | ❌       | —       | GPS accuracy in metres                   |
| `skip_geocode`  | bool  | ❌       | `false` | Skip reverse geocoding                   |

**Response: 200**

```json
{
  "results": [
    {
      "face_id": "face_1",
      "matched": true,
      "name": "Mothieram",
      "score": 0.8731,
      "liveness": {
        "passed": true,
        "method": "passive",
        "score": 0.9102,
        "reason": "Passive ensemble: real",
        "model_scores": {
          "ICM2O": 0.021,
          "IOM2C": 0.018,
          "modelrgb": 0.034,
          "SASF": 0.012
        }
      },
      "geo": {},
      "embedding_mode": "cvlface",
      "auto_updated": false,
      "bbox": [120, 45, 310, 290],
      "landmarks": {},
      "quality": {
        "passed": true,
        "reason": "OK",
        "face_size": 190,
        "blur_score": 142.8,
        "yaw_deg": 8.3
      }
    }
  ],
  "total_faces": 1,
  "matched_count": 1,
  "geo": {}
}
```

| Field           | Type   | Description                                                                                                            |
| --------------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| `matched`       | bool   | `true` = person found in database                                                                                      |
| `name`          | string | Display name if matched. `"No Match"` if below threshold. `"Spoof"` if liveness failed. `"Rejected"` if quality failed |
| `score`         | float  | Cosine similarity (0–1). Higher = more confident match                                                                 |
| `auto_updated`  | bool   | `true` if a new embedding template was added for this person                                                           |
| `total_faces`   | int    | Total faces detected in the image                                                                                      |
| `matched_count` | int    | Number of faces that matched a registered person                                                                       |

**Score interpretation:**

| Score range   | Meaning                          |
| ------------- | -------------------------------- |
| `>= 0.80`     | High confidence match            |
| `0.55 – 0.80` | Accepted match (above threshold) |
| `< 0.55`      | No match — returns `"No Match"`  |

**Business logic tip:** Use `matched_count` for group photo scenarios. Filter `results` by `matched: true` to get only identified people.

---

### POST /v1/attendance/records

Match all faces and log attendance. Same pipeline as `/v1/faces/matches` but with duplicate prevention — the same person cannot be counted twice within the cooldown window (default 30 minutes).

**Auth:** Any key
**Rate limit:** 30 / minute

**Request fields:** Same as `/v1/faces/matches`

**Response: 200**

```json
{
  "timestamp": 1742891234,
  "present": ["Mothieram", "Anish"],
  "present_count": 2,
  "unknown_count": 0,
  "spoofed_count": 0,
  "total_faces": 2,
  "geo": {},
  "detail": [
    {
      "face_id": "face_1",
      "status": "matched",
      "name": "Mothieram",
      "score": 0.8731,
      "mode": "cvlface",
      "liveness_score": 0.9102
    },
    {
      "face_id": "face_2",
      "status": "matched",
      "name": "Anish",
      "score": 0.8412,
      "mode": "cvlface",
      "liveness_score": 0.8876
    }
  ]
}
```

| Field           | Type     | Description                                                     |
| --------------- | -------- | --------------------------------------------------------------- |
| `timestamp`     | int      | Unix timestamp of the attendance scan                           |
| `present`       | string[] | Deduplicated list of matched names logged this scan             |
| `present_count` | int      | Number of unique people marked present                          |
| `unknown_count` | int      | Faces detected but not matched to any registered person         |
| `spoofed_count` | int      | Faces that failed liveness check                                |
| `total_faces`   | int      | Total faces detected in image                                   |
| `geo`           | object   | Geo tag for this scan — see [Geo object](#geo-object-reference) |
| `detail`        | array    | Per-face breakdown with individual status                       |

**Detail status values:**

| Status           | Meaning                                                           |
| ---------------- | ----------------------------------------------------------------- |
| `matched`        | Person identified and logged present                              |
| `already_logged` | Person already logged within cooldown window — not double counted |
| `unknown`        | Face detected but no match in database                            |
| `spoofed`        | Face failed liveness check                                        |

**Business logic tip:** Use `present` array to mark attendance in your own HR or ERP system. Use `detail` array for audit logs. `already_logged` status means the person tried to scan twice — you can flag this if needed.

---

### GET /v1/db/tier

Returns the current Qdrant HNSW scaling tier based on number of registered persons.

**Auth:** Any key
**Rate limit:** 30 / minute

**Response: 200**

```json
{
  "tier_index": 0,
  "strategy": "single",
  "cap": 1,
  "hnsw_m": 8,
  "hnsw_ef": 100,
  "search_ef": 64,
  "people_min": 0,
  "people_max": 99
}
```

| Tier | People      | Strategy | Notes                        |
| ---- | ----------- | -------- | ---------------------------- |
| 0    | 0 – 99      | single   | One embedding per person     |
| 1    | 100 – 999   | multi    | Up to 5 templates per person |
| 2    | 1000 – 9999 | hybrid   | Consolidates to mean + best  |
| 3    | 10000+      | mean     | Running average embedding    |

---

## Integration patterns

### Pattern 1 — Attendance kiosk (most common)

```
1. GET  /v1/status              → verify server ready
2. POST /v1/faces/detections    → live preview loop (every 700ms)
3. POST /v1/attendance/records  → on capture button press
4. Read present[] from response → sync to your HR/ERP system
```

### Pattern 2 — Registration flow

```
1. GET  /v1/persons/{name}  → check if already registered
2. If not found:
   POST /v1/persons         → register with image + name
3. On 201 → show success
   On 403 LIVENESS_FAILED   → ask user to retry in better lighting
   On 409 PERSON_EXISTS     → show "already registered" message
```

### Pattern 3 — Identification only (no attendance logging)

```
1. POST /v1/faces/matches   → identify who is in the frame
2. Read results[]           → filter by matched: true
3. Use name field           → display or pass to your system
```

### Pattern 4 — Bulk person management

```
1. GET /v1/persons?limit=500        → list all registered persons
2. If has_more: true                → paginate using next_cursor
3. DELETE /v1/persons/{name}        → remove specific person
```

---

## Geo object reference

Every endpoint that accepts `lat` and `lon` returns a `geo` object in the response:

```json
{
  "coordinates": {
    "lat": 11.019751,
    "lon": 76.978664,
    "source": "browser",
    "accuracy_m": 12.4
  },
  "address": "Avinashi Road, Peelamedu, Coimbatore, Tamil Nadu, India",
  "city": "Coimbatore",
  "country": "India",
  "country_code": "IN",
  "timezone": "Asia/Kolkata",
  "local_time": "2026-03-24T10:30:45+05:30",
  "geofence_zone": "office",
  "geofence_passed": true,
  "geofence_reason": "inside_zone",
  "geocode_cached": false,
  "error": null
}
```

| Field             | Type           | Description                                                         |
| ----------------- | -------------- | ------------------------------------------------------------------- |
| `coordinates`     | object \| null | Raw GPS coordinates from browser                                    |
| `address`         | string         | Full reverse-geocoded address                                       |
| `city`            | string         | City name                                                           |
| `country`         | string         | Country name                                                        |
| `timezone`        | string         | IANA timezone string                                                |
| `local_time`      | string         | ISO 8601 local timestamp at scan location                           |
| `geofence_zone`   | string \| null | Zone ID the device is inside                                        |
| `geofence_passed` | bool           | `true` if inside allowed zone or enforcement disabled               |
| `geofence_reason` | string         | `"inside_zone"`, `"outside_all_zones"`, `"no_coordinates_provided"` |
| `geocode_cached`  | bool           | `true` if address came from cache                                   |

**Business logic tip:** Store `local_time` in your attendance records — it reflects the actual local time at the scan location, not server time. Use `geofence_passed: false` to flag suspicious scans from outside the office.
