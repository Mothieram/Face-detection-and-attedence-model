# FaceID Attendance System - API Reference

Developer reference for the Face Recognition Attendance Pipeline REST API (`api.py`).

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
- [Geo object reference](#geo-object-reference)

---

## Authentication

When `api.auth.enabled=true`, all `/v1/*` endpoints require `X-API-Key`.

| Key type     | Access                                                                    |
| ------------ | ------------------------------------------------------------------------- |
| **Admin**    | Full access                                                               |
| **Readonly** | `status`, `detect`, `liveness`, `list/get persons`, `match`, `attendance` |

```http
X-API-Key: your_key_here
```

Configure in `application.properties`:

```properties
api.auth.keys=YOUR_ADMIN_KEY
api.auth.keys_readonly=YOUR_READONLY_KEY
```

---

## Request format

Image endpoints use `multipart/form-data`.

```js
const fd = new FormData();
fd.append("image", blob, "frame.jpg"); // required
fd.append("lat", 11.019751); // optional
fd.append("lon", 76.978664); // optional
```

Upload limit: **10 MB** (returns `413 FILE_TOO_LARGE`).

---

## Response envelope

Errors use:

```json
{
  "error": "ERROR_CODE",
  "message": "Human readable message",
  "field": "image",
  "code": "ERROR_CODE"
}
```

Some errors include optional extra payload in `detail`.

---

## Error codes

| Code                       | HTTP | Meaning                          |
| -------------------------- | ---- | -------------------------------- |
| `MISSING_API_KEY`          | 401  | No `X-API-Key` header            |
| `INVALID_API_KEY`          | 403  | Key not recognized               |
| `INSUFFICIENT_PERMISSIONS` | 403  | Readonly key used on admin route |
| `GEOFENCE_DENIED`          | 403  | Outside allowed geofence         |
| `LIVENESS_FAILED`          | 403  | Liveness failed                  |
| `FILE_TOO_LARGE`           | 413  | Payload exceeds 10 MB            |
| `RATE_LIMITED`             | 429  | Rate limit exceeded              |
| `INVALID_IMAGE`            | 400  | Cannot decode image              |
| `NO_FACE_DETECTED`         | 422  | No face detected                 |
| `FACE_QUALITY_FAILED`      | 422  | All faces failed quality checks  |
| `PERSON_ALREADY_EXISTS`    | 409  | Name already registered          |
| `PERSON_NOT_FOUND`         | 404  | Person not found                 |
| `INVALID_NAME`             | 422  | Name contains invalid characters |
| `VALIDATION_ERROR`         | 422  | Request validation failed        |
| `DATABASE_ERROR`           | 500  | SQL store operation failed       |
| `EMBEDDING_SAVE_FAILED`    | 500  | Vector DB save failed            |
| `INTERNAL_ERROR`           | 500  | Unhandled server error           |

---

## Rate limits

| Endpoint                                                | Limit   |
| ------------------------------------------------------- | ------- |
| `POST /v1/persons`, `DELETE /v1/persons/{name}`         | 10/min  |
| `POST /v1/faces/liveness`                               | 20/min  |
| `POST /v1/faces/matches`, `POST /v1/attendance/records` | 30/min  |
| `GET /v1/status`, `GET /v1/persons/{name}`              | 60/min  |
| `POST /v1/faces/detections`                             | 120/min |
| Global default                                          | 200/min |

---

## Endpoints

### GET /v1/status

Health check and runtime status.

**Auth:** Any valid key

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

---

### POST /v1/faces/detections

Detect faces and return bbox, landmarks, quality.

**Auth:** Any valid key  
**Rate limit:** 120/min

**Request fields**

| Field   | Type | Required | Description           |
| ------- | ---- | -------- | --------------------- |
| `image` | file | Yes      | JPEG or PNG (<=10 MB) |

**Response: 200**

```json
{
  "faces": [
    {
      "face_id": "face_1",
      "score": 0.9923,
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
  "total": 1
}
```

---

### POST /v1/faces/liveness

Run liveness on the largest detected face.

**Auth:** Any valid key  
**Rate limit:** 20/min

**Request fields**

| Field          | Type | Required | Default |
| -------------- | ---- | -------- | ------- |
| `image`        | file | Yes      | -       |
| `passive_only` | bool | No       | `true`  |
| `camera_index` | int  | No       | `0`     |

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

---

### POST /v1/persons

Register a new person.

Flow: decode -> preprocess -> detect -> quality -> liveness -> embed -> save.

**Auth:** Admin key only  
**Rate limit:** 10/min  
**Success status:** `201 Created`

**Request fields**

| Field           | Type   | Required | Default |
| --------------- | ------ | -------- | ------- |
| `image`         | file   | Yes      | -       |
| `name`          | string | Yes      | -       |
| `face_index`    | int    | No       | `0`     |
| `skip_liveness` | bool   | No       | `false` |
| `passive_only`  | bool   | No       | `true`  |
| `camera_index`  | int    | No       | `0`     |
| `lat`           | float  | No       | -       |
| `lon`           | float  | No       | -       |
| `accuracy_m`    | float  | No       | -       |
| `zone_id`       | string | No       | -       |
| `skip_geocode`  | bool   | No       | `false` |

Notes:

- Name allows letters, digits, spaces, hyphen, underscore, apostrophe. Max 64.
- If `face_index` is out of range, server falls back to index `0`.

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
    "model_scores": {}
  },
  "geo": {},
  "embedding_mode": "cvlface",
  "message": "'Mothieram' registered successfully."
}
```

**Liveness fail response: 403**

```json
{
  "error": "LIVENESS_FAILED",
  "message": "Liveness check failed: ...",
  "field": "image",
  "code": "LIVENESS_FAILED",
  "detail": {
    "face_id": "face_1",
    "liveness": {},
    "geo": {}
  }
}
```

---

### GET /v1/persons

List registered persons (cursor pagination).

**Auth:** Any valid key  
**Rate limit:** 30/min

**Query params**

| Param              | Type   | Default | Notes     |
| ------------------ | ------ | ------- | --------- |
| `limit`            | int    | `50`    | max `500` |
| `after_created_at` | string | -       | cursor    |
| `after_person_id`  | string | -       | cursor    |

**Response: 200**

```json
{
  "records": [
    {
      "person_id": "713c1055-6f68-476f-8ef7-ae24c2196a5d",
      "name_key": "mothieram",
      "display_name": "Mothieram",
      "created_at": "2026-03-19 18:12:35.564739+05:30"
    }
  ],
  "limit": 50,
  "has_more": false,
  "next_cursor": null
}
```

---

### GET /v1/persons/{name}

Check if person exists.

**Auth:** Any valid key  
**Rate limit:** 60/min

**Response: 200 (found)**

```json
{
  "name": "Mothieram",
  "exists": true
}
```

**Response: 404 (not found)**

```json
{
  "error": "PERSON_NOT_FOUND",
  "message": "'Ravi' is not registered. Check the spelling or call GET /v1/persons to list all registered names.",
  "field": "name",
  "code": "PERSON_NOT_FOUND"
}
```

---

### DELETE /v1/persons/{name}

Delete person and associated templates.

**Auth:** Admin key only  
**Rate limit:** 10/min  
**Success status:** `204 No Content`

---

### POST /v1/faces/matches

Identify faces in an image against DB.

**Auth:** Any valid key  
**Rate limit:** 30/min

**Request fields**

| Field           | Type   | Required | Default |
| --------------- | ------ | -------- | ------- |
| `image`         | file   | Yes      | -       |
| `skip_liveness` | bool   | No       | `false` |
| `passive_only`  | bool   | No       | `true`  |
| `camera_index`  | int    | No       | `0`     |
| `auto_update`   | bool   | No       | `true`  |
| `lat`           | float  | No       | -       |
| `lon`           | float  | No       | -       |
| `accuracy_m`    | float  | No       | -       |
| `zone_id`       | string | No       | -       |
| `skip_geocode`  | bool   | No       | `false` |

**Response: 200**

```json
{
  "results": [
    {
      "face_id": "face_1",
      "matched": true,
      "name": "Mothieram",
      "score": 0.8731,
      "liveness": {},
      "geo": {},
      "embedding_mode": "cvlface",
      "auto_updated": false,
      "bbox": [120, 45, 310, 290],
      "landmarks": {},
      "quality": {}
    }
  ],
  "total_faces": 1,
  "matched_count": 1,
  "geo": {}
}
```

`name` may be:

- matched display name
- `"No Match"`
- `"Rejected"` (failed quality)
- `"Spoof"` or `"Uncertain"` (failed liveness)

---

### POST /v1/attendance/records

Match faces and return attendance payload with cooldown dedupe.

**Auth:** Any valid key  
**Rate limit:** 30/min

**Request fields**

| Field           | Type   | Required | Default |
| --------------- | ------ | -------- | ------- |
| `image`         | file   | Yes      | -       |
| `skip_liveness` | bool   | No       | `false` |
| `passive_only`  | bool   | No       | `true`  |
| `camera_index`  | int    | No       | `0`     |
| `lat`           | float  | No       | -       |
| `lon`           | float  | No       | -       |
| `accuracy_m`    | float  | No       | -       |
| `zone_id`       | string | No       | -       |
| `skip_geocode`  | bool   | No       | `false` |

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
    }
  ]
}
```

`detail.status` values:

- `matched`
- `already_logged`
- `unknown`
- `spoofed`

Important: `total_faces` here is the number of **quality-passed** faces processed for attendance.

---

### GET /v1/db/tier

Return current HNSW tier info.

**Auth:** Any valid key  
**Rate limit:** 30/min

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

---

## Geo object reference

Endpoints that accept `lat/lon` return `geo`.

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
