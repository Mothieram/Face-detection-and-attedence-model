# FaceID Attendance API Reference

This document is written for developers who receive this project and need to quickly build business logic on top of the API.

It covers:

- How to call each endpoint
- What each endpoint returns
- How to use responses in real business flows (registration, identity, attendance)

**Base URL:** `http://localhost:8000/v1`  
**Swagger UI:** `http://localhost:8000/docs`  
**Auth header:** `X-API-Key: <your_key>`

---

## 1. Developer Quick Start

### 1.1 Authentication

When `api.auth.enabled=true`, all `/v1/*` routes require an API key.

```http
X-API-Key: your_key_here
```

Key types:

| Key type | Access                                                                      |
| -------- | --------------------------------------------------------------------------- |
| Admin    | Full access                                                                 |
| Readonly | `status`, `detections`, `liveness`, `persons` read, `matches`, `attendance` |

Admin-only routes:

- `POST /v1/persons`
- `DELETE /v1/persons/{name}`

### 1.2 Request content type

All image routes expect `multipart/form-data`.

```js
const fd = new FormData();
fd.append("image", fileBlob, "frame.jpg");
fd.append("lat", "11.019751"); // optional
fd.append("lon", "76.978664"); // optional
```

### 1.3 Common error shape

Most errors follow this envelope:

```json
{
  "error": "ERROR_CODE",
  "message": "Human readable explanation",
  "field": "image",
  "code": "ERROR_CODE"
}
```

Some errors include `detail` with additional context.

---

## 2. Business Logic Flows

### 2.1 Registration flow (new employee onboarding)

1. `GET /v1/persons/{name}` to check if already exists.
2. If 404, call `POST /v1/persons` with image + name.
3. On success (`201`), store `name`, `face_id`, and optionally `geo` metadata.
4. On `LIVENESS_FAILED`, ask user to retake image.

### 2.2 Identification flow (no attendance marking)

1. Call `POST /v1/faces/matches`.
2. Loop over `results`.
3. Use:

- `matched: true` for known identities
- `name` for UI labels
- `score` for confidence filtering

### 2.3 Attendance flow

1. Call `POST /v1/attendance/records`.
2. Use:

- `present` for final attendance list
- `detail` for per-face audit
- `already_logged` to avoid duplicate marking in your ERP/HRMS

---

## 3. Endpoint Reference

## GET /v1/status

Health + model readiness + matching thresholds.

**Auth:** Any valid key  
**Rate limit:** 60/min

### Response `200`

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

Use this endpoint at app startup to decide if capture should be enabled.

---

## POST /v1/faces/detections

Detect all faces and quality stats. No matching, no attendance marking.

**Auth:** Any valid key  
**Rate limit:** 120/min

### Request fields

| Field   | Type | Required | Description          |
| ------- | ---- | -------- | -------------------- |
| `image` | file | Yes      | JPEG or PNG (<=10MB) |

### Response `200`

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

Typical UI logic:

- Enable capture only when exactly one face has `quality.passed=true`.

---

## POST /v1/faces/liveness

Runs liveness on the largest detected face.

**Auth:** Any valid key  
**Rate limit:** 20/min

### Request fields

| Field          | Type | Required | Default | Description                       |
| -------------- | ---- | -------- | ------- | --------------------------------- |
| `image`        | file | Yes      | -       | JPEG/PNG image                    |
| `passive_only` | bool | No       | `true`  | Skip active challenge             |
| `camera_index` | int  | No       | `0`     | Camera index for active challenge |

### Response `200`

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

Business usage:

- If `passed=false`, block registration/match and ask user to retry.

---

## POST /v1/persons

Register person with full pipeline (detect -> quality -> liveness -> embed -> save).

**Auth:** Admin only  
**Rate limit:** 10/min  
**Success status:** `201`

### Request fields

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

### Response `201`

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

### Liveness failed example `403`

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

Business note:

- Always inspect `embedding_mode`. `fallback` means lower quality embedding.

---

## GET /v1/persons

Cursor-based list of persons.

**Auth:** Any valid key  
**Rate limit:** 30/min

### Query params

| Param              | Type   | Default | Notes   |
| ------------------ | ------ | ------- | ------- |
| `limit`            | int    | `50`    | max 500 |
| `after_created_at` | string | -       | cursor  |
| `after_person_id`  | string | -       | cursor  |

### Response `200`

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

Business usage:

- Admin user-management screens.
- Export employees in pages.

---

## GET /v1/persons/{name}

Checks if person exists.

**Auth:** Any valid key  
**Rate limit:** 60/min

### Response `200`

```json
{
  "name": "Mothieram",
  "exists": true
}
```

### Response `404`

```json
{
  "error": "PERSON_NOT_FOUND",
  "message": "'Ravi' is not registered. Check the spelling or call GET /v1/persons to list all registered names.",
  "field": "name",
  "code": "PERSON_NOT_FOUND"
}
```

Business usage:

- Pre-check before registration.

---

## DELETE /v1/persons/{name}

Delete person from SQL + vector templates.

**Auth:** Admin only  
**Rate limit:** 10/min  
**Response on success:** `204 No Content`

---

## POST /v1/faces/matches

Identifies all faces from image.

**Auth:** Any valid key  
**Rate limit:** 30/min

### Request fields

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

### Response `200`

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

`results[].name` can be:

- matched user name
- `"No Match"`
- `"Rejected"` (quality failed)
- `"Spoof"` or `"Uncertain"` (liveness failed)

Business usage:

- Visitor recognition
- Access control pre-check

---

## POST /v1/attendance/records

Matching + attendance-ready output with duplicate prevention.

**Auth:** Any valid key  
**Rate limit:** 30/min

### Request fields

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

### Response `200`

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
      "status": "already_logged",
      "name": "Anish",
      "score": 0.8412
    }
  ]
}
```

`detail[].status` values:

- `matched`
- `already_logged`
- `unknown`
- `spoofed`

Important:

- `total_faces` here is quality-passed faces processed in attendance pipeline.

Business usage:

- Directly sync `present` to payroll/HRMS.
- Store `detail` for compliance/audit.

---

## GET /v1/db/tier

Current vector DB scaling tier metadata.

**Auth:** Any valid key  
**Rate limit:** 30/min

### Response `200`

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

## 4. Error Codes

| Code                       | HTTP | Meaning                     |
| -------------------------- | ---- | --------------------------- |
| `MISSING_API_KEY`          | 401  | Missing API key             |
| `INVALID_API_KEY`          | 403  | Invalid key                 |
| `INSUFFICIENT_PERMISSIONS` | 403  | Readonly key on admin route |
| `GEOFENCE_DENIED`          | 403  | Outside allowed geofence    |
| `LIVENESS_FAILED`          | 403  | Liveness failed             |
| `FILE_TOO_LARGE`           | 413  | Upload over 10MB            |
| `RATE_LIMITED`             | 429  | Too many requests           |
| `INVALID_IMAGE`            | 400  | Invalid image bytes         |
| `NO_FACE_DETECTED`         | 422  | No face found               |
| `FACE_QUALITY_FAILED`      | 422  | All faces failed quality    |
| `PERSON_ALREADY_EXISTS`    | 409  | Name already exists         |
| `PERSON_NOT_FOUND`         | 404  | Person not found            |
| `INVALID_NAME`             | 422  | Name not valid              |
| `VALIDATION_ERROR`         | 422  | Request validation failure  |
| `DATABASE_ERROR`           | 500  | SQL operation failed        |
| `EMBEDDING_SAVE_FAILED`    | 500  | Vector save failed          |
| `INTERNAL_ERROR`           | 500  | Unexpected internal failure |

---

## 5. Rate Limits

| Route                                                                                         | Limit   |
| --------------------------------------------------------------------------------------------- | ------- |
| `POST /v1/persons`, `DELETE /v1/persons/{name}`                                               | 10/min  |
| `POST /v1/faces/liveness`                                                                     | 20/min  |
| `POST /v1/faces/matches`, `POST /v1/attendance/records`, `GET /v1/persons`, `GET /v1/db/tier` | 30/min  |
| `GET /v1/status`, `GET /v1/persons/{name}`                                                    | 60/min  |
| `POST /v1/faces/detections`                                                                   | 120/min |
| Global default                                                                                | 200/min |

---

## 6. Geo Object Reference

Endpoints that accept geo fields (`lat`, `lon`, `accuracy_m`, `zone_id`, `skip_geocode`) can return:

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

---

## 7. Implementation Checklist

Use this when integrating:

1. Call `/v1/status` on startup.
2. For registration UI:

- pre-check with `/v1/persons/{name}`
- register with `/v1/persons`

3. For live attendance:

- preview with `/v1/faces/detections`
- commit scan with `/v1/attendance/records`

4. Persist API response metadata you care about:

- `timestamp`, `present`, `detail`, `geo`, `score`

5. Handle key error codes explicitly in UI:

- `LIVENESS_FAILED`
- `NO_FACE_DETECTED`
- `FACE_QUALITY_FAILED`
- `RATE_LIMITED`
