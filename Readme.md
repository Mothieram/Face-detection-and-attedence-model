# Face Recognition Attendance System

A production-grade attendance pipeline built on **RetinaFace** (detection) + **AdaFace IR-101** (embedding) + **Qdrant** (vector search), with passive liveness detection, GPS geofencing, PostgreSQL person registry, RBAC authentication, rate limiting, and a FastAPI REST backend.

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture overview](#architecture-overview)
- [Pipeline — step by step](#pipeline--step-by-step)
- [Performance optimisations](#performance-optimisations)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Authentication and RBAC](#authentication-and-rbac)
- [API reference](#api-reference)
- [Error handling](#error-handling)
- [Rate limiting](#rate-limiting)
- [Liveness detection](#liveness-detection)
- [Geofencing and geotagging](#geofencing-and-geotagging)
- [Database and scaling tiers](#database-and-scaling-tiers)
- [Person registry (SQL)](#person-registry-sql)
- [Attendance cooldown store](#attendance-cooldown-store)
- [Data migration](#data-migration)
- [Mobile / ngrok access](#mobile--ngrok-access)
- [Frontend](#frontend)
- [Production deployment](#production-deployment)
- [Running the CLI (main.py)](#running-the-cli-mainpy)
- [Troubleshooting](#troubleshooting)

---

## What it does

1. **Registers** a person — creates a PostgreSQL identity record and stores a 512-d face embedding in Qdrant.
2. **Matches** a face from a live image against the database using HNSW approximate nearest-neighbour search.
3. **Guards** against spoofing with a 4-model passive liveness ensemble (ICM2O, IOM2C, modelrgb ONNX, SASF/MiniFASNet).
4. **Enforces** location with GPS geofencing — circle and polygon zones.
5. **Logs** attendance with reverse-geocoded location, local timezone, and per-face liveness scores.
6. **Prevents duplicate attendance** — same person cannot be counted twice within the cooldown window, enforced across all workers and servers via Redis.
7. **Auto-scales** the Qdrant index across four tiers (1–10K+ people) without manual re-indexing.
8. **Rate-limits** every endpoint to protect ML inference from abuse.
9. **Enforces RBAC** — admin keys can register and delete, readonly keys can only match and view.
10. **Works on mobile** over ngrok tunnels with CORS and interstitial bypass built in.

---

## Architecture overview

```
Browser / Camera
       │
       ▼
   index.html  (12 ES modules — no build step)
       │  HTTP multipart/form-data  +  X-API-Key header
       ▼
   api.py  (FastAPI + slowapi rate limiter)
       │
       ├── Upload size limit middleware  (10 MB max)
       ├── NgrokSkip middleware          (bypasses interstitial for page load)
       │
       ├── /v1/status               → system health check       [any key]
       ├── /v1/faces/detections     → detector.py  (RetinaFace) [any key]
       ├── /v1/faces/liveness       → liveness.py  (4-model)    [any key]
       ├── /v1/faces/matches        → full pipeline             [any key]
       ├── /v1/attendance/records   → full pipeline             [any key]
       ├── /v1/persons    GET       → persons_db.py (SQL)       [any key]
       ├── /v1/persons    POST      → SQL + full pipeline       [admin only]
       ├── /v1/persons    DELETE    → SQL + Qdrant delete       [admin only]
       └── /v1/db/tier              → Qdrant tier info          [any key]

Full pipeline (register / match / attendance):
  preprocessor.py  →  detector.py (early-exit)  →  aligner.py
       →  liveness.py  →  embedder.py (batch)  →  matcher.py
       →  database.py  +  geotagging.py  (parallel via asyncio.gather)
       →  Redis cooldown check  (attendance only)
```

---

## Pipeline — step by step

Every image through `/v1/persons`, `/v1/faces/matches`, or `/v1/attendance/records` runs the same pipeline:

### 1. Preprocess (`preprocessor.py`)

- **Upscale** if smallest dimension < 400 px
- **CLAHE** contrast enhancement in LAB space (low-light robustness)
- **Auto gamma** correction (LUTs precomputed at import — zero per-frame allocation)
- **Bilateral denoise** + **unsharp masking** — skipped in `fast=True` mode used by live preview

### 2. Detect (`detector.py`)

**RetinaFace-PyTorch** runs a multi-threshold cascade (`[0.80, 0.60, 0.45]`) with **early exit** — stops as soon as the first threshold finds a face. Lower thresholds only run on difficult images (dark room, partial occlusion). Returns per-face:

```python
{
  "face_id": "face_1", "score": 0.97,
  "bbox": [x1, y1, x2, y2],
  "landmarks": { "right_eye": [x,y], "left_eye": [x,y], "nose": [x,y],
                 "mouth_right": [x,y], "mouth_left": [x,y] }
}
```

### 3. Quality gate (`aligner.py`)

- **Minimum size** — short side ≥ `MIN_FACE_SIZE` px (default 90)
- **Blur** — Laplacian variance ≥ `MIN_BLUR_SCORE` (default 20.0)
- **Yaw** — head turn ≤ `MAX_YAW_DEGREES` (default 60°)

### 4. Align → 5. Liveness → 6. Embed (batch) → 7. Match → 8. Geotag (parallel) → 9. Cooldown check

See individual sections below.

---

## Performance optimisations

Four targeted optimisations applied to the pipeline:

### 1. RetinaFace early exit (`detector.py`)

The threshold cascade stops as soon as any pass finds a face. On clean, well-lit images (the common case) this means **1 inference pass instead of 3** — ~30% latency reduction on detection.

### 2. Batch embedding (`embedder.py`)

All live faces in a frame are stacked into a single `(N, 3, 112, 112)` tensor and run through AdaFace IR-101 in **one forward pass**. For a group photo with 5 people this is ~5× faster than the previous per-face loop. Single-face scans are unaffected.

```python
# New batch API — used internally in match / attendance endpoints
from modules.embedder import generate_embeddings_batch

aligned_crops = [align_face(raw, f["landmarks"], bbox=f["bbox"]) for f in live_faces]
batch_embeddings, mode = generate_embeddings_batch(aligned_crops)
```

The original `generate_embedding(single_crop)` still works unchanged — it delegates to the batch function internally.

### 3. Parallel geocode + pipeline (`api.py`)

The reverse geocode (Nominatim HTTP call, 200–800 ms) and the face detection pipeline (~500 ms) now run **simultaneously** via `asyncio.gather`:

```
Before: geocode (~500ms) + pipeline (~500ms) = ~1000ms total
After:  max(geocode, pipeline)               = ~500ms total
```

Full address information is always present in the response — no "Unknown" in the geo card.

### 4. Worker-safe attendance cooldown (`api.py` + Redis)

The duplicate prevention cooldown uses a **two-tier store**:

```
Redis available   →  SET key NX EX <ttl>  (atomic, cross-server, multi-region safe)
Redis unavailable →  in-process dict      (single-worker fallback, same as original)
```

The server never fails because of Redis — it silently falls back when Redis is unavailable. Redis is needed only when running multiple workers or multiple servers.

---

## Project structure

```
.
├── api.py                            # FastAPI server — all REST endpoints
├── main.py                           # CLI entry point — live camera mode
├── config.py                         # Loads settings from application.properties
├── applicationpropertiesExample.txt  # Copy → application.properties to configure
├── index.html                        # Browser frontend
├── requirement.txt                   # Pinned pip dependencies + gunicorn
├── start_tunnel.ps1                  # ngrok + uvicorn launcher (Windows)
│
├── modules/
│   ├── preprocessor.py               # Image cleanup
│   ├── detector.py                   # RetinaFace wrapper — early-exit cascade
│   ├── aligner.py                    # 5-pt affine align + quality gate
│   ├── liveness.py                   # 4-model spoof detection ensemble
│   ├── SASF.py                       # MiniFASNet wrapper
│   ├── embedder.py                   # CVLFace AdaFace IR-101 — single + batch API
│   ├── matcher.py                    # Qdrant HNSW search
│   ├── database.py                   # Qdrant client, tier management, auto-update
│   ├── persons_db.py                 # SQL person registry (PostgreSQL)
│   └── geotagging.py                 # GPS → geofence → reverse geocode
│
├── datamigration/
│   └── migrate_qdrant_to_sql.py      # One-time Qdrant → SQL backfill
│
└── styles/js/                        # 12 ES modules (no build step)
    ├── config.js                     # API URL + key from URL param / localStorage
    ├── api.js                        # fetch wrapper — key, ngrok header, error format
    ├── main.js                       # Boot — starts GPS, wires modules
    ├── action.js                     # Register / match handlers
    ├── camera.js                     # MediaDevices, frame capture
    ├── liveness.js                   # Client-side liveness pre-flight
    ├── geo.js                        # GPS watch — live pill, tap-to-enable
    ├── status.js                     # Polls /v1/status every 30 s
    ├── result.js                     # Match result cards
    ├── geocard.js                    # Geo result card
    ├── map.js                        # Leaflet tile rendering
    └── toast.js                      # Notification banners
```

---

## Quick start

### Prerequisites

- Python 3.10+
- CUDA 12.1 (optional — CPU works but slower)
- Windows or Linux
- PostgreSQL (recommended) or SQLite for development
- Redis (optional — required only for multi-worker / multi-server deployments)

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

All packages are pinned. For CPU-only, replace the torch line:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

### 2. Generate secret keys

```bash
# Admin key (full access)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Readonly key (optional — for display screens / kiosks)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Configure

```bash
cp applicationpropertiesExample.txt application.properties
```

Minimum required settings:

```properties
api.auth.enabled        = true
api.auth.keys           = YOUR_ADMIN_KEY
api.auth.keys_readonly  = YOUR_READONLY_KEY   # optional

sql.backend             = postgres
sql.database_url        = postgresql://user:pass@host:5432/dbname

geofence.zone.office.lat    = YOUR_LAT
geofence.zone.office.lon    = YOUR_LON
geofence.zone.office.radius = 100
```

> **Never commit `application.properties`** — it contains your secret keys. It is already in `.gitignore`.

### 4. Start the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

A rotating log file `face_api.log` is created automatically.

### 5. Open the browser (first visit)

```
http://localhost:8000/?key=YOUR_ADMIN_KEY
```

Key saves to `localStorage` — every visit after just use `http://localhost:8000`.

### 6. Mobile access via ngrok

```powershell
.\start_tunnel.ps1
```

Open the printed URL with your key once per device:

```
https://abc123.ngrok-free.app/?key=YOUR_ADMIN_KEY
```

---

## Configuration

| Key                                | Default                    | What it controls                       |
| ---------------------------------- | -------------------------- | -------------------------------------- |
| `api.auth.enabled`                 | `true`                     | Enable API key auth                    |
| `api.auth.keys`                    | —                          | Admin keys (comma-separated)           |
| `api.auth.keys_readonly`           | —                          | Readonly keys (comma-separated)        |
| `liveness.enabled`                 | `true`                     | Master liveness switch                 |
| `liveness.passive.real_threshold`  | `0.55`                     | Score above this → live                |
| `liveness.passive.spoof_threshold` | `0.50`                     | Score below this → spoof               |
| `match.threshold.cvlface`          | `0.55`                     | Cosine similarity to accept a match    |
| `quality.min_face_size`            | `90`                       | Min face bbox px                       |
| `geofence.enforce`                 | `false`                    | Block requests outside zone            |
| `auto_update.enabled`              | `true`                     | Add embeddings on confident matches    |
| `geo.geocoder_backend`             | `nominatim`                | `nominatim` / `google` / `here`        |
| `sql.backend`                      | `postgres`                 | `sqlite` or `postgres`                 |
| `sql.database_url`                 | _(empty)_                  | PostgreSQL connection string           |
| `attendance.cooldown_sec`          | `1800`                     | Duplicate scan block window in seconds |
| `attendance.redis_url`             | `redis://localhost:6379/0` | Redis URL for cross-server cooldown    |

---

## Authentication and RBAC

Send the key on every request:

```
X-API-Key: your_key_here
```

**Admin keys** (`api.auth.keys`) — full access including register and delete.

**Readonly keys** (`api.auth.keys_readonly`) — match, detect, liveness, list, status only. Returns `403 INSUFFICIENT_PERMISSIONS` on register/delete.

### Opening the app per device (first visit only)

| Device                 | URL                                   |
| ---------------------- | ------------------------------------- |
| Admin tablet           | `https://ngrok-url/?key=ADMIN_KEY`    |
| Display screen / kiosk | `https://ngrok-url/?key=READONLY_KEY` |

Key saves to `localStorage` after first visit. To revoke: DevTools → Application → Local Storage → delete `API_KEY`.

---

## API reference

Interactive docs: `http://localhost:8000/docs`

| Method   | Path                     | Auth      | Status  |
| -------- | ------------------------ | --------- | ------- |
| `GET`    | `/v1/status`             | Any       | 200     |
| `POST`   | `/v1/faces/detections`   | Any       | 200     |
| `POST`   | `/v1/faces/liveness`     | Any       | 200     |
| `POST`   | `/v1/persons`            | **Admin** | 201     |
| `GET`    | `/v1/persons`            | Any       | 200     |
| `GET`    | `/v1/persons/{name}`     | Any       | 200/404 |
| `DELETE` | `/v1/persons/{name}`     | **Admin** | 204     |
| `POST`   | `/v1/faces/matches`      | Any       | 200     |
| `POST`   | `/v1/attendance/records` | Any       | 200     |
| `GET`    | `/v1/db/tier`            | Any       | 200     |

**Upload limit:** 10 MB max on all image endpoints (`413 FILE_TOO_LARGE` if exceeded).

**Attendance cooldown:** Same person not counted twice within the cooldown window. Second scan returns `"status": "already_logged"` in the detail array.

---

## Error handling

All errors return a consistent envelope:

```json
{
  "error": "NO_FACE_DETECTED",
  "message": "No face was detected. Ensure the face is clearly visible and well-lit.",
  "field": "image",
  "code": "NO_FACE_DETECTED"
}
```

| Code                       | Status | Meaning                                     |
| -------------------------- | ------ | ------------------------------------------- |
| `MISSING_API_KEY`          | 401    | No `X-API-Key` header                       |
| `INVALID_API_KEY`          | 403    | Key not recognised                          |
| `INSUFFICIENT_PERMISSIONS` | 403    | Readonly key on admin endpoint              |
| `GEOFENCE_DENIED`          | 403    | Outside allowed zone                        |
| `LIVENESS_FAILED`          | 403    | Face failed liveness                        |
| `FILE_TOO_LARGE`           | 413    | Upload > 10 MB                              |
| `RATE_LIMITED`             | 429    | Too many requests                           |
| `INVALID_IMAGE`            | 400    | Cannot decode uploaded file                 |
| `NO_FACE_DETECTED`         | 422    | No face in image                            |
| `FACE_QUALITY_FAILED`      | 422    | Face too small/blurry/angled                |
| `PERSON_ALREADY_EXISTS`    | 409    | Name already registered                     |
| `PERSON_NOT_FOUND`         | 404    | Name not in database                        |
| `VALIDATION_ERROR`         | 422    | Wrong field type / missing field            |
| `INTERNAL_ERROR`           | 500    | Unexpected error (logged to `face_api.log`) |

---

## Rate limiting

| Endpoint                                                                                      | Limit        |
| --------------------------------------------------------------------------------------------- | ------------ |
| `POST /v1/persons`, `DELETE /v1/persons/{name}`                                               | 10 / minute  |
| `POST /v1/faces/liveness`                                                                     | 20 / minute  |
| `POST /v1/faces/matches`, `POST /v1/attendance/records`, `GET /v1/persons`, `GET /v1/db/tier` | 30 / minute  |
| `GET /v1/status`, `POST /v1/faces/detections`, `GET /v1/persons/{name}`                       | 60 / minute  |
| Global default                                                                                | 200 / minute |

---

## Liveness detection

Four models run **in parallel** via `ThreadPoolExecutor`:

| ID         | Type                          | Focus                              |
| ---------- | ----------------------------- | ---------------------------------- |
| `ICM2O`    | PyTorch                       | Cross-modal texture analysis       |
| `IOM2C`    | PyTorch                       | Illumination-aware spoof detection |
| `modelrgb` | ONNX                          | RGB-channel spoof detection        |
| `SASF`     | PyTorch (MiniFASNetV2 + V1SE) | Silent Face Anti-Spoofing          |

Weights auto-download from `mothieram/face-anti-spoofing` on HuggingFace. If a model fails to load, its weight redistributes — system degrades gracefully.

A 5-frame micro-motion guard flags still faces (printed photo attacks) before models run.

---

## Geofencing and geotagging

Coordinate resolution order: Browser GPS → EXIF GPS → no coordinates.

The geofence check (local haversine / Shapely) and the face detection pipeline run **in parallel** via `asyncio.gather` — the Nominatim reverse geocode does not add to response latency. Full address information is always present in every response.

```properties
# Circle zone
geofence.zone.office.type   = circle
geofence.zone.office.lat    = 11.019751
geofence.zone.office.lon    = 76.978664
geofence.zone.office.radius = 100

# Polygon zone
geofence.zone.campus.type   = polygon
geofence.zone.campus.points = 11.020,76.978 | 11.021,76.979 | 11.021,76.977
```

Set `geofence.enforce = true` to return `403 GEOFENCE_DENIED` for out-of-zone requests.

---

## Database and scaling tiers

| Tier | People    | Strategy | HNSW m | search ef |
| ---- | --------- | -------- | ------ | --------- |
| 0    | 0–99      | single   | 8      | 64        |
| 1    | 100–999   | multi    | 16     | 128       |
| 2    | 1000–9999 | hybrid   | 32     | 256       |
| 3    | 10000+    | mean     | 64     | 512       |

---

## Person registry (SQL)

```sql
CREATE TABLE persons (
    person_id    UUID PRIMARY KEY,
    name_key     TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
```

PostgreSQL is the recommended backend for all deployments. SQLite is supported for local development only.

---

## Attendance cooldown store

The duplicate prevention cooldown uses a two-tier store:

| Tier            | When used         | Behaviour                                                       |
| --------------- | ----------------- | --------------------------------------------------------------- |
| Redis           | Redis connected   | Atomic `SET NX EX` — enforced across all workers and servers ✅ |
| In-process dict | Redis unavailable | Single-worker fallback — still works, not cross-server safe     |

Redis is **optional for single-server setups**. It is required when running multiple workers (`gunicorn -w 2+`) or multiple servers across offices / regions.

### How the cooldown works

```
First scan  →  Redis key created with TTL = attendance.cooldown_sec
               e.g. "attendance:cooldown:ravi" expires in 1800s
Re-scan within window  →  key exists  → status: already_logged
Re-scan after window   →  key expired → status: matched (new entry)
```

### Enabling Redis

```bash
# Install Redis Python package
pip install redis>=5.0.0

# Windows — install Memurai (native Redis for Windows)
# https://www.memurai.com/get-memurai
# Choose "Memurai for Redis" during install
```

Add to `application.properties`:

```properties
attendance.cooldown_sec = 1800
attendance.redis_url    = redis://localhost:6379/0

# Multi-region (replace with your hosted Redis URL):
# attendance.redis_url = redis://default:password@your-host.redis.cloud:6379/0
```

Add to `config.py`:

```python
ATTENDANCE_COOLDOWN_SEC = _int("attendance.cooldown_sec", 1800)
ATTENDANCE_REDIS_URL    = _str("attendance.redis_url", "redis://localhost:6379/0")
```

On startup the server logs:

```
[STARTUP] [OK] Redis connected — attendance cooldown is cross-server safe
```

Or if Redis is not running:

```
[STARTUP] [WARN] Redis unavailable — using in-process dict (single-worker mode)
```

### Checking Redis data

```bash
# Check active cooldowns (Windows)
"C:\Program Files\Memurai\memurai-cli.exe" keys attendance:cooldown:*

# Check remaining cooldown for a person (in seconds)
"C:\Program Files\Memurai\memurai-cli.exe" ttl attendance:cooldown:ravi
```

---

## Data migration

```bash
cd datamigration
python migrate_qdrant_to_sql.py
```

Backfills the SQL `persons` table from existing Qdrant records. Idempotent — safe to run multiple times.

---

## Mobile / ngrok access

- **CORS** — regex covers `*.ngrok-free.app`, `*.ngrok.io`, `*.loca.lt`, `*.trycloudflare.com`
- **API URL** — `config.js` uses `window.location.origin` — auto-detects ngrok URL
- **Interstitial bypass** — `api.js` header for fetch calls + `NgrokSkipMiddleware` for page load
- **GPS** — tap the pill to trigger OS permission dialog; device-specific instructions shown when denied

```powershell
.\start_tunnel.ps1                     # basic
.\start_tunnel.ps1 -AutoReload $true  # development
.\start_tunnel.ps1 -Port 8080         # custom port
```

---

## Frontend

| Module                                             | Responsibility                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------- |
| `config.js`                                        | API URL, key from `?key=` URL param or `localStorage`                 |
| `api.js`                                           | fetch wrapper — key header, ngrok header, structured error formatting |
| `main.js`                                          | Boot — starts GPS on page load                                        |
| `action.js`                                        | Register / match button handlers                                      |
| `camera.js`                                        | Camera capture, live detection overlay                                |
| `liveness.js`                                      | Client-side passive liveness pre-flight                               |
| `geo.js`                                           | GPS watch, live pill, tap-to-enable, device instructions              |
| `status.js`                                        | Server health polling every 30 s                                      |
| `result.js` / `geocard.js` / `map.js` / `toast.js` | UI rendering                                                          |

**Key is never hardcoded.** Read from `?key=` URL param → saved to `localStorage` → used on all subsequent visits.

---

## Production deployment

### Single server (current setup)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Multi-worker (when traffic increases beyond ~50 simultaneous scans)

Install and configure Redis first — see [Attendance cooldown store](#attendance-cooldown-store).

```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Multi-region (multiple offices / servers)

1. Provision a hosted Redis — Redis Cloud (free tier), Railway, or AWS ElastiCache.
2. Set `attendance.redis_url` on **all servers** to the same hosted Redis URL.
3. All servers share one cooldown register — a person cannot double-scan across locations.

**Logging** — `face_api.log` created automatically. Rotating: 10 MB × 5 backups.

**HTTPS** — use nginx reverse proxy with SSL for direct LAN access without ngrok.

---

## Running the CLI (`main.py`)

```bash
python main.py
```

`SPACE` capture · `Q` quit · `r` register · `m` match · `q` cancel.

---

## Troubleshooting

**`INVALID_API_KEY` in browser**

1. `python -c "from config import API_KEYS; print(repr(API_KEYS))"` — what did server load?
2. DevTools Console → `localStorage.getItem("API_KEY")` — what does browser have?
3. If mismatch: `localStorage.clear()` then reopen with `?key=YOUR_KEY`

**Liveness always fails** — lower `liveness.passive.spoof_threshold` to 0.30 temporarily; check `model_scores` in response.

**Geofence always blocks** — set `geofence.enforce = false` during dev; verify lat/lon order (lat first).

**Geo card shows "Unknown"** — ensure `lat` and `lon` are being sent from the browser. Check GPS permissions. Geocode runs in parallel with the pipeline and address is always in the response.

**Attendance counts person twice** — on single worker this should not happen. On multi-worker, ensure Redis is configured and `[STARTUP] [OK] Redis connected` appears in server logs.

**Redis not connecting** — check `attendance.redis_url` in `application.properties`. Server falls back to in-process dict automatically. On Windows, ensure Memurai service is running: `net start memurai`.

**CVLFace download stalls** — download manually to `~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m/`.

---

## Model sources

| Model                                 | Source                                                 |
| ------------------------------------- | ------------------------------------------------------ |
| RetinaFace-PyTorch                    | `retinaface-pytorch` pip package                       |
| CVLFace AdaFace IR-101                | HuggingFace `minchul/cvlface_adaface_ir101_webface12m` |
| ICM2O / IOM2C / modelrgb / MiniFASNet | HuggingFace `mothieram/face-anti-spoofing`             |
