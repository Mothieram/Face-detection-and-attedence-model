# Face Recognition Attendance System

A production-grade attendance pipeline built on **RetinaFace** (detection) + **AdaFace IR-101** (embedding) + **Qdrant** (vector search), with passive liveness detection, GPS geofencing, SQL person registry, rate limiting, and a FastAPI REST backend.

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture overview](#architecture-overview)
- [Pipeline — step by step](#pipeline--step-by-step)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Rate limiting](#rate-limiting)
- [Liveness detection](#liveness-detection)
- [Geofencing and geotagging](#geofencing-and-geotagging)
- [Database and scaling tiers](#database-and-scaling-tiers)
- [Person registry (SQL)](#person-registry-sql)
- [Data migration](#data-migration)
- [Mobile / ngrok access](#mobile--ngrok-access)
- [Frontend](#frontend)
- [Running the CLI (main.py)](#running-the-cli-mainpy)
- [Troubleshooting](#troubleshooting)

---

## What it does

1. **Registers** a person — creates a SQL identity record and stores a 512-d face embedding in Qdrant.
2. **Matches** a face from a live image against the database using HNSW approximate nearest-neighbour search.
3. **Guards** against spoofing with a 4-model passive liveness ensemble (ICM2O, IOM2C, modelrgb ONNX, SASF/MiniFASNet).
4. **Enforces** location with GPS geofencing — circle and polygon zones.
5. **Logs** attendance with reverse-geocoded location, local timezone, and per-face liveness scores.
6. **Auto-scales** the Qdrant index across four tiers (1–10K+ people) without manual re-indexing.
7. **Rate-limits** every endpoint to protect ML inference from abuse.
8. **Works on mobile** over ngrok tunnels with CORS and interstitial bypass built in.

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
       ├── /v1/status               → system health check
       ├── /v1/faces/detections     → detector.py  (RetinaFace)
       ├── /v1/faces/liveness       → liveness.py  (4-model ensemble)
       ├── /v1/persons    POST      → persons_db.py (SQL) + full pipeline
       ├── /v1/faces/matches        → full pipeline
       ├── /v1/attendance/records   → full pipeline
       ├── /v1/persons    GET       → persons_db.py (SQL)
       └── /v1/db/tier              → database.py  (Qdrant tier info)

Full pipeline (register / match / attendance):
  preprocessor.py  →  detector.py  →  aligner.py
       →  liveness.py  →  embedder.py  →  matcher.py
       →  database.py  +  geotagging.py  (parallel)
```

---

## Pipeline — step by step

Every image through `/v1/persons`, `/v1/faces/matches`, or `/v1/attendance/records` runs the same pipeline:

### 1. Preprocess (`preprocessor.py`)

Raw BGR image is cleaned up before detection:

- **Upscale** if smallest dimension < 400 px
- **CLAHE** contrast enhancement in LAB space (low-light robustness)
- **Auto gamma** correction (LUTs precomputed at import — zero per-frame allocation)
- **Bilateral denoise** + **unsharp masking** — skipped in `fast=True` mode used by live preview

### 2. Detect (`detector.py`)

**RetinaFace-PyTorch** runs multi-threshold passes (`[0.80, 0.60, 0.45]`) so faces that miss the primary threshold still get a second chance. Returns per-face:

```python
{
  "face_id":   "face_0",
  "score":     0.97,              # RetinaFace confidence
  "bbox":      [x1, y1, x2, y2],
  "landmarks": {
      "right_eye": [x, y], "left_eye": [x, y], "nose": [x, y],
      "mouth_right": [x, y], "mouth_left": [x, y],
  }
}
```

### 3. Quality gate (`aligner.py`)

Each face is checked before anything expensive runs:

- **Minimum size** — bounding box short side ≥ `MIN_FACE_SIZE` px (default 90)
- **Blur** — Laplacian variance ≥ `MIN_BLUR_SCORE` (default 20.0)
- **Yaw** — estimated head turn ≤ `MAX_YAW_DEGREES` (default 60°)

Faces that fail become `"Rejected"` in match results — no liveness or embedding runs on them.

### 4. Align (`aligner.py`)

5-point affine transform (RANSAC) warps the face to a canonical 112×112 template matching the AdaFace training distribution. Falls back to eye-axis crop if RANSAC fails.

### 5. Liveness (`liveness.py`)

See [Liveness detection](#liveness-detection).

### 6. Embed (`embedder.py`)

**CVLFace AdaFace IR-101** generates a 512-d L2-normalised embedding (downloaded from HuggingFace on first run, cached at `~/.cvlface_cache/`). Falls back to a handcrafted descriptor if CVLFace dependencies are missing.

### 7. Match (`matcher.py`)

Qdrant HNSW search filtered by `embedding_mode` to prevent cross-mode false positives:

- `cvlface` mode → threshold 0.55
- `fallback` mode → threshold 0.94

Returns `"No Match"` or `"No DB records"` on failure.

### 8. Geotag (`geotagging.py`)

GPS → geofence check → reverse geocode → timezone resolution. See [Geofencing and geotagging](#geofencing-and-geotagging).

---

## Project structure

```
.
├── api.py                      # FastAPI server — all REST endpoints
├── main.py                     # CLI entry point — live camera mode
├── config.py                   # Loads all settings from application.properties
├── applicationpropertiesExample.txt  # Copy → application.properties to configure
├── index.html                  # Browser frontend entry point
├── requirement.txt             # pip dependencies
├── start_tunnel.ps1            # ngrok + uvicorn launcher (Windows)
│
├── modules/
│   ├── preprocessor.py         # Image cleanup before detection
│   ├── detector.py             # RetinaFace wrapper
│   ├── aligner.py              # 5-pt affine align + quality gate
│   ├── liveness.py             # 4-model spoof detection ensemble
│   ├── SASF.py                 # MiniFASNet wrapper (used by liveness.py)
│   ├── embedder.py             # CVLFace AdaFace IR-101 embedding
│   ├── matcher.py              # Qdrant HNSW search
│   ├── database.py             # Qdrant client, tier management, auto-update
│   ├── persons_db.py           # SQL person registry (SQLite / PostgreSQL)
│   ├── geotagging.py           # GPS → geofence → reverse geocode
│   └── helper/                 # SASF model internals (MiniFASNet architecture)
│       ├── anti_spoof_predict.py
│       ├── generate_patches.py
│       ├── utility.py
│       ├── default_config.py
│       └── data_io/
│           ├── functional.py
│           └── transform.py
│
├── datamigration/
│   └── migrate_qdrant_to_sql.py  # One-time migration: backfill SQL from Qdrant
│
├── utils/
│   ├── visualizer.py           # OpenCV drawing helpers (CLI only)
│   └── file_picker.py          # File dialog helper (CLI only)
│
├── styles/
│   ├── styles.css
│   └── js/                     # ES modules (no build step)
│       ├── main.js             # Boot — imports all modules, starts GPS on load
│       ├── config.js           # API base URL (auto-detects ngrok/localhost)
│       ├── api.js              # fetch wrapper — injects API key + ngrok header
│       ├── camera.js           # MediaDevices, frame capture
│       ├── liveness.js         # Client-side passive liveness pre-flight
│       ├── geo.js              # GPS watch — live pill update, tap-to-enable
│       ├── status.js           # Polls /v1/status every 30 s
│       ├── toast.js            # Notification banners
│       ├── map.js              # Leaflet map tile rendering
│       ├── geocard.js          # Geo result card (address, timezone, zone)
│       ├── result.js           # Per-face match result cards
│       └── action.js           # Register / match button handlers
│
└── weights/                    # Liveness model weights (auto-downloaded)
    ├── ICM2O.pth.tar
    ├── IOM2C.pth.tar
    ├── modelrgb.onnx
    ├── 2.7_80x80_MiniFASNetV2.pth
    └── 4_0_0_80x80_MiniFASNetV1SE.pth
```

---

## Quick start

### Prerequisites

- Python 3.10+
- CUDA 12.1 (optional — CPU works but inference is slower)
- Windows or Linux (macOS untested with `CAP_DSHOW`)

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

For CPU-only (no CUDA), replace the torch line in `requirement.txt`:

```bash
pip install torch torchvision torchaudio
```

### 2. Configure

Copy the example properties file and edit it:

```bash
cp applicationpropertiesExample.txt application.properties
```

Key values to set before first run:

```properties
# API authentication
api.auth.keys = your_secret_key_here

# SQL person registry backend (sqlite is the default — no setup needed)
sql.backend = sqlite
sql.sqlite_path = ./persons.sqlite3

# Geofence zone (update lat/lon to your location)
geofence.zone.office.lat = 12.9716
geofence.zone.office.lon = 77.5946
geofence.zone.office.radius = 100
```

For secrets in production use environment variables — they override the properties file:

```bash
export API_KEYS="your_secret_key_here"
export API_AUTH_ENABLED=true
export GEOCODER_API_KEY="your_google_or_here_key"   # only if not using nominatim
```

### 3. Start the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

On first run, CVLFace (~250 MB) and liveness weights (~30 MB) download automatically from HuggingFace. Startup output:

```
[STARTUP] [OK] RetinaFace ready
[STARTUP] [OK] CVLFace ready
[STARTUP] [OK] Liveness ready - ['icm2o', 'iom2c', 'modelrgb', 'sasf']
[STARTUP] All models loaded — server ready.
```

### 4. Open the browser UI

Go to `http://localhost:8000` — the API serves `index.html` directly.

### 5. For mobile access — use the tunnel script

```powershell
.\start_tunnel.ps1
```

This starts uvicorn and ngrok together, prints the public URL, and keeps both alive. Open the printed `https://xxx.ngrok-free.app` URL on any mobile browser — no extra config needed.

### 6. (Optional) CLI mode

```bash
python main.py
```

Live camera window. `SPACE` to capture, then `r` to register or `m` to match.

---

## Configuration

All settings are loaded from `application.properties` at startup. The file is never required — every key has a hardcoded default. Environment variables always win over the properties file.

### Most commonly changed

| Key                                | Default             | What it controls                             |
| ---------------------------------- | ------------------- | -------------------------------------------- |
| `liveness.enabled`                 | `true`              | Master liveness switch                       |
| `liveness.passive.real_threshold`  | `0.55`              | Score above this → live                      |
| `liveness.passive.spoof_threshold` | `0.50`              | Score below this → spoof                     |
| `match.threshold.cvlface`          | `0.55`              | Cosine similarity to accept a match          |
| `quality.min_face_size`            | `90`                | Min face bounding box px                     |
| `geofence.enforce`                 | `false`             | Block requests from outside zone             |
| `auto_update.enabled`              | `true`              | Add embeddings on repeated confident matches |
| `geo.geocoder_backend`             | `nominatim`         | `nominatim` / `google` / `here`              |
| `sql.backend`                      | `sqlite`            | `sqlite` or `postgres`                       |
| `sql.sqlite_path`                  | `./persons.sqlite3` | SQLite file location                         |
| `sql.database_url`                 | _(empty)_           | PostgreSQL connection string                 |

### Liveness threshold zones

```
real_score ≥ 0.55              →  PASS  (clearly live)
real_score < 0.50              →  FAIL  (clearly spoof)
0.50 ≤ real_score < 0.55  →  uncertain
    → active head-pose challenge if LIVENESS_ACTIVE_ENABLED=True
    → otherwise FAIL with reason "Passive uncertain (active disabled)"
```

---

## API reference

All image endpoints accept `multipart/form-data`. Authenticate with `X-API-Key: <your_key>` header.

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method   | Path                     | Description                             | Status code        |
| -------- | ------------------------ | --------------------------------------- | ------------------ |
| `GET`    | `/v1/status`             | Health check + model status + DB tier   | 200                |
| `POST`   | `/v1/faces/detections`   | Detect all faces + quality check        | 200                |
| `POST`   | `/v1/faces/liveness`     | Liveness on largest detected face       | 200                |
| `POST`   | `/v1/persons`            | Register a new person                   | **201 Created**    |
| `GET`    | `/v1/persons`            | List registered persons (paginated)     | 200                |
| `GET`    | `/v1/persons/{name}`     | Check if person exists                  | 200 / 404          |
| `DELETE` | `/v1/persons/{name}`     | Delete person (SQL + Qdrant)            | **204 No Content** |
| `POST`   | `/v1/faces/matches`      | Match all faces in image                | 200                |
| `POST`   | `/v1/attendance/records` | Attendance summary payload              | 200                |
| `GET`    | `/v1/db/tier`            | Active Qdrant scaling tier              | 200                |
| `POST`   | `/v1/detect`             | Legacy alias for `/v1/faces/detections` | 200                |

### Error response shape

All errors return a consistent envelope:

```json
{
  "error":   "LIVENESS_FAILED",
  "message": "Liveness failed: Passive ensemble: spoof detected",
  "detail":  { ... }
}
```

### Example: register response

```json
{
  "success": true,
  "name": "Alice",
  "face_id": "face_0",
  "liveness": {
    "passed": true,
    "method": "passive",
    "score": 0.82,
    "reason": "Passive ensemble: real",
    "model_scores": {
      "ICM2O": 0.12,
      "IOM2C": 0.09,
      "modelrgb": 0.15,
      "SASF": 0.11
    }
  },
  "geo": {
    "coordinates": { "lat": 11.019, "lon": 76.978, "accuracy_m": 12.5 },
    "city": "Coimbatore",
    "country": "India",
    "geofence_passed": true,
    "geofence_zone": "office"
  },
  "embedding_mode": "cvlface",
  "message": "'Alice' registered successfully."
}
```

### List persons — pagination

```
GET /v1/persons?limit=50&offset=0&include_embeddings=false
```

Response includes `total`, `limit`, `offset`, `has_more` for cursor-style iteration.

---

## Rate limiting

Every endpoint is rate-limited per IP using [slowapi](https://github.com/laurentS/slowapi). Exceeding the limit returns HTTP 429.

| Endpoint                      | Limit        |
| ----------------------------- | ------------ |
| `POST /v1/persons` (register) | 10 / minute  |
| `DELETE /v1/persons/{name}`   | 10 / minute  |
| `POST /v1/faces/matches`      | 30 / minute  |
| `POST /v1/attendance/records` | 30 / minute  |
| `GET /v1/persons`             | 30 / minute  |
| `GET /v1/db/tier`             | 30 / minute  |
| `POST /v1/faces/liveness`     | 20 / minute  |
| `GET /v1/status`              | 60 / minute  |
| `POST /v1/faces/detections`   | 60 / minute  |
| `GET /v1/persons/{name}`      | 60 / minute  |
| Global default                | 200 / minute |

---

## Liveness detection

The passive liveness system runs four models **in parallel** via `ThreadPoolExecutor`.

### Models

| ID         | Type                          | Weights          | Focus                              |
| ---------- | ----------------------------- | ---------------- | ---------------------------------- |
| `ICM2O`    | PyTorch `.pth.tar`            | `ICM2O.pth.tar`  | Cross-modal texture analysis       |
| `IOM2C`    | PyTorch `.pth.tar`            | `IOM2C.pth.tar`  | Illumination-aware spoof detection |
| `modelrgb` | ONNX                          | `modelrgb.onnx`  | RGB-channel spoof detection        |
| `SASF`     | PyTorch (MiniFASNetV2 + V1SE) | two `.pth` files | Silent Face Anti-Spoofing          |

All weights auto-download from `mothieram/face-anti-spoofing` on HuggingFace on first use.

### Ensemble fusion

```python
fused_spoof = weighted_sum(spoof_probs) / total_active_weight
real_score  = 1.0 - fused_spoof
```

If a model fails to load its weight is redistributed to the remaining models — the system degrades gracefully rather than crashing.

### Micro-motion guard

A 5-frame rolling buffer checks for pixel-level movement between frames. Still faces (printed photo attacks) are flagged before the models even run. Threshold: mean absolute diff > 2.5.

### Active challenge

If `real_score` falls in the uncertain zone and `LIVENESS_ACTIVE_ENABLED=True`, a head-pose challenge is triggered: the user must turn their head left or right within `liveness.active.headpose_timeout_sec` seconds (default 8).

---

## Geofencing and geotagging

`modules/geotagging.py` runs on every write endpoint. Coordinate resolution order:

1. **Browser GPS** — `lat` / `lon` form fields
2. **EXIF GPS** — extracted from the uploaded JPEG
3. **No coordinates** — geofence skipped; `geofence_passed=True` unless `geofence.enforce=true`

### Zone format (in `application.properties`)

```properties
# Circle zone
geofence.zone.office.type   = circle
geofence.zone.office.name   = Main Office
geofence.zone.office.lat    = 11.019751
geofence.zone.office.lon    = 76.978664
geofence.zone.office.radius = 100

# Polygon zone
geofence.zone.campus.type   = polygon
geofence.zone.campus.name   = Campus
geofence.zone.campus.points = 11.020,76.978 | 11.021,76.979 | 11.021,76.977
```

Set `geofence.enforce = true` to return HTTP 403 for requests outside all zones.

### Reverse geocoding backends

| Backend     | Key needed               | Notes                                                   |
| ----------- | ------------------------ | ------------------------------------------------------- |
| `nominatim` | No                       | Free, OSM-backed, 1 req/s — results cached aggressively |
| `google`    | Yes (`GEOCODER_API_KEY`) | Higher quota, better accuracy                           |
| `here`      | Yes (`GEOCODER_API_KEY`) | Enterprise-grade                                        |

Cache: coordinate grid 0.0001° ≈ 11 m, TTL 1 hour (configurable).

---

## Database and scaling tiers

`modules/database.py` manages a local Qdrant instance at `./qdrant_storage`. HNSW parameters auto-scale by registered person count:

| Tier | People    | Strategy | Templates/person     | HNSW m | search ef |
| ---- | --------- | -------- | -------------------- | ------ | --------- |
| 0    | 0–99      | single   | 1                    | 8      | 64        |
| 1    | 100–999   | multi    | 5                    | 16     | 128       |
| 2    | 1000–9999 | hybrid   | 3 (then consolidate) | 32     | 256       |
| 3    | 10000+    | mean     | rolling average      | 64     | 512       |

### Auto-update

When `auto_update.enabled=true`, a match in the medium-confidence band adds the new embedding as an additional template — keeping the system adapting to gradual appearance changes without re-registration. At the `hybrid` tier, templates consolidate into a deterministic `mean` + `best` pair (UUIDv5 IDs prevent duplicates).

---

## Person registry (SQL)

`modules/persons_db.py` maintains a lightweight SQL table alongside Qdrant, providing stable `person_id` UUIDs that decouple identity from name strings.

### Schema

```sql
CREATE TABLE persons (
    person_id   UUID PRIMARY KEY,
    name_key    TEXT UNIQUE NOT NULL,   -- lowercased name for lookups
    display_name TEXT NOT NULL,          -- original casing
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

### Backends

**SQLite (default)** — zero setup, works immediately:

```properties
sql.backend     = sqlite
sql.sqlite_path = ./persons.sqlite3
```

**PostgreSQL** — for multi-process / production deployments:

```properties
sql.backend      = postgres
sql.database_url = postgresql://user:pass@host:5432/dbname
```

Or via environment variable: `export SQL_DATABASE_URL=postgresql://...`

The SQLite connection is a module-level singleton (WAL mode, `check_same_thread=False`) — no connection overhead on hot paths.

---

## Data migration

If you had faces registered before the SQL layer was added, run the one-time migration script to backfill the `persons` table from existing Qdrant records:

```bash
cd datamigration
python migrate_qdrant_to_sql.py
```

The script scans every Qdrant point, creates a SQL row for each unique person, and writes the `person_id` UUID back into the Qdrant payload. Idempotent — safe to run multiple times.

Output:

```
Migration complete.
Qdrant points scanned : 42
SQL rows ensured      : 8
Qdrant points updated : 38
Skipped points        : 0
```

---

## Mobile / ngrok access

The system is built to work over ngrok tunnels with no extra config:

**CORS** — `allow_origin_regex` covers `*.ngrok-free.app`, `*.ngrok.io`, `*.loca.lt`, and `*.trycloudflare.com` out of the box.

**API base URL** — `styles/js/config.js` uses `window.location.origin`, so when the mobile browser opens `https://abc.ngrok-free.app` the JS automatically targets that same origin for all API calls.

**ngrok interstitial** — `api.js` sends `ngrok-skip-browser-warning: true` on every fetch call, bypassing the warning page that would otherwise return HTML instead of JSON.

**GPS pill** — the GPS status pill in the header is tappable when location is off. Tapping it calls `getCurrentPosition` synchronously inside the tap gesture — the only browser-approved way to trigger the OS location permission dialog on Android and iOS.

### Using the tunnel script

```powershell
# Basic
.\start_tunnel.ps1

# With auto-reload for development
.\start_tunnel.ps1 -AutoReload $true

# Custom port
.\start_tunnel.ps1 -Port 8080
```

---

## Frontend

The browser UI is 12 plain ES modules — no bundler, no framework, no build step.

| Module        | Responsibility                                                               |
| ------------- | ---------------------------------------------------------------------------- |
| `config.js`   | API base URL (auto-detects origin), auth key, tuning constants               |
| `api.js`      | `fetch` wrapper — injects `X-API-Key` and `ngrok-skip-browser-warning`       |
| `main.js`     | Boot — starts GPS on page load, wires all modules                            |
| `camera.js`   | `MediaDevices`, frame capture, live face detection overlay                   |
| `liveness.js` | Client-side passive liveness pre-flight before capture                       |
| `geo.js`      | GPS `watchPosition` — live pill, tap-to-enable, device-specific instructions |
| `status.js`   | Polls `/v1/status` every 30 s                                                |
| `toast.js`    | Notification banners                                                         |
| `map.js`      | Leaflet tile rendering for geo cards                                         |
| `geocard.js`  | Geo result card (address, timezone, zone name)                               |
| `result.js`   | Per-face match result cards                                                  |
| `action.js`   | Register / match button handlers                                             |

---

## Running the CLI (`main.py`)

Camera-based mode with no server required — calls module functions directly.

```bash
python main.py
```

Controls:

- `SPACE` — capture frame
- `Q` — quit

After capture: `r` register, `m` match, `q` cancel.

The CLI uses the same pipeline as the API but skips geofencing and rate limiting.

---

## Troubleshooting

**Server starts but models show [WARN] not [OK]**

The API now starts in degraded mode when optional models fail. Check `[STARTUP]` logs — detection falls back to Haar if RetinaFace is unavailable, embedding falls back to a handcrafted descriptor if CVLFace dependencies are missing.

**"No DB records" even after registering**

The `embedding_mode` stored doesn't match the query mode. Happens if CVLFace loaded during registration but not during matching (or vice versa). Check `cvlface_loaded` in `/v1/status` — it must be consistent between calls.

**Liveness always fails**

- Call `POST /v1/faces/liveness` directly with a test image to isolate the issue.
- Lower `liveness.passive.spoof_threshold` to 0.30 temporarily to see raw scores.
- Check `model_scores` in the response to identify which model is driving the result.
- Ensure good lighting and that the face is not partially occluded.

**GPS pill not showing permission dialog on mobile**

The tap handler uses `getCurrentPosition` called synchronously inside the tap gesture — this is the only path that triggers the OS permission dialog. If location was hard-denied, the banner shows device-specific step-by-step instructions (differs between Android Chrome, Samsung Internet, and iOS Safari).

**Mobile browser sees ngrok interstitial instead of the app**

Upgrade to the latest `api.js` — it includes the `ngrok-skip-browser-warning: true` header on every fetch. For the initial page load, add `NgrokSkipMiddleware` to `api.py` (see middleware section).

**Geofence always blocks (HTTP 403)**

Set `geofence.enforce = false` in `application.properties` during development. Check that zone `lat`/`lon` are decimal degrees in the correct order (latitude first, not longitude).

**Camera not found (CLI on Windows)**

The code uses `cv2.CAP_DSHOW` on Windows. Try `camera_index` values 0, 1, 2 if you have multiple cameras.

**SQLite `persons.sqlite3` errors on startup**

The file is created automatically. If you see corruption errors, delete `persons.sqlite3` and run `python datamigration/migrate_qdrant_to_sql.py` to rebuild it from existing Qdrant data.

**CVLFace download stalls**

Download manually into `~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m/`. The loader checks file presence before attempting a download.

---

## Model sources

| Model                  | Source                                                 |
| ---------------------- | ------------------------------------------------------ |
| RetinaFace-PyTorch     | `retinaface-pytorch` pip package                       |
| CVLFace AdaFace IR-101 | HuggingFace `minchul/cvlface_adaface_ir101_webface12m` |
| ICM2O / IOM2C          | HuggingFace `mothieram/face-anti-spoofing`             |
| modelrgb ONNX          | HuggingFace `mothieram/face-anti-spoofing`             |
| MiniFASNetV2 / V1SE    | HuggingFace `mothieram/face-anti-spoofing`             |

All liveness weights auto-download on first use. For offline environments, place them manually in the `weights/` directory.
