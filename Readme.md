# Face Recognition Attendance System

A production-grade attendance pipeline built on **RetinaFace** (detection) + **AdaFace IR-101** (embedding) + **Qdrant** (vector search), with passive liveness detection, GPS geofencing, and a FastAPI REST backend.

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture overview](#architecture-overview)
- [Pipeline — step by step](#pipeline--step-by-step)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Liveness detection](#liveness-detection)
- [Geofencing and geotagging](#geofencing-and-geotagging)
- [Database and scaling tiers](#database-and-scaling-tiers)
- [Frontend](#frontend)
- [Running the CLI (main.py)](#running-the-cli-mainpy)
- [Troubleshooting](#troubleshooting)

---

## What it does

1. **Registers** a person by storing a 512-d face embedding in a Qdrant vector database.
2. **Matches** a face from a live image against the database using HNSW approximate nearest-neighbour search.
3. **Guards** against spoofing with a 4-model passive liveness ensemble (ICM2O, IOM2C, modelrgb ONNX, SASF/MiniFASNet).
4. **Enforces** location with GPS geofencing — circle and polygon zones.
5. **Logs** attendance with reverse-geocoded location, local timezone, and per-face liveness scores.
6. **Auto-scales** the Qdrant index across four tiers (1–10K+ people) without manual re-indexing.

---

## Architecture overview

```
Browser / Camera
       │
       ▼
   index.html  ──────────────────────────────────────────────────
   (JS modules)                                                  │
       │  HTTP multipart/form-data                               │
       ▼                                                         │
   api.py  (FastAPI)                                             │
       │                                                         │
       ├── /detect   ──► detector.py  ──► RetinaFace             │
       ├── /register ──► full pipeline (see below)               │
       ├── /match    ──► full pipeline                           │
       ├── /attendance ► full pipeline                           │
       └── /liveness ──► liveness.py (ensemble)                  │
                                                                  │
Full pipeline:                                                    │
  preprocessor.py                                                 │
       │                                                          │
  detector.py  (RetinaFace-PyTorch)                               │
       │                                                          │
  aligner.py   (5-pt affine → 112×112)                            │
       │                                                          │
  liveness.py  (4-model ensemble, ThreadPoolExecutor)             │
       │                                                          │
  embedder.py  (CVLFace AdaFace IR-101 → 512-d vector)            │
       │                                                          │
  matcher.py   (Qdrant HNSW ANN)                                  │
       │                                                          │
  database.py  (Qdrant local, dynamic tier management)            │
       │                                                          │
  geotagging.py (GPS → geofence check → reverse geocode)  ◄───────
```

---

## Pipeline — step by step

Every image that comes through `/register`, `/match`, or `/attendance` runs the same pipeline:

### 1. Preprocess (`preprocessor.py`)

Raw BGR image is cleaned up before detection. Steps (all configurable in `config.py`):

- **Upscale** if smallest dimension < 400 px (prevents RetinaFace from missing small faces)
- **CLAHE** contrast enhancement in LAB space — helps in low-light conditions
- **Auto gamma** correction based on mean brightness (LUTs are precomputed at import time for speed)
- **Bilateral denoise** and **unsharp masking** — skipped in `fast=True` mode (used during live camera preview)

### 2. Detect (`detector.py`)

**RetinaFace-PyTorch** runs multi-threshold passes (`DETECTION_CONFIDENCE_PASSES = [0.80, 0.60, 0.45]`) so faces that miss the primary threshold still get a second chance. Each face is returned as:

```python
{
  "face_id":   "face_0",          # deterministic index string
  "score":     0.97,              # RetinaFace confidence
  "bbox":      [x1, y1, x2, y2], # pixel coords in the original image
  "landmarks": {                  # 5-point dict
      "right_eye":   [x, y],
      "left_eye":    [x, y],
      "nose":        [x, y],
      "mouth_right": [x, y],
      "mouth_left":  [x, y],
  }
}
```

### 3. Quality gate (`aligner.py → check_face_quality`)

Before anything expensive runs, each face is checked for:

- **Minimum size** — bounding box short side ≥ 90 px (set `MIN_FACE_SIZE` in config)
- **Blur** — Laplacian variance ≥ 20.0 (set `MIN_BLUR_SCORE`)
- **Yaw** — estimated head turn ≤ 60° (set `MAX_YAW_DEGREES`)

Faces that fail are returned as `"Rejected"` in match results; they don't run liveness or embedding.

### 4. Align (`aligner.py`)

5-point affine transform (RANSAC) warps the face to a canonical 112×112 template matching the AdaFace training distribution. Falls back to simple eye-axis crop if RANSAC fails.

### 5. Liveness (`liveness.py`)

See [Liveness detection](#liveness-detection) below.

### 6. Embed (`embedder.py`)

**CVLFace AdaFace IR-101** (downloaded from HuggingFace on first run, cached at `~/.cvlface_cache/`) generates a 512-d L2-normalised embedding. If CVLFace dependencies are missing, falls back to a handcrafted descriptor (low accuracy — install the full deps).

### 7. Match (`matcher.py`)

Qdrant HNSW approximate nearest-neighbour search. Only compares against records with the same `embedding_mode` to prevent cross-mode false positives. Threshold per mode:

- `cvlface` → 0.55 cosine similarity
- `fallback` → 0.94 cosine similarity

Returns `"No Match"` if best score is below threshold, `"No DB records"` if the database is empty.

### 8. Geotag (`geotagging.py`)

Runs in parallel with or after matching. See [Geofencing and geotagging](#geofencing-and-geotagging) below.

---

## Project structure

```
.
├── api.py                  # FastAPI server — all REST endpoints
├── main.py                 # CLI entry point — live camera mode
├── config.py               # Central config — all tunable parameters
├── index.html              # Browser frontend entry point
├── requirement.txt         # pip dependencies
├── start_tunnel.ps1        # Local tunnel helper (dev only)
│
├── modules/
│   ├── preprocessor.py     # Image cleanup before detection
│   ├── detector.py         # RetinaFace wrapper
│   ├── aligner.py          # 5-pt affine align + quality gate
│   ├── liveness.py         # 4-model spoof ensemble
│   ├── SASF.py             # MiniFASNet wrapper (used by liveness.py)
│   ├── embedder.py         # CVLFace AdaFace IR-101 embedding
│   ├── matcher.py          # Qdrant HNSW search
│   ├── database.py         # Qdrant client, tier management, auto-update
│   ├── geotagging.py       # GPS coords → geofence → reverse geocode
│   └── helper/             # SASF model internals (MiniFASNet arch)
│       ├── anti_spoof_predict.py
│       ├── generate_patches.py
│       ├── utility.py
│       ├── default_config.py
│       └── data_io/
│           ├── functional.py
│           └── transform.py
│
├── utils/
│   ├── visualizer.py       # OpenCV drawing helpers (CLI only)
│   └── file_picker.py      # File dialog helper (CLI only)
│
├── styles/
│   ├── styles.css
│   └── js/                 # ES module frontend (12 modules)
│       ├── main.js         # App entry, wires modules together
│       ├── config.js       # API base URL, constants
│       ├── api.js          # fetch wrapper
│       ├── camera.js       # MediaDevices, frame capture
│       ├── liveness.js     # Passive liveness polling
│       ├── geo.js          # navigator.geolocation
│       ├── status.js       # /status polling
│       ├── toast.js        # Notification UI
│       ├── map.js          # Leaflet map display
│       ├── geocard.js      # Geo result card
│       ├── result.js       # Match result rendering
│       └── action.js       # Register / match button handlers
│
└── weights/                # Model weight files (auto-downloaded if missing)
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
- CUDA 12.1 (optional, CPU works but slower)
- Windows or Linux (macOS untested with the CAP_DSHOW camera path)

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

The requirement file installs PyTorch with CUDA 12.1 binaries. For CPU-only, replace the torch line:

```bash
pip install torch torchvision torchaudio
```

### 2. Set environment variables

```bash
# Required in production — controls who can call the API
export API_KEYS="your_secret_key_here"

# Optional — set to false to disable auth during local dev
export API_AUTH_ENABLED=false

# Optional — only needed if using Google or HERE geocoding
export GEOCODER_API_KEY="your_geocoder_key"
```

### 3. Start the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

On first run, the server will download CVLFace (~250 MB) and liveness weights (~30 MB total) from HuggingFace. This only happens once — they're cached locally.

Startup output confirms what loaded:

```
[STARTUP] ✓ RetinaFace ready
[STARTUP] ✓ CVLFace ready
[STARTUP] ✓ Liveness ready — ['icm2o', 'iom2c', 'modelrgb', 'sasf']
[STARTUP] All models loaded — server ready.
```

### 4. Open the browser UI

Navigate to `http://localhost:8000` — the API serves `index.html` directly.

### 5. (Optional) Use the CLI instead

```bash
python main.py
```

This opens a live camera window. Press `SPACE` to capture, then `r` to register or `m` to match.

---

## Configuration

All parameters live in `config.py`. The most commonly changed ones:

| Parameter                          | Default        | What it controls                      |
| ---------------------------------- | -------------- | ------------------------------------- |
| `LIVENESS_ENABLED`                 | `True`         | Master liveness switch                |
| `LIVENESS_PASSIVE_REAL_THRESHOLD`  | `0.55`         | Score above this → live               |
| `LIVENESS_PASSIVE_SPOOF_THRESHOLD` | `0.50`         | Score below this → spoof              |
| `MATCH_THRESHOLDS["cvlface"]`      | `0.55`         | Cosine sim to accept a match          |
| `MIN_FACE_SIZE`                    | `90`           | Min bounding box px to accept a face  |
| `GEOFENCE_ENFORCE`                 | `False`        | Block requests from outside geofence  |
| `GEOFENCE_ZONES`                   | office example | List of circle/polygon zones          |
| `AUTO_UPDATE_ENABLED`              | `True`         | Add new embeddings on confident match |
| `GEOCODER_BACKEND`                 | `"nominatim"`  | `"nominatim"` / `"google"` / `"here"` |

### Liveness thresholds explained

The ensemble returns a `real_score` (1.0 = definitely real, 0.0 = definitely spoof):

```
real_score ≥ 0.55  →  PASS  (clearly live)
real_score < 0.50  →  FAIL  (clearly spoof)
0.50 ≤ real_score < 0.55  →  uncertain zone
    → if LIVENESS_ACTIVE_ENABLED=True and camera available → trigger head pose challenge
    → otherwise → FAIL with reason "Passive uncertain (active disabled)"
```

---

## API reference

All endpoints accept `multipart/form-data`. Authenticate by sending `X-API-Key: <your_key>` header (when `API_AUTH_ENABLED=true`).

Interactive docs available at `http://localhost:8000/docs`.

### `POST /register`

Register a new person. Fails with 409 if the name already exists.

| Field           | Type   | Required | Description                          |
| --------------- | ------ | -------- | ------------------------------------ |
| `image`         | file   | yes      | JPEG or PNG                          |
| `name`          | string | yes      | Person's name                        |
| `skip_liveness` | bool   | no       | Bypass liveness (default `false`)    |
| `passive_only`  | bool   | no       | No camera challenge (default `true`) |
| `lat` / `lon`   | float  | no       | GPS coordinates from browser         |
| `accuracy_m`    | float  | no       | GPS accuracy in metres               |

**Response:**

```json
{
  "success": true,
  "name": "Alice",
  "face_id": "face_0",
  "liveness": { "passed": true, "method": "passive", "score": 0.82, "reason": "Passive ensemble: real", "model_scores": {...} },
  "geo": { "coordinates": {...}, "address": "...", "city": "...", "geofence_passed": true },
  "embedding_mode": "cvlface",
  "message": "'Alice' registered successfully."
}
```

### `POST /match`

Identify all faces in an image. Returns one result per detected face.

Same fields as `/register`, plus `auto_update` (bool, default `true`).

**Response:**

```json
{
  "results": [
    {
      "face_id": "face_0",
      "matched": true,
      "name": "Alice",
      "score": 0.74,
      "liveness": { "passed": true, ... },
      "auto_updated": false,
      "embedding_mode": "cvlface"
    }
  ],
  "total_faces": 1,
  "matched_count": 1
}
```

### `POST /attendance`

Convenience wrapper around `/match` that returns a clean attendance payload:

```json
{
  "timestamp": 1710000000,
  "present": ["Alice", "Bob"],
  "present_count": 2,
  "unknown_count": 0,
  "spoofed_count": 0,
  "total_faces": 2,
  "geo": { ... },
  "detail": [ { "face_id": "face_0", "status": "matched", "name": "Alice", "score": 0.74 } ]
}
```

### `POST /detect`

Run detection + quality check only. No embedding or matching.

### `POST /liveness`

Run liveness on the largest face in an image. Useful for testing the ensemble.

### `GET /status`

Health check. Returns which models are loaded and current DB tier.

### `GET /db/list`

List all registered people. Add `?include_embeddings=true` to include raw 512-d vectors.

### `GET /db/exists?name=Alice`

Check if a name is registered.

### `DELETE /db/delete?name=Alice`

Remove all templates for a person.

### `GET /db/tier`

Return current HNSW scaling tier info.

---

## Liveness detection

The passive liveness system runs four models **in parallel** via `ThreadPoolExecutor` and fuses their outputs with weighted averaging.

### Models

| ID         | Type                          | Weights          | Focus                              |
| ---------- | ----------------------------- | ---------------- | ---------------------------------- |
| `ICM2O`    | PyTorch `.pth.tar`            | `ICM2O.pth.tar`  | Cross-modal texture analysis       |
| `IOM2C`    | PyTorch `.pth.tar`            | `IOM2C.pth.tar`  | Illumination-aware spoof detection |
| `modelrgb` | ONNX                          | `modelrgb.onnx`  | RGB-channel spoof detection        |
| `SASF`     | PyTorch (MiniFASNetV2 + V1SE) | two `.pth` files | Silent Face Anti-Spoofing          |

All weights are auto-downloaded from HuggingFace on first use. Place them in `weights/` manually if you're working offline.

### Ensemble fusion

Each model returns a `spoof_probability` (0.0 = definitely real, 1.0 = definitely spoof). The ensemble computes:

```python
fused_spoof = weighted_sum(spoof_probs) / total_active_weight
real_score  = 1.0 - fused_spoof
```

If a model fails to load (missing weight, CUDA OOM, etc.), its weight is redistributed proportionally to the remaining models so the system degrades gracefully rather than crashing.

### Micro-motion guard

A 5-frame rolling buffer checks for pixel-level movement between frames. Completely still faces (typical of printed photos being held still) are flagged before the models even run. Threshold: mean absolute pixel diff > 2.5 across the 5-frame window.

### Active challenge (optional)

If the passive score falls in the uncertain zone AND `LIVENESS_ACTIVE_ENABLED=True` AND a camera is available, the system triggers a **yaw head pose challenge**: the user must turn their head left or right past a threshold within `LIVENESS_HEADPOSE_TIMEOUT` seconds (default 8s).

To implement active liveness from the browser, the client must pass `passive_only=false` and a camera must be attached to the server (not the browser).

---

## Geofencing and geotagging

`modules/geotagging.py` handles all location logic. It's called on every `/register`, `/match`, and `/attendance` request.

### Coordinate resolution order

1. **Browser GPS** — if `lat` and `lon` are passed in the request form
2. **EXIF GPS** — extracted from the uploaded JPEG if browser GPS is missing
3. **No coordinates** — geofence check skipped; `geofence_passed=True` unless `GEOFENCE_ENFORCE=True`

### Geofence zone format

Define zones in `config.py` under `GEOFENCE_ZONES`:

```python
# Circular zone
{
  "id": "office",
  "name": "Main Office",
  "type": "circle",
  "center": [LAT, LON],   # decimal degrees
  "radius_m": 200
}

# Polygon zone (needs shapely installed)
{
  "id": "campus",
  "name": "Campus",
  "type": "polygon",
  "polygon": [[LAT1,LON1], [LAT2,LON2], [LAT3,LON3], ...]
}
```

Set `GEOFENCE_ENFORCE=True` to return HTTP 403 for requests outside all zones. When `False` (default), the geofence result is included in the response but doesn't block the request.

### Reverse geocoding

Supported backends (set `GEOCODER_BACKEND` in config):

- `"nominatim"` — free, OpenStreetMap-backed, 1 req/s rate limit. No key needed.
- `"google"` — requires `GEOCODER_API_KEY`
- `"here"` — requires `GEOCODER_API_KEY`

Results are cached by coordinate grid (0.0001° ≈ 11 m) for `GEOCODE_CACHE_TTL` seconds (default 1 hour). Fixed cameras will almost always hit cache.

---

## Database and scaling tiers

`modules/database.py` manages a local Qdrant instance at `./qdrant_storage`. The HNSW index parameters auto-scale based on how many unique people are registered:

| Tier | People    | Strategy | Templates/person     | HNSW m | search ef |
| ---- | --------- | -------- | -------------------- | ------ | --------- |
| 0    | 0–99      | single   | 1                    | 8      | 64        |
| 1    | 100–999   | multi    | 5                    | 16     | 128       |
| 2    | 1000–9999 | hybrid   | 3 (then consolidate) | 32     | 256       |
| 3    | 10000+    | mean     | rolling average only | 64     | 512       |

### Auto-update

When `AUTO_UPDATE_ENABLED=True`, a successful match in the medium-confidence band (`AUTO_UPDATE_MIN_CONFIDENCE` ≤ score < `AUTO_UPDATE_HIGH_CONFIDENCE`) adds the new embedding as an additional template. This keeps the model adapting to gradual appearance changes (haircuts, glasses, aging) without re-registration.

At the `hybrid` tier, templates are periodically consolidated into a `mean` embedding + `best` embedding (closest to mean) using deterministic UUIDv5 IDs — preventing duplicates on repeated consolidations.

---

## Frontend

The browser UI at `index.html` is built with 12 plain ES modules under `styles/js/`. No build step needed.

| Module        | Responsibility                                          |
| ------------- | ------------------------------------------------------- |
| `config.js`   | API base URL and shared constants                       |
| `api.js`      | `fetch` wrapper with error handling                     |
| `camera.js`   | `MediaDevices` access, frame capture to `<canvas>`      |
| `liveness.js` | Polls liveness score during preview, shows pass/fail UI |
| `geo.js`      | `navigator.geolocation` with accuracy reporting         |
| `status.js`   | Polls `/status`, shows model load state                 |
| `toast.js`    | Lightweight notification banners                        |
| `map.js`      | Leaflet map rendering for geo results                   |
| `geocard.js`  | Renders geo info card (address, timezone, zone)         |
| `result.js`   | Renders per-face match results                          |
| `action.js`   | Register and match button click handlers                |
| `main.js`     | Wires all modules together on `DOMContentLoaded`        |

To point the frontend at a different backend host, edit `BASE_URL` in `styles/js/config.js`.

---

## Running the CLI (`main.py`)

The CLI mode runs a live camera window using OpenCV — no server needed.

```bash
python main.py
```

Controls inside the camera window:

- `SPACE` — capture current frame
- `Q` — quit

After capture:

- `r` — register the captured face (prompts for name in terminal)
- `m` — match the captured face against the database
- `q` — discard and return to camera

The CLI skips geofencing and the REST layer entirely — it calls the same module functions directly.

---

## Troubleshooting

**Server exits immediately at startup**

RetinaFace or CVLFace failed to load — check the `[STARTUP]` output. Usually a missing CUDA library or a `transformers` version mismatch. The server intentionally refuses to start without detection and embedding working.

**"No DB records" even after registering**

The `embedding_mode` of the stored record doesn't match what the query is using. If CVLFace loaded for registration but not for matching (or vice versa), modes will differ. Check `/status` to confirm `cvlface_loaded` is consistent between calls.

**Liveness always fails**

- Check `/liveness` directly with a test image to isolate the issue.
- Lower `LIVENESS_PASSIVE_SPOOF_THRESHOLD` to 0.30 temporarily to see if scores are just miscalibrated.
- Check `model_scores` in the response to see which model is dragging the score down.
- Ensure the face is well-lit and not partially occluded — the quality gate runs before liveness but a near-miss face can still produce bad liveness scores.

**Geofence always blocks**

Set `GEOFENCE_ENFORCE=False` during development. Check that the `center` coordinates in `GEOFENCE_ZONES` are `[lat, lon]` order (not `[lon, lat]`).

**Camera not found (CLI)**

On Windows, the code already uses `cv2.CAP_DSHOW`. Try passing a different `camera_index` (0, 1, 2...) if you have multiple cameras.

**CVLFace download stalls**

The model is ~250 MB from HuggingFace. If it stalls, download manually into `~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m/` — the loader checks for file presence before downloading.

---

## Model sources

| Model                  | Source                                                 |
| ---------------------- | ------------------------------------------------------ |
| RetinaFace-PyTorch     | `retinaface-pytorch` pip package                       |
| CVLFace AdaFace IR-101 | HuggingFace `minchul/cvlface_adaface_ir101_webface12m` |
| ICM2O / IOM2C          | HuggingFace `mothieram/face-anti-spoofing`             |
| modelrgb ONNX          | HuggingFace `mothieram/face-anti-spoofing`             |
| MiniFASNetV2 / V1SE    | HuggingFace `mothieram/face-anti-spoofing`             |

All liveness weights auto-download on first use if not present in `weights/`.
