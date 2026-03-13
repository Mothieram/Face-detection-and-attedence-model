// ── config.js ─────────────────────────────────────────
// Central place for API base URL, auth key, and tuning constants.

export const API =
  window.location.protocol === "file:"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname || "localhost"}:8000`;

export const API_KEY = "developer_key";

// Active liveness challenge tuning
export const ACTIVE_CALIB_TARGET = 5;
export const ACTIVE_CALIB_MIN = 3;
export const ACTIVE_TURN_THRESHOLD = 0.3;
export const ACTIVE_TIMEOUT_MS = 12000;

// Map thumbnail zoom level
export const MAP_ZOOM = 13;
