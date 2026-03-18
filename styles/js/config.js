// ── config.js ─────────────────────────────────────────
// Central place for API base URL, auth key, and tuning constants.

export const API =
  window.location.protocol === "file:" ? "http://localhost:8000" : "";

export const API_KEY = "developer_key";

// Passive liveness threshold alignment (real score >= threshold => skip active challenge)
export const PASSIVE_REAL_THRESHOLD = 0.55;

// Active liveness challenge tuning
export const ACTIVE_CALIB_TARGET = 5;
export const ACTIVE_CALIB_MIN = 3;
export const ACTIVE_TURN_THRESHOLD = 0.2;
export const ACTIVE_TIMEOUT_MS = 12000;

// Map thumbnail zoom level
export const MAP_ZOOM = 13;
