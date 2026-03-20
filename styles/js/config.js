// ── config.js ─────────────────────────────────────────
// Central place for API base URL, auth key, and tuning constants.

function normalizeBaseUrl(url) {
  return String(url || "")
    .trim()
    .replace(/\/+$/, "");
}

const apiFromQuery = new URLSearchParams(window.location.search).get("api");
const apiFromStorage = window.localStorage.getItem("API_BASE");

export const API = normalizeBaseUrl(
  apiFromQuery ||
    apiFromStorage ||
    // file:// protocol → explicit localhost (no origin available)
    // http/https       → use whatever origin served the page.
    //                    Works for localhost:8000, ngrok tunnels, and
    //                    any other public URL without changing this file.
    (window.location.protocol === "file:"
      ? "http://localhost:8000"
      : window.location.origin),
);

// All endpoints are under /v1/
export const API_V1 = `${API}/v1`;

// API key — read from ?key= URL param or localStorage.
// First visit: open https://your-ngrok-url/?key=YOUR_KEY
// After that: key is remembered in localStorage automatically.
const _keyFromQuery = new URLSearchParams(window.location.search).get("key");
const _keyFromStorage = window.localStorage.getItem("API_KEY");
export const API_KEY = _keyFromQuery || _keyFromStorage || "";

// Save key from URL to localStorage so it persists across refreshes
if (_keyFromQuery) window.localStorage.setItem("API_KEY", _keyFromQuery);

// Passive liveness threshold alignment (real score >= threshold => skip active challenge)
export const PASSIVE_REAL_THRESHOLD = 0.55;

// Active liveness challenge tuning
export const ACTIVE_CALIB_TARGET = 5;
export const ACTIVE_CALIB_MIN = 3;
export const ACTIVE_TURN_THRESHOLD = 0.2;
export const ACTIVE_TIMEOUT_MS = 12000;

// Map thumbnail zoom level
export const MAP_ZOOM = 13;
