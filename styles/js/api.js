// ── api.js ────────────────────────────────────────────
// Thin fetch wrapper that:
//  - injects the X-API-Key header on every request
//  - normalises error responses to always expose .message
//    (handles both old {"detail":...} and new {"error":...,"message":...} shapes)

import { API_V1, API_KEY } from "./config.js";

export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-API-Key", API_KEY);
  // Tells ngrok to skip its browser interstitial warning page and return
  // the real API response directly. Harmless on non-ngrok servers.
  headers.set("ngrok-skip-browser-warning", "true");
  return fetch(`${API_V1}${path}`, { ...options, headers });
}

/**
 * Extract a human-readable message from any error response shape:
 *   new shape → { error: "CODE", message: "..." }
 *   FastAPI default → { detail: "..." }
 *   fallback → HTTP status text
 */
export function extractErrorMessage(payload, httpStatus) {
  if (!payload) return String(httpStatus);
  return payload.message || payload.detail || String(httpStatus);
}
