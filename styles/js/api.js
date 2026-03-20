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
 * Extract the human-readable message from any error response shape:
 *   new shape  → { error: "CODE", message: "...", field: "...", code: "..." }
 *   old shape  → { detail: "..." }
 *   fallback   → HTTP status string
 */
export function extractErrorMessage(payload, httpStatus) {
  if (!payload) return String(httpStatus);
  // New structured shape
  if (payload.message) return payload.message;
  // FastAPI default / old shape
  if (typeof payload.detail === "string") return payload.detail;
  // FastAPI validation error — detail is an array of objects
  if (Array.isArray(payload.detail) && payload.detail.length > 0) {
    const first = payload.detail[0];
    const field = first.loc ? first.loc[first.loc.length - 1] : null;
    return field ? `Invalid value for '${field}': ${first.msg}` : first.msg;
  }
  return String(httpStatus);
}

/**
 * Extract the machine-readable error code from the response payload.
 * Returns null when no code is available (non-structured error).
 */
export function extractErrorCode(payload) {
  if (!payload) return null;
  return payload.code || payload.error || null;
}

/**
 * Format error for display — includes the code badge when available.
 * e.g. "[NO_FACE_DETECTED] No face detected in the image..."
 */
export function formatError(payload, httpStatus) {
  const msg = extractErrorMessage(payload, httpStatus);
  const code = extractErrorCode(payload);
  return code ? `[${code}] ${msg}` : msg;
}
