// ── api.js ────────────────────────────────────────────
// Thin wrapper around fetch that injects the X-API-Key header.

import { API, API_KEY } from "./config.js";

export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-API-Key", API_KEY);
  return fetch(`${API}${path}`, { ...options, headers });
}
