// ── status.js ─────────────────────────────────────────
// Server status bar: polls /v1/status and updates the header pills.

import { apiFetch } from "./api.js";
import { toast } from "./toast.js";

export function setPill(id, text, ok) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `<span class="dot"></span>${text}`;
  el.className = "pill " + (ok ? "ok" : "warn");
}

export async function loadStatus() {
  try {
    const resp = await apiFetch("/status");
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const d = await resp.json();
    setPill("p-server", "Server OK", true);
    setPill(
      "p-cvl",
      d.cvlface_loaded ? "CVLFace ✓" : "CVLFace ✗",
      d.cvlface_loaded,
    );
    setPill(
      "p-live",
      d.liveness_enabled ? "Liveness ✓" : "Liveness off",
      d.liveness_enabled,
    );
    const t = d.db_tier;
    setPill("p-tier", `${t.strategy} · m=${t.hnsw_m}`, true);
  } catch {
    setPill("p-server", "Server offline", false);
  }
}

export function initStatus() {
  loadStatus();
  setInterval(loadStatus, 30000);
}
