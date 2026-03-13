// ── results.js ────────────────────────────────────────
// Renders the face match result cards into #results.

import { toast } from "./toast.js";
import { buildGeoCard } from "./geocard.js";

function fmtBbox(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 4) return null;
  return `bbox [${bbox
    .slice(0, 4)
    .map((v) => Math.round(Number(v) || 0))
    .join(", ")}]`;
}

export function renderResults(data) {
  const box = document.getElementById("results");

  if (!data.results || data.results.length === 0) {
    box.innerHTML =
      '<div class="empty"><span class="icon">🔍</span>No faces detected or all failed quality check.</div>';
    return;
  }

  const sharedGeo = data.geo || null;

  box.innerHTML = data.results
    .map((res) => {
      const isSpoof = res.name === "Spoof";
      const isUncertain = res.name === "Uncertain";
      const isRejected = res.name === "Rejected";

      const cls = res.matched
        ? "matched"
        : isSpoof
          ? "spoof"
          : isUncertain
            ? "uncertain"
            : isRejected
              ? "spoof"
              : "";
      const emoji = res.matched
        ? "👤"
        : isSpoof
          ? "⚠️"
          : isUncertain
            ? "🟡"
            : isRejected
              ? "🚫"
              : "❓";
      const sc = res.score > 0.65 ? "high" : "low";
      const lv = res.liveness;
      const q = res.quality;

      const meta = [
        res.face_id,
        fmtBbox(res.bbox),
        q ? `quality ${q.passed ? "ok" : "fail"}` : null,
        q?.reason || null,
        lv?.method ? `method ${lv.method}` : null,
        lv ? `liveness ${lv.score.toFixed(2)}` : null,
        lv?.reason || null,
        res.auto_updated ? "updated" : null,
        res.embedding_mode || null,
      ]
        .filter(Boolean)
        .join(" · ");

      const geoHtml = res.matched ? buildGeoCard(res.geo || sharedGeo) : "";

      return `
    <div class="card ${cls}">
      <div class="avatar ${cls}">${emoji}</div>
      <div class="card-info">
        <div class="card-name">${res.name}</div>
        <div class="card-meta">${meta}</div>
      </div>
      <div class="score ${sc}">${res.score.toFixed(3)}</div>
    </div>${geoHtml}`;
    })
    .join("");

  toast(`${data.matched_count} of ${data.total_faces} face(s) matched`);
}
