// ── geocard.js ────────────────────────────────────────
// Builds the geo-tag card HTML shown below matched face cards.

import { renderMapCanvas } from "./map.js";
import { MAP_ZOOM } from "./config.js";

function formatLocalTime(localTime) {
  if (!localTime) return "";
  try {
    const d = new Date(localTime);
    const dd = String(d.getDate()).padStart(2, "0");
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const yy = String(d.getFullYear()).slice(2);
    let h = d.getHours();
    const mins = String(d.getMinutes()).padStart(2, "0");
    const ampm = h >= 12 ? "PM" : "AM";
    h = h % 12 || 12;
    return `${dd}/${mm}/${yy} ${String(h).padStart(2, "0")}:${mins} ${ampm}`;
  } catch {
    return localTime.replace("T", " ").slice(0, 19);
  }
}

export function buildGeoCard(geo) {
  if (!geo?.coordinates) return "";

  const { lat, lon, accuracy_m } = geo.coordinates;
  const city = geo.city || "";
  const country = geo.country || "";

  // Line 1: "Coimbatore / India" — bold
  const placeLine1 = [city, country].filter(Boolean).join(" / ") || "Unknown";

  // Line 2: full address minus city & country to avoid repetition
  const addressParts = (geo.address || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const placeLine2 = addressParts
    .filter(
      (p) =>
        p.toLowerCase() !== city.toLowerCase() &&
        p.toLowerCase() !== country.toLowerCase(),
    )
    .join(", ");

  const acc = accuracy_m != null ? `±${Math.round(accuracy_m)}m` : "";
  const mapsUrl = `https://www.google.com/maps?q=${lat},${lon}`;
  const timeStr = formatLocalTime(geo.local_time);
  const canvasId = `geo-map-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

  // Kick off tile rendering after the element is in the DOM
  setTimeout(() => renderMapCanvas(canvasId, lat, lon, MAP_ZOOM), 0);

  return `
  <a class="geo-card-v2" href="${mapsUrl}" target="_blank" rel="noopener">
    <div class="geo-thumb-wrap">
      <canvas id="${canvasId}" class="geo-thumb" width="100" height="90"></canvas>
    </div>
    <div class="geo-info">
      <div class="geo-info-line1">${placeLine1}</div>
      ${placeLine2 ? `<div class="geo-info-line2">${placeLine2}</div>` : ""}
      <div class="geo-info-coords">Lat: ${Number(lat).toFixed(6)}, Long: ${Number(lon).toFixed(6)}${acc ? " " + acc : ""}</div>
      ${timeStr ? `<div class="geo-info-time">${timeStr}</div>` : ""}
    </div>
  </a>`;
}
