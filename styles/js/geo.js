// ── geo.js ────────────────────────────────────────────
// Browser geolocation: watch position, update header pill, show/hide banner.

// Exported so camera.js and actions.js can read the current fix
export let geoCoords = null; // { lat, lon, accuracy_m }
let geoWatchId = null;

// ── Pill ──────────────────────────────────────────────
export function updateGeoPill(text, ok) {
  const el = document.getElementById("p-geo");
  if (!el) return;
  el.innerHTML = `<span class="dot"></span>${text}`;
  el.className = "pill" + (ok === true ? " ok" : ok === false ? " warn" : "");
}

// ── Banner ────────────────────────────────────────────
const BANNER_MESSAGES = {
  denied: {
    text: "📍 Location access denied — please allow GPS in your browser settings to enable geotagging.",
    action: null,
  },
  unavailable: {
    text: "📍 Location unavailable — check that GPS/Location Services are enabled on this device.",
    action: null,
  },
  timeout: { text: "📍 Location timed out — retrying…", action: "retry" },
  unsupported: {
    text: "📍 This browser doesn't support geolocation. Geotagging will be skipped.",
    action: null,
  },
};

export function showGpsBanner(reason) {
  const banner = document.getElementById("gps-banner");
  if (!banner) return;
  const cfg = BANNER_MESSAGES[reason] || BANNER_MESSAGES.unavailable;
  banner.innerHTML = `
    <span class="gps-banner-icon">⚠️</span>
    <span class="gps-banner-text">${cfg.text}</span>
    ${
      cfg.action === "retry"
        ? `<button class="gps-banner-btn" onclick="window._retryGeo()">Retry</button>`
        : `<button class="gps-banner-btn" onclick="window._dismissGpsBanner()">Dismiss</button>`
    }`;
  banner.classList.remove("hidden");
}

export function hideGpsBanner() {
  document.getElementById("gps-banner")?.classList.add("hidden");
}

// ── Watch lifecycle ───────────────────────────────────
export function startGeoWatch() {
  if (!navigator.geolocation) {
    updateGeoPill("📍 No GPS", false);
    showGpsBanner("unsupported");
    return;
  }
  updateGeoPill("📍 Locating…", null);
  geoWatchId = navigator.geolocation.watchPosition(
    (pos) => {
      geoCoords = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy_m: pos.coords.accuracy,
      };
      updateGeoPill(
        `📍 Location Fixed ±${Math.round(pos.coords.accuracy)}m`,
        true,
      );
      hideGpsBanner();
    },
    (err) => {
      geoCoords = null;
      // err.code: 1=PERMISSION_DENIED, 2=POSITION_UNAVAILABLE, 3=TIMEOUT
      const reason =
        err.code === 1 ? "denied" : err.code === 3 ? "timeout" : "unavailable";
      updateGeoPill("📍 GPS " + reason, false);
      showGpsBanner(reason);
    },
    { enableHighAccuracy: true, maximumAge: 10000, timeout: 10000 },
  );
}

export function stopGeoWatch() {
  if (geoWatchId !== null) {
    navigator.geolocation.clearWatch(geoWatchId);
    geoWatchId = null;
  }
  geoCoords = null;
  updateGeoPill("📍 GPS off", null);
}

// Exposed to window so inline onclick in the banner HTML can call them
window._retryGeo = () => {
  hideGpsBanner();
  stopGeoWatch();
  startGeoWatch();
};
window._dismissGpsBanner = hideGpsBanner;
