// ── geo.js ────────────────────────────────────────────
// Browser geolocation: watch position, update header pill live, show/hide banner.
//
// Changes from original:
//  1. Pill updates live (no page refresh needed) — watchPosition fires on every
//     GPS state change and updates the pill + banner immediately.
//  2. Pill is clickable when GPS is off — tapping it prompts the user to enable
//     location, and on mobile opens device location settings where possible.

// Exported so camera.js and actions.js can read the current fix
export let geoCoords = null; // { lat, lon, accuracy_m }
let geoWatchId = null;
let _geoState = "off"; // "off" | "locating" | "fixed" | "denied" | "unavailable" | "timeout"

// ── Pill ──────────────────────────────────────────────
export function updateGeoPill(text, ok) {
  const el = document.getElementById("p-geo");
  if (!el) return;
  el.innerHTML = `<span class="dot"></span>${text}`;
  el.className = "pill" + (ok === true ? " ok" : ok === false ? " warn" : "");

  // Make pill clickable when GPS is off or errored so the user can tap to retry.
  const isOff = ok !== true && _geoState !== "locating";
  el.style.cursor = isOff ? "pointer" : "default";
  el.onclick = isOff ? _handlePillTap : null;
  el.title = isOff ? "Tap to enable GPS" : "";
}

// ── Tap handler — called when user taps the pill while GPS is off ─────────
// IMPORTANT: must be called synchronously inside a user gesture (tap/click).
// Browsers only show the location permission dialog when getCurrentPosition()
// is called directly inside a gesture handler — watchPosition() called outside
// a gesture is silently denied on Android Chrome and iOS Safari.
function _handlePillTap() {
  if (_geoState === "locating") return;

  stopGeoWatch();
  _geoState = "locating";
  updateGeoPill("📍 Locating…", null);

  // getCurrentPosition called synchronously inside the tap — this is what
  // triggers the OS permission popup on mobile. watchPosition alone won't do it.
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      // Permission granted — store the fix immediately
      geoCoords = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy_m: pos.coords.accuracy,
      };
      _geoState = "fixed";
      updateGeoPill(`📍 GPS ±${Math.round(pos.coords.accuracy)}m`, true);
      hideGpsBanner();
      // Hand off to watchPosition for continuous live updates
      startGeoWatch();
    },
    (err) => {
      geoCoords = null;
      if (err.code === 1) {
        _geoState = "denied";
        updateGeoPill("📍 GPS denied — tap to enable", false);
        showGpsBanner("denied");
      } else {
        _geoState = "unavailable";
        updateGeoPill("📍 GPS off — tap to enable", false);
        showGpsBanner("settings");
      }
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 },
  );
}

// ── Device detection ─────────────────────────────────
function _getDevice() {
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod/i.test(ua)) return "ios";
  if (/Android/i.test(ua)) {
    return /SamsungBrowser/i.test(ua) ? "samsung" : "android";
  }
  return "desktop";
}

// Returns step-by-step instructions for the current device/browser combo.
// No button can programmatically open device settings — so we give the user
// the exact path to tap themselves.
function _getSettingsSteps(reason) {
  const device = _getDevice();
  const isDenied = reason === "denied";

  if (device === "ios") {
    return isDenied
      ? [
          "Open the <strong>Settings</strong> app on your iPhone/iPad",
          "Scroll down and tap <strong>Safari</strong> (or your browser)",
          "Tap <strong>Location</strong>",
          "Select <strong>Allow</strong>",
          "Come back here and tap <strong>Retry</strong>",
        ]
      : [
          "Open the <strong>Settings</strong> app on your iPhone/iPad",
          "Tap <strong>Privacy &amp; Security</strong>",
          "Tap <strong>Location Services</strong>",
          "Turn the toggle <strong>On</strong>",
          "Come back here and tap <strong>Retry</strong>",
        ];
  }

  if (device === "samsung") {
    return isDenied
      ? [
          "Tap the <strong>⋮ menu</strong> in Samsung Internet",
          "Go to <strong>Settings → Sites and downloads → Location</strong>",
          "Find this site and set it to <strong>Allow</strong>",
          "Tap <strong>Retry</strong> below",
        ]
      : [
          "Swipe down from the top of your screen",
          "Tap the <strong>Location</strong> tile to turn it on",
          "Tap <strong>Retry</strong> below",
        ];
  }

  if (device === "android") {
    return isDenied
      ? [
          "Tap the <strong>🔒 lock icon</strong> in Chrome's address bar",
          "Tap <strong>Permissions → Location</strong>",
          "Set it to <strong>Allow</strong>",
          "Tap <strong>Retry</strong> below",
        ]
      : [
          "Swipe down from the top of your screen",
          "Tap the <strong>Location</strong> tile to turn it on",
          "Tap <strong>Retry</strong> below",
        ];
  }

  // Desktop
  return isDenied
    ? [
        "Click the <strong>🔒 lock icon</strong> in your browser's address bar",
        "Find <strong>Location</strong> and set it to <strong>Allow</strong>",
        "Click <strong>Retry</strong> below",
      ]
    : [
        "Enable Location Services on this device, then click <strong>Retry</strong>",
      ];
}

export function showGpsBanner(reason) {
  const banner = document.getElementById("gps-banner");
  if (!banner) return;

  const isTimeout = reason === "timeout";
  const isUnsupported = reason === "unsupported";

  let bodyHtml = "";
  let retryBtn = `<button class="gps-banner-btn" onclick="window._retryGeo()">Retry</button>`;

  if (isUnsupported) {
    bodyHtml =
      '<span class="gps-banner-text">This browser doesn\'t support geolocation. Geotagging will be skipped.</span>';
    retryBtn = "";
  } else if (isTimeout) {
    bodyHtml =
      '<span class="gps-banner-text">Location timed out — tap Retry to try again.</span>';
  } else {
    // denied / unavailable / settings — show device-specific step-by-step
    const steps = _getSettingsSteps(reason);
    const stepsHtml = steps
      .map(
        (s, i) =>
          `<li><span class="gps-step-num">${i + 1}</span><span class="gps-step-text">${s}</span></li>`,
      )
      .join("");
    const heading =
      reason === "denied"
        ? "Location blocked — how to fix:"
        : "GPS is off — how to fix:";
    bodyHtml = `
      <div class="gps-banner-guide">
        <div class="gps-banner-heading">${heading}</div>
        <ol class="gps-steps">${stepsHtml}</ol>
      </div>`;
  }

  banner.innerHTML = `
    <div class="gps-banner-top">
      <span class="gps-banner-icon">📍</span>
      <div class="gps-banner-body">${bodyHtml}</div>
      <button class="gps-banner-btn gps-banner-dismiss" onclick="window._dismissGpsBanner()">✕</button>
    </div>
    ${retryBtn ? `<div class="gps-banner-actions">${retryBtn}</div>` : ""}`;
  banner.classList.remove("hidden");
}

export function hideGpsBanner() {
  document.getElementById("gps-banner")?.classList.add("hidden");
}

// ── Watch lifecycle ───────────────────────────────────
export function startGeoWatch() {
  // Already watching — don't open a second watchPosition
  if (geoWatchId !== null) return;

  if (!navigator.geolocation) {
    _geoState = "unsupported";
    updateGeoPill("📍 No GPS", false);
    showGpsBanner("unsupported");
    return;
  }

  _geoState = "locating";
  updateGeoPill("📍 Locating…", null);

  geoWatchId = navigator.geolocation.watchPosition(
    // ── Success — position received ──────────────────
    (pos) => {
      geoCoords = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy_m: pos.coords.accuracy,
      };
      _geoState = "fixed";
      // Live update every time GPS accuracy improves — no page refresh needed
      updateGeoPill(`📍 GPS ±${Math.round(pos.coords.accuracy)}m`, true);
      hideGpsBanner();
    },

    // ── Error — GPS denied, off, or timed out ────────
    (err) => {
      geoCoords = null;
      // err.code: 1 = PERMISSION_DENIED, 2 = POSITION_UNAVAILABLE, 3 = TIMEOUT
      if (err.code === 1) {
        _geoState = "denied";
        updateGeoPill("📍 GPS denied — tap to enable", false);
        showGpsBanner("denied");
      } else if (err.code === 3) {
        _geoState = "timeout";
        updateGeoPill("📍 GPS timeout — tap to retry", false);
        showGpsBanner("timeout");
      } else {
        _geoState = "unavailable";
        updateGeoPill("📍 GPS off — tap to enable", false);
        showGpsBanner("unavailable");
      }
    },

    { enableHighAccuracy: true, maximumAge: 10000, timeout: 15000 },
  );
}

export function stopGeoWatch() {
  if (geoWatchId !== null) {
    navigator.geolocation.clearWatch(geoWatchId);
    geoWatchId = null;
  }
  geoCoords = null;
  _geoState = "off";
  updateGeoPill("📍 GPS off — tap to enable", null);
}

// ── Window-level callbacks used by inline banner buttons ─────────────────

// Export so main.js can wire it to window._tapGeoPill
// (calling this from an onclick is what triggers the OS permission dialog)
export function tapGeoPill() {
  _handlePillTap();
}

window._retryGeo = () => {
  hideGpsBanner();
  stopGeoWatch();
  startGeoWatch();
};

window._dismissGpsBanner = hideGpsBanner;

// _openLocationSettings removed — no browser API can open device settings.
// The banner now shows device-specific step-by-step instructions instead.
