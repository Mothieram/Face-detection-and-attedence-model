// ── main.js ───────────────────────────────────────────
// Entry point. Initialises all modules and exposes the functions
// that index.html's onclick attributes need on window.

import { toast } from "./toast.js";
import { initStatus } from "./status.js";
import {
  toggleCamera,
  switchCamera,
  pauseCamera,
  resumeCamera,
} from "./camera.js";
import { captureAndMatch, captureAndRegister } from "./action.js";
import { startGeoWatch, tapGeoPill } from "./geo.js";

// ── Expose to HTML onclick handlers ──────────────────
window.toggleCamera = toggleCamera;
window.switchCamera = switchCamera;
window.captureAndMatch = captureAndMatch;
window.captureAndRegister = captureAndRegister;

// Exposed so the static onclick on the GPS pill in index.html always works,
// even before watchPosition has fired for the first time.
window._tapGeoPill = () => tapGeoPill();

// ── Boot ──────────────────────────────────────────────
initStatus();

// ── Page visibility — stop camera when tab/app is hidden ─────────────────
// Fires on: tab switch, home button, screen lock, app switcher, browser close.
// This is the correct cross-browser way to detect background/foreground.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    // Page going to background — stop detection and release camera
    pauseCamera();
  } else if (document.visibilityState === "visible") {
    // Page coming back to foreground — restart camera if it was running
    resumeCamera();
  }
});

// pagehide fires on iOS Safari when the page is being unloaded / swiped away.
// visibilitychange alone is not reliable on iOS for true tab close.
window.addEventListener(
  "pagehide",
  () => {
    pauseCamera();
  },
  { capture: true },
);

// Start GPS immediately on page load — don't wait for camera start.
// Pill becomes live and tappable as soon as the page opens.
startGeoWatch();

if (window.location.protocol === "file:") {
  toast(
    "Open this page via http://127.0.0.1:5500 (not file://) to avoid CORS.",
    true,
  );
}
