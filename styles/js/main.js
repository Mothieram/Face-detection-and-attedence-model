// ── main.js ───────────────────────────────────────────
// Entry point. Initialises all modules and exposes the functions
// that index.html's onclick attributes need on window.

import { toast } from "./toast.js";
import { initStatus } from "./status.js";
import { toggleCamera, switchCamera } from "./camera.js";
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

// Start GPS immediately on page load — don't wait for camera start.
// Pill becomes live and tappable as soon as the page opens.
startGeoWatch();

if (window.location.protocol === "file:") {
  toast(
    "Open this page via http://127.0.0.1:5500 (not file://) to avoid CORS.",
    true,
  );
}
