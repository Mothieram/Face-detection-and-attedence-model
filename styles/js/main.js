// ── main.js ───────────────────────────────────────────
// Entry point. Initialises all modules and exposes the functions
// that index.html's onclick attributes need on window.

import { toast } from "./toast.js";
import { initStatus } from "./status.js";
import { toggleCamera } from "./camera.js";
import { captureAndMatch, captureAndRegister } from "./action.js";

// ── Expose to HTML onclick handlers ──────────────────
window.toggleCamera = toggleCamera;
window.captureAndMatch = captureAndMatch;
window.captureAndRegister = captureAndRegister;

// ── Boot ──────────────────────────────────────────────
initStatus();

if (window.location.protocol === "file:") {
  toast(
    "Open this page via http://127.0.0.1:5500 (not file://) to avoid CORS.",
    true,
  );
}
