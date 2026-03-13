// ── camera.js ─────────────────────────────────────────
// Camera stream lifecycle, live face-detect polling, and canvas overlay.

import { apiFetch } from "./api.js";
import { toast } from "./toast.js";
import { startGeoWatch, stopGeoWatch } from "./geo.js";

// Shared state — exported so other modules can read stream & captureCanvas
export let stream = null;
export const captureCanvas = document.createElement("canvas");

let detectTimer = null;
let detectBusy = false;

// ── Live eligibility ──────────────────────────────────
export function updateLiveEligibility(faces) {
  const w = document.getElementById("cam-warning");
  const btnSnap = document.getElementById("btn-snap");
  if (!w || !btnSnap) return;

  const list = Array.isArray(faces) ? faces : [];
  const tooMany = list.length > 1;
  const oneFace = list.length === 1;
  const qualityOk = oneFace && !!(list[0].quality && list[0].quality.passed);

  w.classList.toggle("hidden", !tooMany);
  btnSnap.disabled = !(!!stream && oneFace && qualityOk);
}

export function clearOverlay() {
  const cvs = document.getElementById("cvs");
  const ctx = cvs.getContext("2d");
  ctx.clearRect(0, 0, cvs.width || 1, cvs.height || 1);
  updateLiveEligibility([]);
}

// ── Bounding-box overlay ──────────────────────────────
export function drawLiveBBoxes(faces) {
  const vid = document.getElementById("vid");
  const cvs = document.getElementById("cvs");
  const ctx = cvs.getContext("2d");

  cvs.width = vid.videoWidth || 640;
  cvs.height = vid.videoHeight || 480;
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  if (!Array.isArray(faces) || !faces.length) return;

  for (const face of faces) {
    const bbox = face.bbox || [];
    if (bbox.length < 4) continue;
    const [x1, y1, x2, y2] = bbox.map((v) => Math.round(Number(v) || 0));
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const ok = !!(face.quality && face.quality.passed);

    ctx.strokeStyle = ok ? "#4af0a0" : "#ff6b6b";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    const label = `${face.face_id || "face"} ${Number(face.score || 0).toFixed(2)}`;
    ctx.font = '12px "DM Mono", monospace';
    ctx.textBaseline = "top";
    const tw = Math.ceil(ctx.measureText(label).width) + 8;
    const th = 18;
    const tx = Math.max(0, x1);
    const ty = Math.max(0, y1 - th - 2);
    ctx.fillStyle = ok ? "rgba(74,240,160,0.9)" : "rgba(255,107,107,0.9)";
    ctx.fillRect(tx, ty, tw, th);
    ctx.fillStyle = "#0a0a0f";
    ctx.fillText(label, tx + 4, ty + 3);
  }
}

// ── Live detect loop ──────────────────────────────────
async function tickLiveDetect() {
  if (!stream || detectBusy) return;
  const vid = document.getElementById("vid");
  if (!vid.videoWidth || !vid.videoHeight) return;

  detectBusy = true;
  try {
    captureCanvas.width = vid.videoWidth;
    captureCanvas.height = vid.videoHeight;
    captureCanvas.getContext("2d").drawImage(vid, 0, 0);
    captureCanvas.toBlob(
      async (blob) => {
        if (!blob) {
          clearOverlay();
          detectBusy = false;
          return;
        }
        try {
          const fd = new FormData();
          fd.append("image", blob, "live.jpg");
          const r = await apiFetch("/detect", { method: "POST", body: fd });
          if (!r.ok) throw new Error(`detect ${r.status}`);
          const data = await r.json();
          const faces = data.faces || [];
          updateLiveEligibility(faces);
          drawLiveBBoxes(faces);
        } catch {
          clearOverlay();
        } finally {
          detectBusy = false;
        }
      },
      "image/jpeg",
      0.7,
    );
  } catch {
    detectBusy = false;
  }
}

export function startLiveDetect() {
  if (detectTimer) return;
  detectTimer = setInterval(tickLiveDetect, 700);
}

export function stopLiveDetect() {
  if (detectTimer) {
    clearInterval(detectTimer);
    detectTimer = null;
  }
  clearOverlay();
}

// ── Capture helper ────────────────────────────────────
export function captureFrameBlob(quality = 0.92) {
  return new Promise((resolve) => {
    const vid = document.getElementById("vid");
    captureCanvas.width = vid.videoWidth || 640;
    captureCanvas.height = vid.videoHeight || 480;
    captureCanvas.getContext("2d").drawImage(vid, 0, 0);
    captureCanvas.toBlob(
      (blob) => resolve(blob || null),
      "image/jpeg",
      quality,
    );
  });
}

// ── Camera toggle (bound to button in HTML) ───────────
export async function toggleCamera() {
  const vid = document.getElementById("vid");
  const offline = document.getElementById("cam-offline");
  const btnCam = document.getElementById("btn-cam");
  const btnSnap = document.getElementById("btn-snap");
  const btnReg = document.getElementById("btn-reg");

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    vid.srcObject = null;
    offline.style.display = "flex";
    btnSnap.disabled = true;
    btnReg.disabled = true;
    btnCam.textContent = "Start Camera";
    stopLiveDetect();
    stopGeoWatch();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
    });
    vid.srcObject = stream;
    await vid.play();
    offline.style.display = "none";
    btnSnap.disabled = true;
    btnReg.disabled = false;
    btnCam.textContent = "Stop Camera";
    startLiveDetect();
    startGeoWatch();
  } catch (e) {
    toast("Camera error: " + e.message, true);
  }
}
