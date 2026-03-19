// ── camera.js ─────────────────────────────────────────
// Camera stream lifecycle, live face-detect polling, and canvas overlay.

import { apiFetch } from "./api.js";
import { toast } from "./toast.js";
import { startGeoWatch, stopGeoWatch } from "./geo.js";

// Shared state — exported so other modules can read stream & captureCanvas
export let stream = null;
export const captureCanvas = document.createElement("canvas");
export let latestFaces = [];

let detectTimer = null;
let detectBusy = false;
let detectIntervalMs = 700;
let responsiveBound = false;
let currentFacingMode = "user"; // "user" = front, "environment" = rear

function applyResponsiveCameraFit() {
  const vid = document.getElementById("vid");
  const wrap = document.querySelector(".camera-wrap");
  if (!vid || !wrap || !vid.videoWidth || !vid.videoHeight) return;

  const vw = window.innerWidth || document.documentElement.clientWidth || 0;
  const vh = window.innerHeight || document.documentElement.clientHeight || 0;
  const screenAspect = vh > 0 ? vw / vh : 1;
  const videoAspect = vid.videoWidth / vid.videoHeight;

  // Mobile/tablet: always use contain to avoid perceived zoom/cropping.
  // Desktop: keep adaptive behavior.
  const useContain = vw <= 900 || (screenAspect < 1.1 && videoAspect > 1.2);
  wrap.classList.toggle("camera-fit-contain", useContain);
}

function bindResponsiveCameraFit() {
  if (responsiveBound) return;
  responsiveBound = true;
  window.addEventListener("resize", applyResponsiveCameraFit, {
    passive: true,
  });
  window.addEventListener("orientationchange", applyResponsiveCameraFit, {
    passive: true,
  });
}

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
  latestFaces = [];
  updateLiveEligibility([]);
}

export function getLargestLiveFace() {
  if (!Array.isArray(latestFaces) || !latestFaces.length) return null;
  return [...latestFaces].sort((a, b) => {
    const sa = Number(a?.score || 0);
    const sb = Number(b?.score || 0);
    return sb - sa;
  })[0];
}

// ── Bounding-box overlay ──────────────────────────────
export function drawLiveBBoxes(faces) {
  const vid = document.getElementById("vid");
  const cvs = document.getElementById("cvs");
  const ctx = cvs.getContext("2d");

  // Match canvas resolution to actual video frame size
  const vw = vid.videoWidth || 640;
  const vh = vid.videoHeight || 480;
  cvs.width = vw;
  cvs.height = vh;
  ctx.clearRect(0, 0, vw, vh);
  if (!Array.isArray(faces) || !faces.length) return;

  // Scale factor: video pixels → canvas display pixels
  // The canvas CSS size matches the wrap, but its pixel buffer = video res,
  // so no extra scaling needed — CSS handles the visual scaling.
  for (const face of faces) {
    const bbox = face.bbox || [];
    if (bbox.length < 4) continue;

    // Clamp coords to video frame bounds to prevent overflow
    const x1 = Math.max(0, Math.min(Math.round(Number(bbox[0]) || 0), vw));
    const y1 = Math.max(0, Math.min(Math.round(Number(bbox[1]) || 0), vh));
    const x2 = Math.max(0, Math.min(Math.round(Number(bbox[2]) || 0), vw));
    const y2 = Math.max(0, Math.min(Math.round(Number(bbox[3]) || 0), vh));
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
          const r = await apiFetch("/faces/detections", { method: "POST", body: fd });
          if (!r.ok) throw new Error(`detect ${r.status}`);
          const data = await r.json();
          const faces = data.faces || [];
          latestFaces = faces;
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

export function startLiveDetect(intervalMs = 700) {
  const nextMs = Math.max(200, Number(intervalMs) || 700);
  if (detectTimer && detectIntervalMs === nextMs) return;
  if (detectTimer) {
    clearInterval(detectTimer);
    detectTimer = null;
  }
  detectIntervalMs = nextMs;
  detectTimer = setInterval(tickLiveDetect, detectIntervalMs);
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

// ── Device detection ──────────────────────────────────
function isTouchPrimary() {
  return window.matchMedia("(hover: none) and (pointer: coarse)").matches;
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
    currentFacingMode = "user";
    document.getElementById("btn-switch-cam")?.classList.add("hidden");
    stopLiveDetect();
    stopGeoWatch();
    return;
  }

  // Mobile: request a smaller resolution so the front camera uses
  // its native sensor without digital zoom/upscaling.
  // 480x640 is the natural selfie resolution on most Android phones.
  // Desktop stays 1280x720 (standard webcam).
  const mobile = isTouchPrimary();
  const videoConstraints = mobile
    ? {
        facingMode: { ideal: "user" },
        width: { ideal: 480, max: 720 },
        height: { ideal: 640, max: 1280 },
      }
    : {
        facingMode: { ideal: "user" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      };

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });

    // Some mobile devices expose digital zoom on the track.
    // Force 1x to avoid sudden cropped/zoomed framing.
    try {
      const track = stream.getVideoTracks()[0];
      const caps = track?.getCapabilities?.();
      if (caps && typeof caps.zoom === "number") {
        await track.applyConstraints({ advanced: [{ zoom: 1 }] });
      }
    } catch {
      // Ignore if zoom control is unsupported.
    }

    vid.srcObject = stream;
    await vid.play();
    bindResponsiveCameraFit();
    applyResponsiveCameraFit();
    offline.style.display = "none";
    btnSnap.disabled = true;
    btnReg.disabled = false;
    btnCam.textContent = "Stop Camera";
    document.getElementById("btn-switch-cam")?.classList.remove("hidden");
    startLiveDetect();
    startGeoWatch();
  } catch (e) {
    if (e.name === "OverconstrainedError") {
      toast("Preferred camera unavailable — trying default.", false);
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "user" } },
          audio: false,
        });
        vid.srcObject = stream;
        await vid.play();
        bindResponsiveCameraFit();
        applyResponsiveCameraFit();
        offline.style.display = "none";
        btnSnap.disabled = true;
        btnReg.disabled = false;
        btnCam.textContent = "Stop Camera";
        document.getElementById("btn-switch-cam")?.classList.remove("hidden");
        startLiveDetect();
        startGeoWatch();
      } catch (e2) {
        toast("Camera error: " + e2.message, true);
      }
    } else if (e.name === "NotAllowedError") {
      toast("Camera permission denied. Please allow access and retry.", true);
    } else if (e.name === "NotFoundError") {
      toast("No camera found on this device.", true);
    } else if (e.name === "NotReadableError") {
      toast("Camera is already in use by another app.", true);
    } else {
      toast("Camera error: " + e.message, true);
    }
  }
}

// ── Switch front ↔ rear camera ────────────────────────
export async function switchCamera() {
  if (!stream) return;

  const vid = document.getElementById("vid");
  const btn = document.getElementById("btn-switch-cam");

  // Toggle facing mode
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  const isUser = currentFacingMode === "user";

  // Spin animation on button
  if (btn) {
    btn.disabled = true; // prevent double-tap during switch
    btn.classList.add("spinning");
    setTimeout(() => {
      btn.classList.remove("spinning");
      btn.disabled = false;
    }, 400);
  }

  const mobile = isTouchPrimary();
  const videoConstraints = mobile
    ? {
        facingMode: { exact: currentFacingMode },
        width: { ideal: isUser ? 480 : 1280, max: isUser ? 720 : 1920 },
        height: { ideal: isUser ? 640 : 720, max: isUser ? 1280 : 1080 },
      }
    : {
        facingMode: { ideal: currentFacingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      };

  let newStream = null;
  try {
    newStream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
  } catch (e) {
    // exact facingMode failed — try ideal fallback
    if (e.name === "OverconstrainedError" || e.name === "NotFoundError") {
      try {
        newStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: currentFacingMode } },
          audio: false,
        });
      } catch (e2) {
        currentFacingMode = isUser ? "environment" : "user"; // revert
        toast("Could not switch camera.", true);
        return;
      }
    } else {
      currentFacingMode = isUser ? "environment" : "user"; // revert
      toast("Could not switch camera: " + e.message, true);
      return;
    }
  }

  // ── Seamless swap ─────────────────────────────────────
  // Stop OLD tracks AFTER new stream is ready — no black frame gap
  const oldTracks = stream.getTracks();
  stream = newStream;
  vid.srcObject = newStream;
  // Let video load then stop old tracks — keeps detect loop alive throughout
  vid.onloadedmetadata = () => {
    vid.play().catch(() => {});
    oldTracks.forEach((t) => t.stop());
    vid.onloadedmetadata = null;
    applyResponsiveCameraFit();
  };
  // Reset zoom to 1x on new track
  try {
    const track = newStream.getVideoTracks()[0];
    const caps = track?.getCapabilities?.();
    if (caps && typeof caps.zoom === "number") {
      await track.applyConstraints({ advanced: [{ zoom: 1 }] });
    }
  } catch {
    /* ignore */
  }
}

