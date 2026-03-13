// Lightweight liveness helpers used by action.js.
// Keeps UI consistent and delegates final checks to backend /liveness.

import { apiFetch } from "./api.js";
import { captureFrameBlob, startLiveDetect, stopLiveDetect, stream } from "./camera.js";
import {
  ACTIVE_CALIB_MIN,
  ACTIVE_CALIB_TARGET,
  ACTIVE_TIMEOUT_MS,
  ACTIVE_TURN_THRESHOLD,
} from "./config.js";

export function setChallengeUI(text, progress = 0, active = false, failed = false) {
  const box = document.getElementById("challenge-box");
  const txt = document.getElementById("challenge-text");
  const fill = document.getElementById("challenge-fill");
  if (!box || !txt || !fill) return;

  txt.textContent = text || "Active challenge idle";
  fill.style.width = `${Math.max(0, Math.min(100, Number(progress) || 0))}%`;

  box.classList.toggle("hidden", !active && !failed && progress <= 0);
  box.classList.toggle("warn", !!failed);
}

export async function runPassiveLivenessCheck(imageBlob) {
  if (!imageBlob) return { passed: false, score: 0, reason: "empty_frame" };

  const fd = new FormData();
  fd.append("image", imageBlob, "passive.jpg");
  fd.append("passive_only", "true");
  fd.append("camera_index", "0");

  const r = await apiFetch("/liveness", { method: "POST", body: fd });
  const payload = await r.json().catch(() => ({}));
  if (!r.ok) {
    return {
      passed: false,
      score: 0,
      reason: payload.detail || payload.reason || `http_${r.status}`,
    };
  }
  return {
    passed: !!payload.passed,
    score: Number(payload.score || 0),
    reason: String(payload.reason || ""),
    method: payload.method || "passive",
  };
}

export async function runClientActiveChallenge() {
  stopLiveDetect();

  // 1) Calibrate neutral head pose
  setChallengeUI("Calibration: hold face straight", 0, false, true);
  const neutralSamples = [];
  const calibDeadline = Date.now() + 3000;

  while (Date.now() < calibDeadline && neutralSamples.length < ACTIVE_CALIB_TARGET) {
    const face = await _detectLargestFaceOnce();
    if (face?.landmarks) {
      neutralSamples.push(_estimateYawFromLandmarks(face.landmarks));
    }
    const p = Math.round((neutralSamples.length / ACTIVE_CALIB_TARGET) * 100);
    setChallengeUI(
      `Calibration: ${Math.min(neutralSamples.length, ACTIVE_CALIB_TARGET)}/${ACTIVE_CALIB_TARGET}`,
      p,
      false,
      true,
    );
    await _sleep(220);
  }

  if (neutralSamples.length < ACTIVE_CALIB_MIN) {
    if (stream) startLiveDetect();
    setChallengeUI("Calibration failed", 100, true, true);
    return { passed: false, reason: "Face not stable for calibration", score: 0 };
  }

  const sorted = [...neutralSamples].sort((a, b) => a - b);
  const neutralYaw = sorted[Math.floor(sorted.length / 2)];

  // 2) Ask for one clear turn either side from neutral
  const deadline = Date.now() + ACTIVE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const face = await _detectLargestFaceOnce();
    const remaining = Math.max(0, deadline - Date.now());
    const secs = (remaining / 1000).toFixed(1);

    if (face?.landmarks) {
      const yaw = _estimateYawFromLandmarks(face.landmarks) - neutralYaw;
      const turnProgress = Math.min(
        100,
        Math.round((Math.abs(yaw) / ACTIVE_TURN_THRESHOLD) * 100),
      );
      setChallengeUI(`Turn head LEFT or RIGHT (${secs}s)`, turnProgress, false, true);

      if (Math.abs(yaw) >= ACTIVE_TURN_THRESHOLD) {
        if (stream) startLiveDetect();
        setChallengeUI("Active challenge passed", 100, false, true);
        return { passed: true, reason: "Active challenge passed", score: 1 };
      }
    } else {
      setChallengeUI(`Turn head LEFT or RIGHT (${secs}s)`, 0, false, true);
    }

    await _sleep(220);
  }

  if (stream) startLiveDetect();
  setChallengeUI("Head turn not detected", 100, true, true);
  return { passed: false, reason: "Head turn not detected in time", score: 0 };
}

async function _detectLargestFaceOnce() {
  const blob = await captureFrameBlob(0.75);
  if (!blob) return null;
  const fd = new FormData();
  fd.append("image", blob, "active.jpg");
  const r = await apiFetch("/detect", { method: "POST", body: fd });
  if (!r.ok) return null;
  const payload = await r.json().catch(() => ({}));
  const faces = Array.isArray(payload.faces) ? payload.faces : [];
  if (!faces.length) return null;
  return faces.sort((a, b) => (Number(b.score) || 0) - (Number(a.score) || 0))[0];
}

function _estimateYawFromLandmarks(landmarks) {
  if (
    !landmarks ||
    !landmarks.left_eye ||
    !landmarks.right_eye ||
    !landmarks.nose
  ) {
    return 0;
  }
  const le = landmarks.left_eye;
  const re = landmarks.right_eye;
  const nose = landmarks.nose;
  const midX = (Number(le[0]) + Number(re[0])) / 2;
  const dx = Number(le[0]) - Number(re[0]);
  const dy = Number(le[1]) - Number(re[1]);
  const eyeDist = Math.sqrt(dx * dx + dy * dy);
  if (!eyeDist || eyeDist < 1e-6) return 0;
  return (Number(nose[0]) - midX) / eyeDist;
}

function _sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
