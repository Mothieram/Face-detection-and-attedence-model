// ── actions.js ────────────────────────────────────────
// High-level button actions: capture + match, capture + register.

import { apiFetch } from "./api.js";
import { toast } from "./toast.js";
import { stream, captureFrameBlob } from "./camera.js";
import {
  runPassiveLivenessCheck,
  runClientActiveChallenge,
  setChallengeUI,
} from "./liveness.js";
import { geoCoords } from "./geo.js";
import { renderResults } from "./result.js";
import { buildGeoCard } from "./geocard.js";

// ── Shared liveness pre-flight ────────────────────────
async function livenessPreFlight(passiveOnly) {
  if (passiveOnly) return true; // skip

  const preBlob = await captureFrameBlob(0.88);
  if (!preBlob) {
    toast("Capture failed", true);
    return false;
  }

  const passive = await runPassiveLivenessCheck(preBlob).catch((e) => ({
    passed: false,
    reason: String(e),
  }));
  const reason = String(passive.reason || "").toLowerCase();

  if (passive.passed) {
    toast(
      `Passive real (${Number(passive.score || 0).toFixed(2)}), skipping active`,
      false,
    );
    return true;
  }

  if (reason.includes("spoof")) {
    document.getElementById("results").innerHTML =
      `<div class="empty" style="color:var(--warn)">Passive spoof detected (${Number(passive.score || 0).toFixed(2)})</div>`;
    toast("Passive spoof detected", true);
    return false;
  }

  // Uncertain — run active challenge
  const active = await runClientActiveChallenge();
  if (!active.passed) {
    document.getElementById("results").innerHTML =
      `<div class="empty" style="color:var(--warn)">Active challenge failed: ${active.reason}</div>`;
    toast(`Active challenge failed: ${active.reason}`, true);
    setTimeout(
      () => setChallengeUI("Active challenge idle", 0, false, false),
      1800,
    );
    return false;
  }

  toast("Active challenge passed", false);
  return true;
}

function appendGeoToForm(fd) {
  if (geoCoords) {
    fd.append("lat", geoCoords.lat);
    fd.append("lon", geoCoords.lon);
    if (geoCoords.accuracy_m != null)
      fd.append("accuracy_m", geoCoords.accuracy_m);
  }
}

// ── Capture & Match ───────────────────────────────────
export async function captureAndMatch() {
  if (!stream) {
    toast("Start camera first", true);
    return;
  }

  const flash = document.getElementById("flash");
  const passiveOnly = document.getElementById("opt-passive").checked;

  flash.classList.add("go");
  setTimeout(() => flash.classList.remove("go"), 110);

  const ok = await livenessPreFlight(passiveOnly);
  if (!ok) return;

  document.getElementById("results").innerHTML =
    '<div class="empty"><span class="spinner"></span></div>';

  const blob = await captureFrameBlob(0.92);
  if (!blob) {
    toast("Capture failed", true);
    return;
  }

  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  fd.append("passive_only", "true");
  fd.append("skip_liveness", passiveOnly ? "false" : "true");
  fd.append("auto_update", document.getElementById("opt-autoupdate").checked);
  appendGeoToForm(fd);

  try {
    const r = await apiFetch("/match", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = `${r.status}`;
      try {
        detail = (await r.json()).detail || detail;
      } catch {}
      throw new Error(detail);
    }
    renderResults(await r.json());
  } catch (e) {
    document.getElementById("results").innerHTML =
      `<div class="empty" style="color:var(--warn)">Error: ${e.message}</div>`;
    toast("Match failed: " + e.message, true);
  } finally {
    if (!passiveOnly)
      setTimeout(
        () => setChallengeUI("Active challenge idle", 0, false, false),
        1200,
      );
  }
}

// ── Capture & Register ────────────────────────────────
export async function captureAndRegister() {
  if (!stream) {
    toast("Start camera first", true);
    return;
  }

  const name = (document.getElementById("reg-name").value || "").trim();
  if (!name) {
    toast("Enter register name", true);
    return;
  }

  const flash = document.getElementById("flash");
  const passiveOnly = document.getElementById("opt-passive").checked;

  flash.classList.add("go");
  setTimeout(() => flash.classList.remove("go"), 110);

  const ok = await livenessPreFlight(passiveOnly);
  if (!ok) return;

  document.getElementById("results").innerHTML =
    '<div class="empty"><span class="spinner"></span></div>';

  const blob = await captureFrameBlob(0.92);
  if (!blob) {
    toast("Capture failed", true);
    return;
  }

  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  fd.append("name", name);
  fd.append("face_index", "0");
  fd.append("skip_liveness", passiveOnly ? "false" : "true");
  fd.append("passive_only", "true");
  fd.append("camera_index", "0");
  appendGeoToForm(fd);

  try {
    const r = await apiFetch("/register", { method: "POST", body: fd });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok)
      throw new Error(payload.detail || payload.message || `${r.status}`);

    document.getElementById("results").innerHTML = `
      <div class="card matched">
        <div class="avatar matched">✅</div>
        <div class="card-info">
          <div class="card-name">${payload.name || name}</div>
          <div class="card-meta">registered · ${payload.embedding_mode || "unknown mode"}</div>
        </div>
        <div class="score high">OK</div>
      </div>
      ${buildGeoCard(payload.geo || null)}`;
    toast(`Registered: ${payload.name || name}`);
  } catch (e) {
    document.getElementById("results").innerHTML =
      `<div class="empty" style="color:var(--warn)">Register failed: ${e.message}</div>`;
    toast("Register failed: " + e.message, true);
  } finally {
    if (!passiveOnly)
      setTimeout(
        () => setChallengeUI("Active challenge idle", 0, false, false),
        1200,
      );
  }
}
