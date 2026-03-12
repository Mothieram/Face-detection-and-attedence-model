const API =
  window.location.protocol === "file:"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname || "localhost"}:8000`;
const API_KEY = "developer_key";
let stream = null;
let detectTimer = null;
let detectBusy = false;
const captureCanvas = document.createElement("canvas");
const ACTIVE_CALIB_TARGET = 5;
const ACTIVE_CALIB_MIN = 3;
const ACTIVE_TURN_THRESHOLD = 0.3;
const ACTIVE_TIMEOUT_MS = 12000;

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-API-Key", API_KEY);
  return fetch(`${API}${path}`, { ...options, headers });
}

function clearOverlay() {
  const cvs = document.getElementById("cvs");
  const ctx = cvs.getContext("2d");
  ctx.clearRect(0, 0, cvs.width || 1, cvs.height || 1);
  updateLiveEligibility([]);
}

function updateLiveEligibility(faces) {
  const w = document.getElementById("cam-warning");
  const btnSnap = document.getElementById("btn-snap");
  if (!w || !btnSnap) return;

  const hasStream = !!stream;
  const list = Array.isArray(faces) ? faces : [];
  const count = list.length;
  const tooMany = count > 1;
  const oneFace = count === 1;
  const qualityOk = oneFace && !!(list[0].quality && list[0].quality.passed);

  // Warning banner is only for the "too many faces" case.
  w.classList.toggle("hidden", !tooMany);

  // Capture is allowed only when all live criteria are satisfied.
  // Current criteria: stream on + exactly one face + quality passed.
  const canCapture = hasStream && oneFace && qualityOk;
  btnSnap.disabled = !canCapture;
}

function drawLiveBBoxes(faces) {
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

function startLiveDetect() {
  if (detectTimer) return;
  detectTimer = setInterval(tickLiveDetect, 700);
}

function stopLiveDetect() {
  if (detectTimer) {
    clearInterval(detectTimer);
    detectTimer = null;
  }
  clearOverlay();
}

function setChallengeUI(text, percent, warn, show) {
  const box = document.getElementById("challenge-box");
  const t = document.getElementById("challenge-text");
  const f = document.getElementById("challenge-fill");
  if (!box || !t || !f) return;
  t.textContent = text;
  f.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  box.classList.toggle("warn", !!warn);
  box.classList.toggle("hidden", !show);
}

// ── Toast ──────────────────────────────────────────────
let _tt;
function toast(msg, err = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "show" + (err ? " err" : "");
  clearTimeout(_tt);
  _tt = setTimeout(() => (el.className = ""), 3000);
}

// ── Status bar ─────────────────────────────────────────
async function loadStatus() {
  try {
    const resp = await apiFetch("/status");
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const d = await resp.json();
    setPill("p-server", "Server OK", true);
    setPill(
      "p-cvl",
      d.cvlface_loaded ? "CVLFace ✓" : "CVLFace ✗",
      d.cvlface_loaded,
    );
    setPill(
      "p-live",
      d.liveness_enabled ? "Liveness ✓" : "Liveness off",
      d.liveness_enabled,
    );
    const t = d.db_tier;
    setPill("p-tier", `${t.strategy} · m=${t.hnsw_m}`, true);
  } catch {
    setPill("p-server", "Server offline", false);
  }
}

function setPill(id, text, ok) {
  const el = document.getElementById(id);
  el.innerHTML = `<span class="dot"></span>${text}`;
  el.className = "pill " + (ok ? "ok" : "warn");
}

if (window.location.protocol === "file:") {
  toast(
    "Open this page via http://127.0.0.1:5500 (not file://) to avoid CORS.",
    true,
  );
}

loadStatus();
setInterval(loadStatus, 30000);

// ── Camera ─────────────────────────────────────────────
async function toggleCamera() {
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
  } catch (e) {
    toast("Camera error: " + e.message, true);
  }
}

// ── Capture helpers ────────────────────────────────────
function captureFrameBlob(quality = 0.92) {
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

async function detectLargestFaceOnce() {
  const blob = await captureFrameBlob(0.75);
  if (!blob) return null;
  const fd = new FormData();
  fd.append("image", blob, "active.jpg");
  const r = await apiFetch("/detect", { method: "POST", body: fd });
  if (!r.ok) return null;
  const data = await r.json();
  const faces = Array.isArray(data.faces) ? data.faces : [];
  if (!faces.length) return null;
  return faces.sort((a, b) => (b.score || 0) - (a.score || 0))[0];
}

async function runPassiveLivenessCheck(blob) {
  const fd = new FormData();
  fd.append("image", blob, "liveness.jpg");
  fd.append("passive_only", "true");
  fd.append("camera_index", "0");
  const r = await apiFetch("/liveness", { method: "POST", body: fd });
  const payload = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(payload.detail || `${r.status}`);
  return payload;
}

function estimateYawFromLandmarks(landmarks) {
  if (
    !landmarks ||
    !landmarks.left_eye ||
    !landmarks.right_eye ||
    !landmarks.nose
  )
    return 0;
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

async function runClientActiveChallenge() {
  stopLiveDetect();
  toast("Active challenge: keep face straight...", false);
  setChallengeUI("Calibration: hold face straight", 0, false, true);

  const neutralSamples = [];
  const calibDeadline = Date.now() + 3000;
  while (
    Date.now() < calibDeadline &&
    neutralSamples.length < ACTIVE_CALIB_TARGET
  ) {
    const face = await detectLargestFaceOnce();
    if (face && face.landmarks)
      neutralSamples.push(estimateYawFromLandmarks(face.landmarks));
    const p = Math.round((neutralSamples.length / ACTIVE_CALIB_TARGET) * 100);
    setChallengeUI(
      `Calibration: ${Math.min(neutralSamples.length, ACTIVE_CALIB_TARGET)}/${ACTIVE_CALIB_TARGET}`,
      p,
      false,
      true,
    );
    await new Promise((r) => setTimeout(r, 220));
  }
  if (neutralSamples.length < ACTIVE_CALIB_MIN) {
    if (stream) startLiveDetect();
    setChallengeUI("Calibration failed", 100, true, true);
    return { passed: false, reason: "Face not stable for calibration" };
  }

  const sorted = [...neutralSamples].sort((a, b) => a - b);
  const neutralYaw = sorted[Math.floor(sorted.length / 2)];
  toast("Turn your head LEFT or RIGHT now", false);
  setChallengeUI("Turn head LEFT or RIGHT", 0, false, true);

  const deadline = Date.now() + ACTIVE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const face = await detectLargestFaceOnce();
    const remaining = Math.max(0, deadline - Date.now());
    const secs = (remaining / 1000).toFixed(1);
    if (face && face.landmarks) {
      const yaw = estimateYawFromLandmarks(face.landmarks) - neutralYaw;
      const turnProgress = Math.min(
        100,
        Math.round((Math.abs(yaw) / ACTIVE_TURN_THRESHOLD) * 100),
      );
      setChallengeUI(
        `Turn head LEFT or RIGHT (${secs}s)`,
        turnProgress,
        false,
        true,
      );
      if (Math.abs(yaw) >= ACTIVE_TURN_THRESHOLD) {
        if (stream) startLiveDetect();
        setChallengeUI("Active challenge passed", 100, false, true);
        return { passed: true, reason: "Active challenge passed" };
      }
    } else {
      setChallengeUI(`Turn head LEFT or RIGHT (${secs}s)`, 0, false, true);
    }
    await new Promise((r) => setTimeout(r, 220));
  }

  if (stream) startLiveDetect();
  setChallengeUI("Head turn not detected", 100, true, true);
  return { passed: false, reason: "Head turn not detected in time" };
}

async function captureAndMatch() {
  if (!stream) {
    toast("Start camera first", true);
    return;
  }

  const flash = document.getElementById("flash");
  const passiveOnly = document.getElementById("opt-passive").checked;

  flash.classList.add("go");
  setTimeout(() => flash.classList.remove("go"), 110);

  if (!passiveOnly) {
    const preBlob = await captureFrameBlob(0.88);
    if (!preBlob) {
      toast("Capture failed", true);
      return;
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
    } else if (reason.includes("spoof")) {
      document.getElementById("results").innerHTML =
        `<div class="empty" style="color:var(--warn)">Passive spoof detected (${Number(passive.score || 0).toFixed(2)})</div>`;
      toast("Passive spoof detected", true);
      return;
    } else {
      const active = await runClientActiveChallenge();
      if (!active.passed) {
        document.getElementById("results").innerHTML =
          `<div class="empty" style="color:var(--warn)">Active challenge failed: ${active.reason}</div>`;
        toast(`Active challenge failed: ${active.reason}`, true);
        setTimeout(
          () => setChallengeUI("Active challenge idle", 0, false, false),
          1800,
        );
        return;
      }
      toast("Active challenge passed", false);
    }
  }

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

  try {
    const r = await apiFetch("/match", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = `${r.status}`;
      try {
        const e = await r.json();
        detail = e.detail || detail;
      } catch {}
      throw new Error(detail);
    }
    renderResults(await r.json());
    if (!passiveOnly)
      setTimeout(
        () => setChallengeUI("Active challenge idle", 0, false, false),
        1200,
      );
  } catch (e) {
    document.getElementById("results").innerHTML =
      `<div class="empty" style="color:var(--warn)">Error: ${e.message}</div>`;
    toast("Match failed: " + e.message, true);
    if (!passiveOnly)
      setTimeout(
        () => setChallengeUI("Active challenge idle", 0, false, false),
        1200,
      );
  }
}

async function captureAndRegister() {
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

  if (!passiveOnly) {
    const preBlob = await captureFrameBlob(0.88);
    if (!preBlob) {
      toast("Capture failed", true);
      return;
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
    } else if (reason.includes("spoof")) {
      document.getElementById("results").innerHTML =
        `<div class="empty" style="color:var(--warn)">Passive spoof detected (${Number(passive.score || 0).toFixed(2)})</div>`;
      toast("Passive spoof detected", true);
      return;
    } else {
      const active = await runClientActiveChallenge();
      if (!active.passed) {
        document.getElementById("results").innerHTML =
          `<div class="empty" style="color:var(--warn)">Active challenge failed: ${active.reason}</div>`;
        toast(`Active challenge failed: ${active.reason}`, true);
        setTimeout(
          () => setChallengeUI("Active challenge idle", 0, false, false),
          1800,
        );
        return;
      }
      toast("Active challenge passed", false);
    }
  }

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
            </div>`;
    toast(`Registered: ${payload.name || name}`);
    if (!passiveOnly)
      setTimeout(
        () => setChallengeUI("Active challenge idle", 0, false, false),
        1200,
      );
  } catch (e) {
    document.getElementById("results").innerHTML =
      `<div class="empty" style="color:var(--warn)">Register failed: ${e.message}</div>`;
    toast("Register failed: " + e.message, true);
    if (!passiveOnly)
      setTimeout(
        () => setChallengeUI("Active challenge idle", 0, false, false),
        1200,
      );
  }
}

// ── Render cards ───────────────────────────────────────
function renderResults(data) {
  const box = document.getElementById("results");

  if (!data.results || data.results.length === 0) {
    box.innerHTML =
      '<div class="empty"><span class="icon">🔍</span>No faces detected or all failed quality check.</div>';
    return;
  }

  const fmtBbox = (bbox) => {
    if (!Array.isArray(bbox) || bbox.length < 4) return null;
    return `bbox [${bbox
      .slice(0, 4)
      .map((v) => Math.round(Number(v) || 0))
      .join(", ")}]`;
  };

  box.innerHTML = data.results
    .map((res) => {
      const isSpoof = res.name === "Spoof";
      const isUncertain = res.name === "Uncertain";
      const isRejected = res.name === "Rejected";
      const cls = res.matched
        ? "matched"
        : isSpoof
          ? "spoof"
          : isUncertain
            ? "uncertain"
            : isRejected
              ? "spoof"
              : "";
      const emoji = res.matched
        ? "👤"
        : isSpoof
          ? "⚠️"
          : isUncertain
            ? "🟡"
            : isRejected
              ? "🚫"
              : "❓";
      const sc = res.score > 0.65 ? "high" : "low";
      const lv = res.liveness;
      const q = res.quality;
      const meta = [
        res.face_id,
        fmtBbox(res.bbox),
        q ? `quality ${q.passed ? "ok" : "fail"}` : null,
        q && q.reason ? q.reason : null,
        lv && lv.method ? `method ${lv.method}` : null,
        lv ? `liveness ${lv.score.toFixed(2)}` : null,
        lv && lv.reason ? lv.reason : null,
        res.auto_updated ? "updated" : null,
        res.embedding_mode || null,
      ]
        .filter(Boolean)
        .join(" · ");

      return `
  <div class="card ${cls}">
    <div class="avatar ${cls}">${emoji}</div>
    <div class="card-info">
      <div class="card-name">${res.name}</div>
      <div class="card-meta">${meta}</div>
    </div>
    <div class="score ${sc}">${res.score.toFixed(3)}</div>
  </div>`;
    })
    .join("");

  toast(`${data.matched_count} of ${data.total_faces} face(s) matched`);
}
