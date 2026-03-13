// ── toast.js ──────────────────────────────────────────
// Lightweight toast notification — shows a message for 3 s then fades.

let _timer;

export function toast(msg, err = false) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.className = "show" + (err ? " err" : "");
  clearTimeout(_timer);
  _timer = setTimeout(() => (el.className = ""), 3000);
}
