// ── map.js ────────────────────────────────────────────
// Renders an OpenStreetMap tile thumbnail onto a <canvas> element.
// Uses tile.openstreetmap.org which allows cross-origin requests.

function latLonToPixel(lat, lon, z) {
  const n = Math.pow(2, z);
  const px = ((lon + 180) / 360) * n * 256;
  const latRad = (lat * Math.PI) / 180;
  const py =
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) *
    n *
    256;
  return { px, py };
}

function drawPin(ctx, cx, cy) {
  // Drop shadow
  ctx.beginPath();
  ctx.arc(cx, cy + 16, 5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.3)";
  ctx.fill();
  // Circle head
  ctx.beginPath();
  ctx.arc(cx, cy - 6, 9, 0, Math.PI * 2);
  ctx.fillStyle = "#ff3b5c";
  ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.stroke();
  // Stem
  ctx.beginPath();
  ctx.moveTo(cx, cy + 3);
  ctx.lineTo(cx, cy + 15);
  ctx.strokeStyle = "#ff3b5c";
  ctx.lineWidth = 2.5;
  ctx.stroke();
  // White inner dot
  ctx.beginPath();
  ctx.arc(cx, cy - 6, 3, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fill();
}

export function renderMapCanvas(canvasId, lat, lon, zoom) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width; // 100
  const H = canvas.height; // 90
  const tileSize = 256;
  const n = Math.pow(2, zoom);

  // World-pixel coords of the center point
  const center = latLonToPixel(lat, lon, zoom);
  const originX = center.px - W / 2;
  const originY = center.py - H / 2;

  // Tile range needed to cover the canvas
  const tileX0 = Math.floor(originX / tileSize);
  const tileY0 = Math.floor(originY / tileSize);
  const tileX1 = Math.floor((originX + W) / tileSize);
  const tileY1 = Math.floor((originY + H) / tileSize);

  // Dark placeholder while tiles load
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);

  let tilesTotal = 0;
  let tilesLoaded = 0;

  const onTileDone = () => {
    tilesLoaded++;
    if (tilesLoaded === tilesTotal) drawPin(ctx, W / 2, H / 2);
  };

  for (let tx = tileX0; tx <= tileX1; tx++) {
    for (let ty = tileY0; ty <= tileY1; ty++) {
      if (ty < 0 || ty >= n) continue;
      const wrappedTx = ((tx % n) + n) % n;
      tilesTotal++;

      const dx = tx * tileSize - originX;
      const dy = ty * tileSize - originY;
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = `https://tile.openstreetmap.org/${zoom}/${wrappedTx}/${ty}.png`;
      img.onload = () => {
        ctx.drawImage(img, dx, dy, tileSize, tileSize);
        onTileDone();
      };
      img.onerror = onTileDone;
    }
  }

  if (tilesTotal === 0) drawPin(ctx, W / 2, H / 2);
}
