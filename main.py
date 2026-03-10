import io
import contextlib
import platform
import time
import cv2
import numpy as np

from config import (
    MATCH_THRESHOLDS,
    AUTO_UPDATE_ENABLED,
    AUTO_UPDATE_MIN_CONFIDENCE,
    AUTO_UPDATE_HIGH_CONFIDENCE,
    LIVE_CAPTURE_WIDTH,
    LIVE_CAPTURE_HEIGHT,
    LIVE_DETECT_EVERY_N_FRAMES,
    LIVE_MAX_DETECT_WIDTH,
    LIVENESS_ENABLED,
)
from modules.preprocessor import preprocess_image
from modules.detector     import detect_faces
from modules.aligner      import align_face, check_face_quality
from modules.liveness     import check_liveness
from modules.embedder     import generate_embedding, is_cvlface_loaded
from modules.database     import save_record, auto_update_add_template, current_tier_info
from modules.matcher      import match_face, is_match, match_result_label

from utils.visualizer import draw_detections, draw_face_count, draw_status_label


# ═══════════════════════════════════════════════
# LIVE CAMERA — AUTO CAPTURE
# ═══════════════════════════════════════════════
def run_live_camera(camera_index: int = 0):
    """
    Flow:
      1. Camera opens — live feed shown with face detection overlay.
      2. User presses SPACE to capture.
      3. Preview window is DESTROYED immediately to prevent UI hang.
      4. Frozen frame shown; terminal becomes active for user choice.
    """
    print("\n[LIVE] Camera ready.")
    print("[LIVE] SPACE = capture    Q = cancel\n")

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Discard dark warmup frames
    for _ in range(10):
        cap.read()
    time.sleep(0.3)

    frame_counter       = 0
    cached_preprocessed = None
    cached_display_faces = []
    cached_faces        = []
    cached_raw          = None
    captured_raw        = None
    captured_faces      = None
    window_preview      = "Live Camera | SPACE: capture Q: cancel"

    try:
        while True:
            ok, raw = cap.read()
            if not ok or raw is None:
                break

            frame_counter += 1
            if frame_counter % LIVE_DETECT_EVERY_N_FRAMES == 0 or cached_preprocessed is None:
                cached_raw = raw.copy()
                detect_frame, detect_scale = _resize_for_detection(cached_raw, LIVE_MAX_DETECT_WIDTH)
                cached_preprocessed = preprocess_image(detect_frame, fast=True)
                cached_display_faces = _silent_detect(cached_preprocessed)
                cached_faces = _scale_faces(cached_display_faces, 1.0 / detect_scale)

            display = cached_preprocessed.copy()
            if cached_display_faces:
                draw_detections(display, cached_display_faces)
            
            cv2.imshow(window_preview, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord(" "):
                if not cached_faces:
                    print("[LIVE] No face in frame.")
                    continue
                captured_raw   = cached_raw
                captured_faces = list(cached_faces)
                break
    finally:
        # Kill the high-frequency preview window BEFORE terminal input.
        # Keep cap open for active liveness challenge in register/match.
        cv2.destroyWindow(window_preview)

    if captured_faces:
        # Show frozen snapshot in a NEW window
        captured_preprocessed = preprocess_image(captured_raw.copy(), fast=False)
        snapshot = draw_detections(captured_preprocessed.copy(), captured_faces)
        draw_status_label(snapshot, "Captured! See terminal.", color=(0, 255, 255))
        
        cv2.imshow("Captured Face", snapshot)
        cv2.waitKey(1) # Force UI thread to render the window

        print(f"\n  Face captured ({len(captured_faces)} face(s) found).")
        print("  [r] Register    [m] Match    [q] Cancel")

        while True:
            action = input("  Choose: ").strip().lower()
            if action == "r":
                cv2.destroyAllWindows()
                _register(
                    captured_raw,
                    captured_preprocessed,
                    captured_faces,
                    camera_index=camera_index,
                    cap=cap,
                )
                break
            elif action == "m":
                cv2.destroyAllWindows()
                _match(
                    captured_raw,
                    captured_preprocessed,
                    captured_faces,
                    camera_index=camera_index,
                    cap=cap,
                )
                break
            elif action == "q":
                cv2.destroyAllWindows()
                break

    if cap is not None and cap.isOpened():
        cap.release()


# ═══════════════════════════════════════════════
# REGISTER
# ═══════════════════════════════════════════════

def _register(raw: np.ndarray, preprocessed: np.ndarray, faces: list, camera_index: int = 0, cap=None) -> None:
    faces = _quality_filter(raw, faces)
    if not faces:
        print("[REGISTER] No faces passed quality check.")
        return

    face = _pick_face(faces, preprocessed)
    if face is None:
        return

    name = input("\nEnter name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        return

    # ── Liveness check ───────────────────────────────────────────────────
    if LIVENESS_ENABLED:
        liveness = check_liveness(raw, face, camera_index=camera_index, cap=cap)
        if not liveness.get("passed", False):
            reason = liveness.get("reason", "Unknown liveness failure")
            print(f"[REGISTER] ✗ Liveness failed — {reason}")
            return
        method = liveness.get("method", "unknown")
        score = float(liveness.get("score", 0.0))
        print(f"[REGISTER] ✓ Liveness passed ({method}, "
              f"score={score:.3f})")

    aligned          = align_face(raw, face["landmarks"], bbox=face["bbox"])
    embedding, mode  = generate_embedding(aligned)

    if mode == "fallback":
        print("[REGISTER] ⚠ Fallback embedding — load AdaFace for reliable results.")

    save_record(name, face["bbox"], face["landmarks"], embedding, embedding_mode=mode)

    result = draw_detections(preprocessed.copy(), faces)
    draw_status_label(result, f"Registered: {name}", color=(0, 255, 255))
    cv2.imshow("Registered", result)
    _wait_for_window_close(timeout_ms=5000)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
# MATCH
# ═══════════════════════════════════════════════

def _match(raw: np.ndarray, preprocessed: np.ndarray, faces: list, camera_index: int = 0, cap=None) -> None:
    faces = _quality_filter(raw, faces)
    if not faces:
        print("[MATCH] No faces passed quality check.")
        return

    labels = {}
    print("\n[MATCH] Results:")
    for face in faces:
        # ── Liveness check ───────────────────────────────────────────────
        if LIVENESS_ENABLED:
            liveness = check_liveness(raw, face, camera_index=camera_index, cap=cap)
            if not liveness.get("passed", False):
                reason = liveness.get("reason", "Unknown liveness failure")
                print(f"  {face['face_id']} → LIVENESS FAIL | {reason}")
                labels[face["face_id"]] = "Spoof"
                continue
            method = liveness.get("method", "unknown")
            score = float(liveness.get("score", 0.0))
            print(f"  {face['face_id']} → liveness ✓ ({method}, "
                  f"score={score:.3f})")

        aligned          = align_face(raw, face["landmarks"], bbox=face["bbox"])
        embedding, mode  = generate_embedding(aligned)
        name, score      = match_face(embedding, query_mode=mode)
        labels[face["face_id"]] = match_result_label(name, score)
        status = "MATCHED" if is_match(name) else "NO MATCH"
        print(f"  {face['face_id']} → {status} | {name} ({score:.3f}) [mode={mode}]")
        if is_match(name):
            _auto_update(name, score, mode, face, embedding)

    result = draw_detections(preprocessed.copy(), faces, labels=labels)
    draw_face_count(result, len(faces))
    draw_status_label(result, "Match complete", color=(0, 255, 255))
    cv2.imshow("Match Result", result)
    _wait_for_window_close(timeout_ms=5000)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def _silent_detect(image: np.ndarray) -> list:
    with contextlib.redirect_stdout(io.StringIO()):
        return detect_faces(image)


def _wait_for_window_close(timeout_ms: int = 5000) -> None:
    """
    Keep the result window responsive and auto-close after timeout.
    Any key press exits early.
    """
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        if cv2.waitKey(50) != -1:
            break


def _resize_for_detection(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Resize frame for faster detection, returning (resized, scale_from_raw)."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image, 1.0
    scale = max_width / float(w)
    resized = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _scale_faces(faces: list[dict], scale: float) -> list[dict]:
    """Scale face bbox/landmarks by factor (used to map detect coords back to raw coords)."""
    if scale == 1.0:
        return faces
    scaled = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        landmarks = {
            k: [float(v[0]) * scale, float(v[1]) * scale]
            for k, v in face["landmarks"].items()
        }
        scaled.append(
            {
                "face_id": face["face_id"],
                "score": face["score"],
                "bbox": [
                    int(round(x1 * scale)),
                    int(round(y1 * scale)),
                    int(round(x2 * scale)),
                    int(round(y2 * scale)),
                ],
                "landmarks": landmarks,
            }
        )
    return scaled


def _quality_filter(raw: np.ndarray, faces: list) -> list:
    passed = []
    for face in faces:
        q = check_face_quality(raw, face)
        if q["passed"]:
            passed.append(face)
        else:
            print(f"[QUALITY] ✗ {face['face_id']} rejected — {q['reason']}")
    return passed


def _pick_face(faces: list, display_image: np.ndarray) -> dict | None:
    if len(faces) == 1:
        return faces[0]

    annotated = display_image.copy()
    for i, face in enumerate(faces, 1):
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, str(i), (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Select Face", annotated)
    cv2.waitKey(400)

    print(f"\n  {len(faces)} faces found:")
    for i, face in enumerate(faces, 1):
        print(f"  [{i}] {face['face_id']}  conf={face['score']:.3f}")

    while True:
        try:
            choice = int(input(f"  Select [1-{len(faces)}]: ").strip())
            if 1 <= choice <= len(faces):
                return faces[choice - 1]
        except ValueError:
            pass
        print("  Invalid input.")


def _auto_update(name: str, score: float, mode: str, face: dict, embedding: list) -> None:
    if not AUTO_UPDATE_ENABLED:
        return

    threshold = MATCH_THRESHOLDS.get(mode, MATCH_THRESHOLDS.get("cvlface", 0.40))
    min_auto  = AUTO_UPDATE_MIN_CONFIDENCE.get(mode, threshold)
    high_conf = AUTO_UPDATE_HIGH_CONFIDENCE.get(mode, max(min_auto, threshold))
    if score < min_auto or score >= high_conf:
        return

    # Atomic path in DB layer: decision + write happen under one per-person lock.
    add, reason = auto_update_add_template(
        name=name,
        mode=mode,
        embedding=embedding,
        bbox=face["bbox"],
        landmarks=face["landmarks"],
    )
    if add:
        tier = current_tier_info()
        print(f"[AUTO-UPDATE] '{name}' — {reason}  "
              f"[tier={tier['strategy']}, score={score:.3f}]")
    else:
        print(f"[AUTO-UPDATE] '{name}' skipped — {reason}")


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    mode_label = "CVLFace ✓" if is_cvlface_loaded() else "Fallback ⚠"
    print(f"\n  Face Pipeline  |  Embedding: {mode_label}")

    while True:
        print("\n  [1] Start camera")
        print("  [q] Quit")
        choice = input("  Choose: ").strip().lower()

        if choice == "1":
            run_live_camera()
        elif choice == "q":
            print("Bye!")
            break
        else:
            print("  Enter 1 or q.")
