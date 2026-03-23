"""
modules/embedder.py
────────────────────
STEP 4 + 5 — Face Embedding via CVLFace

Uses CVLFace's Hugging Face AutoModel API.
Model downloads automatically on first run and caches locally.

Recommended model:
  minchul/cvlface_adaface_ir101_webface12m

Usage (single — unchanged API):
  embedding, mode = generate_embedding(aligned_face_bgr)

Usage (batch — new, for multi-face images):
  embeddings, mode = generate_embeddings_batch([crop1, crop2, crop3])
  # Returns list of 512-d lists, same order as input. Single forward pass.

FIX (bottleneck #2 — batch embedding):
  Added generate_embeddings_batch() that stacks N aligned crops into one
  tensor and runs a single model forward pass, replacing the N-loop pattern
  used in the match and attendance endpoints.
  generate_embedding() is unchanged — it now delegates to the batch function
  with N=1, so all existing callers continue to work without modification.
"""

import os
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import INPUT_SIZE, CVLFACE_REPO_ID, CVLFACE_CACHE_DIR, EMBEDDING_DIM
from modules.preprocessor import preprocess_face

_cvlface_model  = None
_cvlface_device = None   # cached at load time — avoids next(model.parameters()) per inference
_model_lock     = threading.Lock()  # prevents double-load in concurrent contexts


# ═══════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════

def generate_embedding(aligned_face: np.ndarray) -> tuple[list, str]:
    """
    Generate a 512-d embedding from an aligned 112×112 BGR face crop.

    Returns:
        (embedding, mode)
        embedding : list of 512 floats (L2-normalized)
        mode      : 'cvlface' | 'fallback'

    Unchanged API — delegates to generate_embeddings_batch internally.
    """
    embeddings, mode = generate_embeddings_batch([aligned_face])
    return embeddings[0], mode


def generate_embeddings_batch(
    aligned_faces: list[np.ndarray],
) -> tuple[list[list[float]], str]:
    """
    Generate 512-d embeddings for a batch of aligned BGR face crops in a
    SINGLE model forward pass.

    Parameters
    ----------
    aligned_faces : list of BGR ndarrays, each already aligned / 112×112.

    Returns
    -------
    embeddings : list of 512-d float lists, same order as input.
    mode       : 'cvlface' | 'fallback'

    Example (in match / attendance endpoint):
        aligned_crops = [align_face(raw, f["landmarks"], bbox=f["bbox"])
                         for f in live_faces]
        batch_embs, mode = generate_embeddings_batch(aligned_crops)
        for face, embedding in zip(live_faces, batch_embs):
            person_ref, score = match_face(embedding, query_mode=mode)
    """
    if not aligned_faces:
        return [], "cvlface"

    model = _load_cvlface()

    if model is not None:
        return _cvlface_embedding_batch(aligned_faces, model), "cvlface"

    print("[EMBED] CVLFace unavailable — using fallback embedding.")
    print("[EMBED] [WARN] Install: transformers huggingface_hub omegaconf hydra-core timm fvcore iopath yacs")
    return [_enhanced_embedding(face) for face in aligned_faces], "fallback"


def is_cvlface_loaded() -> bool:
    return _cvlface_model is not None


# ═══════════════════════════════════════════════
# CVLFace loader  (unchanged)
# ═══════════════════════════════════════════════

def _load_cvlface():
    global _cvlface_model, _cvlface_device

    if _cvlface_model is not None:
        return _cvlface_model

    with _model_lock:
        if _cvlface_model is not None:
            return _cvlface_model

        try:
            from transformers import AutoModel
            from huggingface_hub import hf_hub_download

            cache_path = os.path.expanduser(CVLFACE_CACHE_DIR)
            os.makedirs(cache_path, exist_ok=True)

            files_txt = os.path.join(cache_path, "files.txt")
            if not os.path.exists(files_txt):
                print("[EMBED] First run — downloading model files...")
                hf_hub_download(CVLFACE_REPO_ID, "files.txt",
                                local_dir=cache_path, local_dir_use_symlinks=False)

            with open(files_txt, "r") as f:
                extra_files = [line.strip() for line in f if line.strip()]

            for fname in extra_files + ["config.json", "wrapper.py", "model.safetensors"]:
                fpath = os.path.join(cache_path, fname)
                if not os.path.exists(fpath):
                    hf_hub_download(CVLFACE_REPO_ID, fname,
                                    local_dir=cache_path, local_dir_use_symlinks=False)

            import sys
            cwd = os.getcwd()
            os.chdir(cache_path)
            sys.path.insert(0, cache_path)

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*copying from a non-meta parameter.*",
                    category=UserWarning,
                )
                model = AutoModel.from_pretrained(cache_path, trust_remote_code=True)
            os.chdir(cwd)

            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = model.to(device)

            _cvlface_model  = model
            _cvlface_device = device
            print(f"[EMBED] CVLFace loaded on {device} [OK]")
            return model

        except ImportError as e:
            print(f"[EMBED] Missing dependency: {e}")
            return None
        except Exception as e:
            print(f"[EMBED] Failed to load CVLFace: {e}")
            return None


# ═══════════════════════════════════════════════
# CVLFace embedding — batch (new core function)
# ═══════════════════════════════════════════════

def _cvlface_embedding_batch(faces_bgr: list[np.ndarray], model) -> list[list[float]]:
    """
    Run model inference once for all faces in the list.

    Each face is preprocessed with preprocess_face() (same as the original
    single-face path), then stacked into a (N, 3, H, W) batch tensor.
    The result is L2-normalised per row and returned as a plain list of lists.
    """
    # Build individual tensors using the same preprocess_face() call the
    # original single-face code used — no change to preprocessing logic.
    tensors = []
    for face_bgr in faces_bgr:
        face_norm = preprocess_face(face_bgr)                          # HWC float32
        t = torch.from_numpy(face_norm).permute(2, 0, 1)              # CHW
        tensors.append(t)

    batch = torch.stack(tensors).to(_cvlface_device)                  # (N, C, H, W)

    with torch.no_grad():
        output = model(batch)
        embeddings = output[0] if isinstance(output, (tuple, list)) else output
        embeddings = F.normalize(embeddings, p=2, dim=1)               # L2 per row

    return embeddings.cpu().numpy().tolist()                           # list of N lists


# ═══════════════════════════════════════════════
# Enhanced fallback  (unchanged)
# ═══════════════════════════════════════════════

def _enhanced_embedding(face_112: np.ndarray) -> list:
    """512-d handcrafted embedding — placeholder only, low accuracy."""
    face     = cv2.resize(face_112, (112, 112)).astype(np.float32) / 255.0
    features = []

    for scale in [112, 56, 28]:
        r = cv2.resize(face, (scale, scale))
        for c in range(3):
            ch = r[:, :, c]
            features += [float(np.mean(ch)), float(np.std(ch))]

    bh, bw = face.shape[0] // 8, face.shape[1] // 8
    for i in range(8):
        for j in range(8):
            block = face[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            for c in range(3):
                features.append(float(np.mean(block[:, :, c])))

    gray   = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    grad   = np.sqrt(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                     cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    gh, gw = grad.shape[0] // 8, grad.shape[1] // 8
    for i in range(8):
        for j in range(8):
            features.append(float(np.mean(grad[i*gh:(i+1)*gh, j*gw:(j+1)*gw])))

    for ry, rx in [(0, 0), (0, 56), (56, 0)]:
        hist = _lbp_hist(gray[ry:ry+56, rx:rx+56])
        features.extend(hist)

    features = (features + [0.0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
    vec      = np.array(features, dtype=np.float32)
    norm     = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def _lbp_hist(patch: np.ndarray) -> list:
    """LBP histogram for a grayscale patch, returned as normalized float list."""
    center = patch[1:-1, 1:-1]
    lbp    = np.zeros_like(center, dtype=np.uint8)
    for i, nb in enumerate([
        patch[0:-2, 0:-2], patch[0:-2, 1:-1], patch[0:-2, 2:],
        patch[1:-1, 2:],   patch[2:,   2:],   patch[2:,   1:-1],
        patch[2:,   0:-2], patch[1:-1, 0:-2],
    ]):
        lbp += ((nb >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(lbp, bins=58, range=(0, 58))
    hist    = hist.astype(np.float32)
    s       = np.sum(hist)
    return (hist / s if s > 0 else hist).tolist()