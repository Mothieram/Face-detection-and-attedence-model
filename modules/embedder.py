"""
modules/embedder.py
────────────────────
STEP 4 + 5 — Face Embedding via CVLFace

Uses CVLFace's Hugging Face AutoModel API.
Model downloads automatically on first run and caches locally.

Recommended model:
  minchul/cvlface_adaface_ir101_webface12m

Usage:
  embedding, mode = generate_embedding(aligned_face_bgr)
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
    """
    model = _load_cvlface()
    if model is not None:
        return _cvlface_embedding(aligned_face, model), "cvlface"

    print("[EMBED] CVLFace unavailable — using fallback embedding.")
    print("[EMBED] [WARN] Install: transformers huggingface_hub omegaconf hydra-core timm fvcore iopath yacs")
    return _enhanced_embedding(aligned_face), "fallback"


def is_cvlface_loaded() -> bool:
    return _cvlface_model is not None


# ═══════════════════════════════════════════════
# CVLFace loader
# ═══════════════════════════════════════════════

def _load_cvlface():
    global _cvlface_model, _cvlface_device

    if _cvlface_model is not None:
        return _cvlface_model

    with _model_lock:
        # Re-check inside lock in case another thread loaded it while we waited
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

            # Suppress hundreds of "copying from non-meta parameter" UserWarnings
            # that CVLFace's checkpoint loader emits during weight loading.
            # These are harmless — the model loads correctly regardless.
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
            _cvlface_device = device   # cache here — used in every _cvlface_embedding call
            print(f"[EMBED] CVLFace loaded on {device} [OK]")
            return model

        except ImportError as e:
            print(f"[EMBED] Missing dependency: {e}")
            return None
        except Exception as e:
            print(f"[EMBED] Failed to load CVLFace: {e}")
            return None


# ═══════════════════════════════════════════════
# CVLFace embedding
# ═══════════════════════════════════════════════

def _cvlface_embedding(face_bgr: np.ndarray, model) -> list:
    face_norm = preprocess_face(face_bgr)
    tensor    = torch.from_numpy(face_norm).permute(2, 0, 1).unsqueeze(0)
    # Use cached device — never call next(model.parameters()) per inference
    tensor    = tensor.to(_cvlface_device)

    with torch.no_grad():
        output = model(tensor)
        embedding = output[0] if isinstance(output, (tuple, list)) else output
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze().cpu().numpy().tolist()


# ═══════════════════════════════════════════════
# Enhanced fallback (no model available)
# ═══════════════════════════════════════════════

def _enhanced_embedding(face_112: np.ndarray) -> list:
    """512-d handcrafted embedding — placeholder only, low accuracy."""
    face     = cv2.resize(face_112, (112, 112)).astype(np.float32) / 255.0
    features = []

    # Multi-scale channel statistics
    for scale in [112, 56, 28]:
        r = cv2.resize(face, (scale, scale))
        for c in range(3):
            ch = r[:, :, c]
            features += [float(np.mean(ch)), float(np.std(ch))]

    # 8×8 spatial grid channel means
    bh, bw = face.shape[0] // 8, face.shape[1] // 8
    for i in range(8):
        for j in range(8):
            block = face[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            for c in range(3):
                features.append(float(np.mean(block[:, :, c])))

    # Sobel gradient 8×8 grid
    gray   = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    grad   = np.sqrt(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                     cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    gh, gw = grad.shape[0] // 8, grad.shape[1] // 8
    for i in range(8):
        for j in range(8):
            features.append(float(np.mean(grad[i*gh:(i+1)*gh, j*gw:(j+1)*gw])))

    # LBP texture — single pass over three 56×56 regions
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
