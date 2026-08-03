"""
BSU Personnel Monitoring - Facial Recognition Service (Multi-Camera Edition)
=============================================================================
Optimised for: NVIDIA RTX 2050 (4 GB) · AMD Ryzen 5 6600H · 8 GB RAM

Setup:
  pip install opencv-python ultralytics facenet-pytorch deep-sort-realtime \
              requests numpy scipy python-dotenv

  # CUDA 12.4 wheels (Python 3.10 / 3.13 compatible):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Environment variables (.env file in the same directory):
  API_URL   = https://your-app.replit.app
  API_KEY   = your-facial-recognition-api-key

  CAM1_NAME = Camera 1
  CAM1_IP   = 192.168.1.64
  CAM1_USER = admin
  CAM1_PASS = bbffu@275

  CAM2_NAME = Camera 2
  CAM2_IP   = 192.168.1.65
  CAM2_USER = admin
  CAM2_PASS = bbffu@274

  # Optional overrides:
  YOLO_IMGSZ = 640     (set 480 for ~40% faster YOLO; lower recall on small faces)

Usage:
  python facial_recognition_service.py

=============================================================================
WALK-THROUGH + VARYING LIGHT OPTIMISATIONS
=============================================================================
[WALK-THROUGH DETECTION]
  Problem: person visible for only 1-3 seconds; face is angled/blurry mid-stride

  YOLO detection
    · MIN_FACE_PX 35 → 28          catch faces further from camera
    · YOLO_FACE_CONF 0.30 → 0.25   more candidates during motion
    · During normal motion: conf floor = 0.15  (was 0.22)
    · During HIGH motion  : conf floor = 0.12  (fast walkers)
    · WALK_CROP_PADDING 0.15 → 0.25  wider crop when mid-stride

  Blur gate
    · BLUR_THRESHOLD        30 → 25   slightly more permissive static gate
    · BLUR_THRESHOLD_MOTION 12 → 8    much more permissive during motion
    · HIGH_MOTION blur floor: 5.0     for fast walkers; catch any in-focus frame

  Recognition speed
    · INSTANT_CONFIRM_DIST  0.28 → 0.38  single frame match triggers confirmation;
                                          walks through diagonal / angled views
                                          typically land in 0.30-0.37 range
    · VOTE_WINDOW           3 → 2     needs only 2 frames, not 3
    · COSINE_THRESHOLD      0.50 → 0.53  slightly wider for walk-through angles
    · DEEPSORT_MAX_AGE      30 → 45   tracks survive brief occlusions mid-walk
    · IDENTITY_CACHE_SECS   15 → 30   re-entry cache lasts longer for doorways
    · IDENTITY_CACHE_THRESHOLD 0.42 → 0.48  looser re-ID on re-entry

[VARYING LIGHT RECOGNITION]
  Problem: face crops are dark (backlit entrance) or overexposed (bright lobby)

  Per-face selective enhancement
    · FACE_DARK_THRESHOLD    85 → 60  trigger CLAHE for darker faces
    · FACE_BRIGHT_THRESHOLD 195 → 210 trigger CLAHE for brighter faces
    · FACE_REPROCESS_MIN_PX  60 → 40  enhance even smaller face crops
    · _clahe_face clip limit   4 → 6   more aggressive local contrast

  DB lighting augmentation (NEW)
    · For every registered photo, the system generates 4 reference embeddings:
        1. Standard (CLAHE + gamma normalised)
        2. Brightened  (+35%)  — matches dark-environment captures
        3. Darkened    (-35%)  — matches over-lit / backlit captures
        4. CLAHE-enhanced      — matches low-contrast / flat-lit captures
    · 4× more reference points → recognition succeeds across a much wider
      range of lighting conditions without storing extra photos
    · CACHE_VERSION bumped to 2 — old single-embedding cache is auto-rebuilt

[GPU BACKEND]
  · cudnn.benchmark + TF32 (Ampere)
  · torch.compile(reduce-overhead) + multi-batch warmup
  · FP16 + channels_last
  · Pinned-memory CPU→GPU transfer

[SHARED MODEL ARCHITECTURE]
  · Single YOLO + FaceNet via GPUInferenceEngine (saves ~180 MB VRAM)
  · Cross-camera batched FaceNet inference
=============================================================================
"""

import sys
from urllib.parse import quote as _url_quote

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import os
import pickle
import queue
import requests
import threading
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.  Run: pip install ultralytics")
    sys.exit(1)

try:
    import torch
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("ERROR: facenet-pytorch not installed.  Run: pip install facenet-pytorch torch torchvision")
    sys.exit(1)

def _probe_cuda() -> bool:
    """
    Return True only when CUDA is both *available* and *actually functional*.
    torch.cuda.is_available() can return True even when the driver is partially
    broken (e.g. version mismatch, cold-boot initialisation failure).  We run a
    tiny tensor op to confirm the runtime works before committing to GPU mode.
    """
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + 1          # forces a real CUDA kernel launch
        del t
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f"[WARNING] CUDA detected but failed probe ({type(exc).__name__}: {exc}).")
        print("          Falling back to CPU.  To force GPU, set CUDA_LAUNCH_BLOCKING=1")
        print("          and check your driver / PyTorch version compatibility.")
        return False

_USE_GPU: bool = _probe_cuda()

# ── CUDA backend tuning – Ampere / RTX 2050 ──────────────────────────────────
if _USE_GPU:
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.empty_cache()
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print("CuDNN: benchmark=True  |  TF32=True")
else:
    print("GPU  : not available or probe failed – running on CPU.")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: deep_sort_realtime not installed.  Run: pip install deep-sort-realtime")
    sys.exit(1)

try:
    from scipy.spatial import cKDTree as _CKDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False

# ── Core API config ──────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "https://YOUR_APP.replit.app")
API_KEY = os.environ.get("API_KEY", "")

REGISTERED_DIR       = os.path.join(os.path.dirname(__file__), "registered_personnel")
COOLDOWN_SECS        = 60
DETECTION_DELAY      = 0.020   # 50 fps cap – RTX 2050 fp16 sustains this
IDLE_DETECTION_DELAY = 0.10    # 10 fps in idle — was 0.5 (too slow for walk-through)

YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))

# ── Walk-through motion thresholds ───────────────────────────────────────────
MOTION_THRESHOLD        = 1.5    # lowered from 4.0 — small/distant walkers barely move global mean
HIGH_MOTION_THRESHOLD   = 15.0   # above this → person is walking fast
MOTION_DETECT_SCALE     = 0.25
MOTION_HYSTERESIS_SECS  = 3.0   # stay out of idle for 3s after last motion spike
FACE_ACTIVITY_SECS      = 5.0   # stay out of idle for 5s after any active track or confirmed face

# ── Recognition constants (tuned for walk-through + varying light) ────────────
COSINE_THRESHOLD         = 0.53   # slightly wider for angled walk-through faces
MIN_FACE_PX              = 28     # catch faces further from camera (was 35)
VOTE_WINDOW              = 2      # confirm after 2 matched frames, not 3
VOTE_REQUIRED            = 1
CLAHE_CLIP_LIMIT         = 2.5
CLAHE_TILE_GRID          = (8, 8)

FACE_REPROCESS_MIN_PX    = 40    # enhance smaller faces too (was 60)
FACE_DARK_THRESHOLD      = 60    # trigger CLAHE enhancement for darker faces (was 85)
FACE_BRIGHT_THRESHOLD    = 210   # trigger CLAHE enhancement for brighter faces (was 195)

DEEPSORT_MAX_AGE         = 45    # tracks survive longer during walk-through occlusions
DEEPSORT_N_INIT          = 1
DEEPSORT_MAX_IOU         = 0.75

# Walk-through blur gates
BLUR_THRESHOLD           = 25.0  # slightly permissive static gate (was 30)
BLUR_THRESHOLD_MOTION    = 8.0   # permissive during normal motion (was 12)
BLUR_THRESHOLD_HIGH_MOT  = 5.0   # very permissive for fast walkers

# Walk-through confirmation – the most important tuning for walk-through
INSTANT_CONFIRM_DIST     = 0.38  # single-frame match at this dist → instant confirm (was 0.28)
                                  # angled/partial faces land in 0.30-0.37; now caught instantly

WALK_CROP_PADDING        = 0.25  # wider crop mid-stride (was 0.15)
HIGH_MOTION_PADDING      = 0.35  # even wider for fast walkers

# YOLO confidence floors (applied during motion, lower = more detections)
YOLO_FACE_CONF           = 0.25  # base confidence (was 0.30)
MOTION_CONF_FLOOR        = 0.15  # min confidence during normal motion
HIGH_MOTION_CONF_FLOOR   = 0.12  # min confidence during fast movement

MAX_FACES_PER_FRAME      = 14
FROZEN_FRAME_LIMIT       = 45

VIEW_THRESHOLD_OFFSETS: dict = {
    "front": 0.00,
    "left":  0.05,
    "right": 0.05,
    "top":   0.03,
}

YOLO_FACE_MODEL = os.environ.get("YOLO_FACE_MODEL", "yolov8n-face.pt")

ANN_MIN_EMBEDDINGS       = 200
CACHE_VERSION            = 2      # bumped: DB now stores 4 augmented embeddings per photo

IDENTITY_CACHE_SECS      = 30.0   # re-entry cache lasts 30 s for doorway scenarios (was 15)
IDENTITY_CACHE_THRESHOLD = 0.48   # looser re-ID threshold (was 0.42)

# High-motion walking mode: applied when motion_score > HIGH_MOTION_THRESHOLD
MOTION_INSTANT_CONFIRM_DIST = 0.52   # very lenient instant confirm for fast walkers at profile angles
MOTION_COSINE_THRESHOLD     = 0.62   # wider DB gate during high-motion frames
MOTION_CACHE_THRESHOLD      = 0.58   # wider cache re-ID during high motion

# EMA embedding smoothing: blends historical embedding with each new frame's
# embedding to produce a stable representation immune to single-frame blur/occlusion.
EMA_ALPHA = 0.65   # 0.65 * old + 0.35 * new each frame

os.makedirs(REGISTERED_DIR, exist_ok=True)

# Three CLAHE instances – read-only after creation, thread-safe
_clahe       = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
_clahe_frame = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
_clahe_face  = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))   # was 4.0

_gamma_lut_cache: dict = {}
_log_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="log")

# ── Web-streaming mode (activated by --web-mode CLI flag) ─────────────────────
_web_mode: bool      = "--web-mode" in sys.argv
_WEB_FRAMES: dict    = {}          # cam_name → latest JPEG bytes
_WEB_EVENTS: dict    = {}          # cam_name → threading.Event (notify new frame)
_WEB_LOCK            = threading.Lock()


def _update_web_frame(cam_name: str, frame) -> None:
    """JPEG-encode a rendered frame and notify the MJPEG server thread."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    data = buf.tobytes()
    with _WEB_LOCK:
        _WEB_FRAMES[cam_name] = data
    ev = _WEB_EVENTS.get(cam_name)
    if ev:
        ev.set()


def _start_mjpeg_server(port: int = 5001) -> None:
    """Start a lightweight Flask MJPEG stream server in a background thread."""
    try:
        from flask import Flask as _Flask, Response as _Response, jsonify as _jsonify
    except ImportError:
        print("[Web] Flask not installed – web streaming disabled. Run: pip install flask")
        return

    srv = _Flask("mjpeg_server")
    # Silence Flask startup noise
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.WARNING)

    @srv.route("/stream/<path:cam_name>")
    def _stream(cam_name):
        if cam_name not in _WEB_EVENTS:
            _WEB_EVENTS[cam_name] = threading.Event()

        def _gen():
            while True:
                ev = _WEB_EVENTS.get(cam_name)
                if ev:
                    ev.wait(timeout=2.0)
                    ev.clear()
                with _WEB_LOCK:
                    frame = _WEB_FRAMES.get(cam_name)
                if frame:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        return _Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @srv.route("/cameras")
    def _cameras():
        with _WEB_LOCK:
            return _jsonify(list(_WEB_FRAMES.keys()))

    threading.Thread(
        target=lambda: srv.run(host="127.0.0.1", port=port, debug=False, threaded=True),
        name="mjpeg-server",
        daemon=True,
    ).start()
    print(f"[Web] MJPEG stream server started → http://127.0.0.1:{port}")


os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000"
    "|fflags;+discardcorrupt+nobuffer"
    "|flags;+low_delay"
    "|allowed_media_types;video",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CameraConfig:
    name:        str
    ip:          str
    username:    str
    password:    str
    port:        int = 554
    stream_path: str = "/Streaming/Channels/101"
    stream_url:  str = field(default="", init=True)

    def __post_init__(self):
        if not self.stream_url:
            user_enc = _url_quote(self.username, safe="")
            pass_enc = _url_quote(self.password, safe="")
            self.stream_url = (
                f"rtsp://{user_enc}:{pass_enc}"
                f"@{self.ip}:{self.port}{self.stream_path}"
            )

    def safe_url(self) -> str:
        user_enc = _url_quote(self.username, safe="")
        return (
            f"rtsp://{user_enc}:****"
            f"@{self.ip}:{self.port}{self.stream_path}"
        )


def load_cameras_from_env() -> list:
    cameras = []
    idx = 1
    while True:
        prefix = f"CAM{idx}_"
        ip = os.environ.get(f"{prefix}IP", "").strip()
        if not ip:
            break
        cameras.append(CameraConfig(
            name        = os.environ.get(f"{prefix}NAME",        f"Camera {idx}").strip(),
            ip          = ip,
            username    = os.environ.get(f"{prefix}USER",        "admin").strip(),
            password    = os.environ.get(f"{prefix}PASS",        "").strip(),
            port        = int(os.environ.get(f"{prefix}PORT",    "554").strip()),
            stream_path = os.environ.get(f"{prefix}STREAM_PATH", "/Streaming/Channels/101").strip(),
            stream_url  = os.environ.get(f"{prefix}STREAM_URL",  "").strip(),
        ))
        idx += 1

    if not cameras:
        legacy_url = os.environ.get(
            "RTSP_URL",
            "rtsp://admin:bbffu@275@192.168.1.64:554/Streaming/Channels/101"
        )
        cameras.append(CameraConfig(
            name="Camera 1", ip="192.168.1.64",
            username="admin", password="",
            stream_url=legacy_url,
        ))
        print("  [Config] No CAMn_* env vars found – using legacy RTSP_URL fallback.")

    return cameras


# ═══════════════════════════════════════════════════════════════════════════════
#  Lighting-robust preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_gamma_lut(l_channel: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction via cached LUT."""
    if gamma == 1.0:
        return l_channel
    key = round(gamma, 2)
    if key not in _gamma_lut_cache:
        inv_gamma = 1.0 / gamma
        _gamma_lut_cache[key] = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
    return cv2.LUT(l_channel, _gamma_lut_cache[key])


def preprocess_face_roi(roi: np.ndarray) -> np.ndarray:
    """CLAHE + adaptive gamma on a BGR face crop (used for DB registered photos)."""
    if roi is None or roi.size == 0:
        return roi
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq   = _clahe.apply(l)
    mean_l = float(np.mean(l_eq))
    gamma  = max(0.5, mean_l / 128.0) if mean_l < 100 else (min(1.8, mean_l / 128.0) if mean_l > 180 else 1.0)
    l_eq   = _apply_gamma_lut(l_eq, gamma)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Gray-world white balance + 16×16 CLAHE on the full frame before YOLO."""
    if frame is None or frame.size == 0:
        return frame
    b_mean   = float(frame[:, :, 0].mean())
    g_mean   = float(frame[:, :, 1].mean())
    r_mean   = float(frame[:, :, 2].mean())
    mean_all = (b_mean + g_mean + r_mean) / 3.0
    if mean_all > 1.0:
        max_dev = max(abs(b_mean - mean_all), abs(g_mean - mean_all), abs(r_mean - mean_all))
        if max_dev > mean_all * 0.12:
            bgr    = frame.astype(np.float32)
            scales = np.clip(
                np.array([mean_all / (b_mean + 1e-6),
                          mean_all / (g_mean + 1e-6),
                          mean_all / (r_mean + 1e-6)], dtype=np.float32),
                0.5, 2.0
            )
            bgr   *= scales
            frame  = np.clip(bgr, 0, 255).astype(np.uint8)
            del bgr
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    del lab
    l_eq   = _clahe_frame.apply(l)
    del l
    mean_l = float(np.mean(l_eq))
    gamma  = max(0.5, mean_l / 128.0) if mean_l < 100 else (min(1.8, mean_l / 128.0) if mean_l > 180 else 1.0)
    l_eq   = _apply_gamma_lut(l_eq, gamma)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_face_crop_enhanced(crop: np.ndarray) -> np.ndarray:
    """
    Aggressive per-face CLAHE (6×6 clip, 4×4 tiles) for dark/overexposed crops.
    Bilateral denoising is applied first on crops ≥ 48 px to remove outdoor
    sensor noise while preserving edges FaceNet relies on.
    """
    if crop is None or crop.size == 0:
        return crop
    if crop.shape[0] >= 48 and crop.shape[1] >= 48:
        crop = cv2.bilateralFilter(crop, d=5, sigmaColor=35, sigmaSpace=35)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq   = _clahe_face.apply(l)
    mean_l = float(np.mean(l_eq))
    gamma  = max(0.4, mean_l / 128.0) if mean_l < 90 else (min(2.0, mean_l / 128.0) if mean_l > 175 else 1.0)
    l_eq   = _apply_gamma_lut(l_eq, gamma)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def _brighten(img: np.ndarray, factor: float = 1.35) -> np.ndarray:
    """Brighten an image by a fixed factor (simulates dark capture conditions)."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _darken(img: np.ndarray, factor: float = 0.65) -> np.ndarray:
    """Darken an image by a fixed factor (simulates overexposed capture conditions)."""
    return (img.astype(np.float32) * factor).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model builders
# ═══════════════════════════════════════════════════════════════════════════════

_YOLO_FACE_DOWNLOAD_URL = (
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection"
    "/resolve/main/model.pt"
)


def build_yolo_face_detector() -> YOLO:
    model_path = YOLO_FACE_MODEL
    if not os.path.isfile(model_path):
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-face.pt")
        if not os.path.isfile(local_path):
            print("  yolov8n-face.pt not found – downloading from HuggingFace…")
            resp = requests.get(_YOLO_FACE_DOWNLOAD_URL, stream=True, timeout=120)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"  Saved to {local_path}")
        model_path = local_path
    return YOLO(model_path)


# ── Pinned-memory staging buffer ─────────────────────────────────────────────
_pinned_buf: Optional[torch.Tensor] = None


def _ensure_pinned_buf(n: int) -> torch.Tensor:
    global _pinned_buf
    if _pinned_buf is None or _pinned_buf.shape[0] < n:
        _pinned_buf = torch.empty(
            max(n, 20), 3, 160, 160, dtype=torch.float32, pin_memory=True
        )
    return _pinned_buf


def _preprocess_for_facenet(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR crop → normalised CHW float32 tensor [-1, 1]."""
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
    arr     = resized.astype(np.float32) * (1.0 / 127.5) - 1.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _build_face_embedder_on(device: str):
    """
    Internal helper: build and warm up FaceNet on the given device.
    Raises if CUDA fails so the caller can retry on CPU.
    """
    use_fp16 = device == "cuda"
    model    = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    if use_fp16:
        model = model.half()
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        print(f"  FaceNet: fp16 + channels_last on {torch.cuda.get_device_name(0)}.")

    model._use_fp16 = use_fp16

    _cxx_ok = False
    if hasattr(torch, "compile"):
        try:
            from torch._inductor.codecache import cpp_compiler as _cpp_compiler
            _cpp_compiler()
            _cxx_ok = True
        except Exception:
            print("  FaceNet: torch.compile skipped – no C++ compiler found.")
            print("           Install Visual Studio Build Tools to enable it.")

    warmup_done = False
    if _cxx_ok:
        try:
            fp16_flag = use_fp16
            compiled  = torch.compile(model, mode="reduce-overhead", dynamic=False)
            compiled._use_fp16 = fp16_flag
            with torch.inference_mode():
                for bs in (1, 4, 8, 14):
                    dummy = torch.zeros(bs, 3, 160, 160, device=device)
                    if use_fp16:
                        dummy = dummy.half()
                    _ = compiled(dummy)
            model       = compiled
            warmup_done = True
            print("  FaceNet: torch.compile(reduce-overhead) active. Warmup: bs=1,4,8,14.")
        except Exception as exc:
            print(f"  FaceNet: torch.compile failed ({type(exc).__name__}) – eager mode.")

    if not warmup_done:
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 160, 160, device=device)
            if use_fp16:
                dummy = dummy.half()
            _ = model(dummy)
        print(f"  FaceNet: eager mode warmup complete on {device}.")

    return model, device


def build_face_embedder():
    """
    FaceNet with RTX 2050 optimisations:
    fp16 + channels_last + torch.compile(reduce-overhead).
    Warmup at bs=1,4,8,14 so compiled kernels are ready before first frame.

    If CUDA is selected but fails at any point (model load, warmup, etc.)
    the function automatically retries on CPU so the service never crashes
    due to a transient GPU driver error.
    """
    primary = "cuda" if _USE_GPU else "cpu"
    try:
        return _build_face_embedder_on(primary)
    except Exception as exc:
        if primary == "cpu":
            raise   # already on CPU, nothing to fall back to
        print(f"[WARNING] FaceNet failed on CUDA ({type(exc).__name__}: {exc}).")
        print("          Retrying on CPU – recognition will be slower.")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _build_face_embedder_on("cpu")


def extract_face_embedding(crop_bgr: np.ndarray, embedder, device: str) -> np.ndarray:
    """Single-crop L2-normalised 512-dim embedding."""
    use_fp16 = getattr(embedder, "_use_fp16", False)
    tensor   = _preprocess_for_facenet(crop_bgr).unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()
    with torch.inference_mode():
        emb = embedder(tensor).squeeze().float().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-6)


def extract_face_embeddings_batch(crops_bgr: list, embedder, device: str) -> list:
    """
    Batch FaceNet inference with pinned-memory CPU→GPU transfer.
    All crops go through one forward pass – maximises GPU utilisation.
    """
    use_fp16 = getattr(embedder, "_use_fp16", False)
    tensors: list = []
    valid:   list = []
    results       = [None] * len(crops_bgr)

    for i, crop in enumerate(crops_bgr):
        if crop is None or crop.size == 0:
            continue
        try:
            tensors.append(_preprocess_for_facenet(crop))
            valid.append(i)
        except Exception:
            pass

    if tensors:
        n = len(tensors)
        if _USE_GPU:
            buf = _ensure_pinned_buf(n)
            for k, t in enumerate(tensors):
                buf[k].copy_(t)
            batch = buf[:n].to(device, non_blocking=True)
        else:
            batch = torch.stack(tensors).to(device)

        if use_fp16:
            batch = batch.half()
        with torch.inference_mode():
            embs = embedder(batch).float().cpu().numpy()
        for out_i, emb in zip(valid, embs):
            results[out_i] = emb / (np.linalg.norm(emb) + 1e-6)

    return results


def is_blurry(crop: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    if crop is None or crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def estimate_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean abs pixel diff on 25%-scale gray. 0–255; above MOTION_THRESHOLD = movement."""
    scale = MOTION_DETECT_SCALE
    h, w  = prev_gray.shape[:2]
    p = cv2.resize(prev_gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    c = cv2.resize(curr_gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    return float(np.mean(np.abs(c.astype(np.int16) - p.astype(np.int16))))


# ═══════════════════════════════════════════════════════════════════════════════
#  Embedding database with lighting augmentation
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingDatabase:
    """
    Multi-angle, multi-lighting identity profile database.

    For every registered photo the system generates FOUR reference embeddings:
      1. Standard   – CLAHE + gamma normalised (existing behaviour)
      2. Brightened – simulates the registered photo being captured under
                      darker conditions than the live camera sees
      3. Darkened   – simulates over-lit / backlit live captures
      4. CLAHE-enhanced – aggressive local contrast normalisation

    These four variants are stored under compound view labels
    ("front", "front_bright", "front_dark", "front_clahe") and searched
    transparently via find_batch(). _threshold_for() strips the suffix so
    the correct per-angle cosine offset is applied.

    Result: recognition succeeds across a much wider range of lighting
    conditions without requiring additional enrolled photos per person.

    CACHE_VERSION = 2: on first run after upgrade the cache is auto-rebuilt.
    Thread-safety: read-only after construction.
    """

    def __init__(self, yolo: YOLO, embedder, embed_device: str, registered_dir: str):
        self._profiles:    dict                 = {}
        self._flat_ids:    list                 = []
        self._flat_views:  list                 = []
        self._flat_matrix: Optional[np.ndarray] = None
        self._ann_index                         = None
        self._yolo         = yolo
        self._embedder     = embedder
        self._embed_device = embed_device
        self._load(registered_dir)

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _cache_path(directory: str) -> str:
        return os.path.join(directory, "embeddings_cache.pkl")

    def _try_load_cache(self, directory: str, manifest: dict) -> bool:
        cache_file = self._cache_path(directory)
        if not os.path.isfile(cache_file):
            print("  [DB] No embedding cache – processing images.")
            return False
        try:
            with open(cache_file, "rb") as fh:
                data = pickle.load(fh)
            if data.get("version") != CACHE_VERSION:
                print(f"  [DB] Cache version {data.get('version')} ≠ {CACHE_VERSION} – rebuilding.")
                return False
            if data.get("manifest") != manifest:
                added   = set(manifest)         - set(data["manifest"])
                removed = set(data["manifest"]) - set(manifest)
                changed = {
                    k for k in manifest
                    if k in data["manifest"] and manifest[k] != data["manifest"][k]
                }
                parts: list = []
                if added:   parts.append(f"+{len(added)} new")
                if removed: parts.append(f"-{len(removed)} removed")
                if changed: parts.append(f"~{len(changed)} changed")
                print(f"  [DB] Cache stale ({', '.join(parts) or 'unknown'}) – rebuilding.")
                return False
            self._profiles = data["profiles"]
            self._rebuild_matrix()
            n = len(self._profiles)
            print(f"  [DB] Cache hit – {self.embedding_count} embeddings across {n} personnel.")
            for eid, views in sorted(self._profiles.items()):
                labels = ", ".join(v for v, _ in views)
                print(f"       {eid}: {len(views)} variant(s) [{labels}]")
            return True
        except Exception as exc:
            print(f"  [DB] Cache load error ({exc}) – rebuilding.")
            return False

    def _save_cache(self, directory: str, manifest: dict):
        cache_file = self._cache_path(directory)
        try:
            with open(cache_file, "wb") as fh:
                pickle.dump(
                    {"version": CACHE_VERSION, "manifest": manifest,
                     "profiles": self._profiles},
                    fh, protocol=pickle.HIGHEST_PROTOCOL,
                )
            size_kb = os.path.getsize(cache_file) / 1024
            print(f"  [DB] Cache saved ({size_kb:.1f} KB) → {cache_file}")
        except Exception as exc:
            print(f"  [DB] Cache save failed: {exc}")

    def _generate_augmented_embeddings(self, crop: np.ndarray, view: str) -> list:
        """
        Generate 4 lighting-augmented embeddings from a single face crop.

        Why 4 variants?
          Standard  : reference for well-lit, balanced conditions
          Brightened: the DB photo was taken in bright light; live feed is dim
                      → brightening the reference closes the gap
          Darkened  : the DB photo was taken in dim light; live feed is bright
                      → darkening the reference closes the gap
          CLAHE     : extreme local contrast normalisation handles flat/foggy
                      lighting and heavy shadows on the face

        Each variant is stored with a compound view label so _threshold_for()
        can still apply the correct per-angle cosine offset.
        """
        results: list = []
        variants = [
            (view,               crop),
            (f"{view}_bright",   _brighten(crop, 1.35)),
            (f"{view}_dark",     _darken(crop,   0.65)),
            (f"{view}_clahe",    preprocess_face_crop_enhanced(crop)),
        ]
        for label, variant in variants:
            if variant is None or variant.size == 0:
                continue
            try:
                emb = extract_face_embedding(variant, self._embedder, self._embed_device)
                results.append((label, emb))
            except Exception:
                pass
        return results

    def _load(self, directory: str):
        known_views = ("_front", "_left", "_right", "_top")
        manifest: dict = {}
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                manifest[fname] = round(
                    os.path.getmtime(os.path.join(directory, fname)), 3)

        if self._try_load_cache(directory, manifest):
            return

        print(f"  [DB] Generating embeddings for {len(manifest)} image(s) "
              f"(4 lighting variants each)…")
        file_meta = []
        for fname in sorted(manifest.keys()):
            stem   = os.path.splitext(fname)[0]
            emp_id = stem
            view   = "front"
            for suffix in known_views:
                if stem.endswith(suffix):
                    emp_id = stem[: -len(suffix)]
                    view   = suffix[1:]
                    break
            file_meta.append((fname, emp_id, view))

        loaded = skipped = 0
        for fname, emp_id, view in file_meta:
            img = cv2.imread(os.path.join(directory, fname))
            if img is None:
                continue
            img         = preprocess_face_roi(img)
            det_results = self._yolo.predict(img, conf=0.25, verbose=False)
            face_boxes  = []
            for r in det_results:
                for b in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    face_boxes.append((x1, y1, x2, y2, float(b.conf[0])))
            if not face_boxes:
                print(f"  [DB] No face in {fname} – skipped.")
                skipped += 1
                continue
            face_boxes.sort(key=lambda fb: fb[4], reverse=True)
            x1, y1, x2, y2, _ = face_boxes[0]
            ih, iw = img.shape[:2]
            crop = img[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
            if crop.size == 0:
                skipped += 1
                continue

            augmented = self._generate_augmented_embeddings(crop, view)
            if augmented:
                self._profiles.setdefault(emp_id, []).extend(augmented)
                loaded += 1
                variant_labels = ", ".join(lbl for lbl, _ in augmented)
                print(f"  [DB] {fname}: {len(augmented)} variants [{variant_labels}]")
            else:
                skipped += 1

        n_people = len(self._profiles)
        print(f"  [DB] {loaded} photos processed ({skipped} skipped) → "
              f"{self.embedding_count} embeddings across {n_people} personnel.")

        self._rebuild_matrix()
        self._save_cache(directory, manifest)

    def _rebuild_matrix(self):
        self._flat_ids   = []
        self._flat_views = []
        rows             = []
        for emp_id, view_embs in self._profiles.items():
            for view, emb in view_embs:
                self._flat_ids.append(emp_id)
                self._flat_views.append(view)
                rows.append(emb)

        if rows:
            self._flat_matrix = np.stack(rows).astype(np.float32)
            if _HAS_KDTREE and len(rows) >= ANN_MIN_EMBEDDINGS:
                self._ann_index = _CKDTree(self._flat_matrix)
                print(f"  [DB] ANN index built – {len(rows)} embeddings.")
            else:
                self._ann_index = None
        else:
            self._flat_matrix = None
            self._ann_index   = None

    def _threshold_for(self, view: str, base: float = None) -> float:
        # Strip augmentation suffix: "front_bright" → "front", "left_clahe" → "left"
        base_view = view.split("_")[0]
        return (base if base is not None else COSINE_THRESHOLD) + VIEW_THRESHOLD_OFFSETS.get(base_view, 0.0)

    @property
    def person_count(self) -> int:
        return len(self._profiles)

    @property
    def embedding_count(self) -> int:
        return len(self._flat_ids)

    def find(self, query_emb: np.ndarray) -> tuple:
        if self._flat_matrix is None:
            return None, 1.0
        q = query_emb / (np.linalg.norm(query_emb) + 1e-6)
        if self._ann_index is not None:
            eucl, idx = self._ann_index.query(q, k=1, workers=1)
            idx  = int(idx)
            dist = float(eucl ** 2) / 2.0
        else:
            sims = self._flat_matrix @ q
            idx  = int(np.argmax(sims))
            dist = float(1.0 - sims[idx])
        if dist > self._threshold_for(self._flat_views[idx]):
            return None, dist
        return self._flat_ids[idx], dist

    def find_batch(self, query_embs: np.ndarray, threshold_override: float = None) -> list:
        n = len(query_embs)
        if self._flat_matrix is None or n == 0:
            return [(None, 1.0)] * n
        norms       = np.linalg.norm(query_embs, axis=1, keepdims=True)
        query_norms = query_embs / (norms + 1e-6)
        if self._ann_index is not None:
            eucl_dists, idxs = self._ann_index.query(query_norms, k=1, workers=-1)
            best_idxs  = idxs.ravel().astype(int)
            best_dists = (eucl_dists.ravel() ** 2) / 2.0
        else:
            sim_matrix = query_norms @ self._flat_matrix.T
            best_idxs  = np.argmax(sim_matrix, axis=1)
            best_dists = 1.0 - sim_matrix[np.arange(n), best_idxs]
        results = []
        for idx, dist in zip(best_idxs, best_dists):
            dist = float(dist)
            view = self._flat_views[int(idx)]
            results.append(
                (self._flat_ids[int(idx)], dist) if dist <= self._threshold_for(view, threshold_override)
                else (None, dist)
            )
        return results

    def reload(self, registered_dir: str):
        self._profiles.clear()
        self._flat_ids    = []
        self._flat_views  = []
        self._flat_matrix = None
        self._ann_index   = None
        self._load(registered_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  DeepSORT-backed face tracker with majority-vote identity confirmation
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSORTFaceTracker:
    """
    DeepSORT tracker with multi-angle recognition and identity re-entry cache.
    Tuned for walk-through: VOTE_WINDOW=2, INSTANT_CONFIRM_DIST=0.38,
    DEEPSORT_MAX_AGE=45 (tracks survive brief mid-stride occlusions).
    """

    def __init__(self):
        self._tracker = DeepSort(
            max_age          = DEEPSORT_MAX_AGE,
            n_init           = DEEPSORT_N_INIT,
            max_iou_distance = 1.0 - DEEPSORT_MAX_IOU,
            embedder         = None,
        )
        self._track_state:    dict = {}
        self._identity_cache: dict = {}

    def update(self, detections: list, db: EmbeddingDatabase, motion_score: float = 0.0) -> list:
        now = time.time()

        # Motion-adaptive thresholds: relax during fast walking so identity is
        # confirmed before the subject leaves the camera's optimal zone.
        _high_motion     = motion_score > HIGH_MOTION_THRESHOLD
        _instant_thresh  = MOTION_INSTANT_CONFIRM_DIST if _high_motion else INSTANT_CONFIRM_DIST
        _db_threshold    = MOTION_COSINE_THRESHOLD     if _high_motion else None
        _cache_threshold = MOTION_CACHE_THRESHOLD      if _high_motion else IDENTITY_CACHE_THRESHOLD
        _effective_vw    = 1 if _high_motion else VOTE_WINDOW   # single vote confirms in high-motion

        # ── Phase 1: build DeepSORT input + initialise per-track state ───────
        raw_detections: list = []
        embeds:         list = []
        for det in detections:
            x, y, w, h = det["box"]
            raw_detections.append(([x, y, w, h], float(det.get("det_score", 0.9)), None))
            embeds.append(det["emb"])

        tracks = self._tracker.update_tracks(raw_detections, embeds=embeds, frame=None)

        confirmed_tracks: list = []
        to_recognize:     list = []
        active_ids:       set  = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            box = [x1, y1, x2 - x1, y2 - y1]
            active_ids.add(tid)

            if tid not in self._track_state:
                self._track_state[tid] = {
                    "votes":          [],
                    "confirmed_id":   None,
                    "confirmed_dist": 1.0,
                    "last_seen":      now,
                    "last_emb":       None,
                    "avg_emb":        None,   # EMA-smoothed embedding for stable recognition
                    "cache_checked":  False,
                    "best_dist":      1.0,
                }

            state = self._track_state[tid]
            state["last_seen"] = now
            if track.features:
                new_emb           = track.features[-1]
                state["last_emb"] = new_emb
                # EMA: blend running average with newest embedding each frame.
                # Reduces sensitivity to single-frame blur / occlusion / bad angles.
                if state["avg_emb"] is None:
                    state["avg_emb"] = new_emb.copy()
                else:
                    blended = EMA_ALPHA * state["avg_emb"] + (1.0 - EMA_ALPHA) * new_emb
                    norm    = np.linalg.norm(blended)
                    state["avg_emb"] = blended / (norm + 1e-6)

            confirmed_tracks.append((tid, box, state))
            if state["confirmed_id"] is None and track.features:
                # Use EMA embedding when available – more stable than raw latest frame
                emb = state["avg_emb"] if state["avg_emb"] is not None else track.features[-1]
                to_recognize.append((tid, emb))

        # ── Phase 1.5: identity re-entry cache ───────────────────────────────
        for k in [k for k, v in self._identity_cache.items() if v["expire"] < now]:
            del self._identity_cache[k]

        if self._identity_cache and to_recognize:
            cache_ids   = list(self._identity_cache.keys())
            cache_mat   = np.stack([self._identity_cache[k]["emb"] for k in cache_ids])
            cache_norms = cache_mat / (np.linalg.norm(cache_mat, axis=1, keepdims=True) + 1e-6)

            still_unresolved: list = []
            for tid, emb in to_recognize:
                state = self._track_state[tid]
                if state["cache_checked"]:
                    still_unresolved.append((tid, emb))
                    continue
                state["cache_checked"] = True
                q    = emb / (np.linalg.norm(emb) + 1e-6)
                sims = cache_norms @ q
                best = int(np.argmax(sims))
                dist = float(1.0 - sims[best])
                if dist < _cache_threshold:
                    state["confirmed_id"]   = cache_ids[best]
                    state["confirmed_dist"] = dist
                    print(f"  [Re-entry]  T{tid} → {cache_ids[best]} (dist={dist:.3f})")
                else:
                    still_unresolved.append((tid, emb))
            to_recognize = still_unresolved

        # ── Phase 2: batch DB search ──────────────────────────────────────────
        # During high motion, threshold_override widens the gate so fast walkers
        # at profile angles still produce a valid match.
        if to_recognize:
            emb_arr       = np.stack([e for _, e in to_recognize])
            batch_results = db.find_batch(emb_arr, threshold_override=_db_threshold)
            for (tid, _), (emp_id, cos_dist) in zip(to_recognize, batch_results):
                state = self._track_state[tid]
                state["best_dist"] = min(state["best_dist"], cos_dist if emp_id else 1.0)

                if emp_id is not None:
                    state["votes"].append((emp_id, cos_dist))

                if emp_id is not None and cos_dist < _instant_thresh \
                        and state["confirmed_id"] is None:
                    state["confirmed_id"]   = emp_id
                    state["confirmed_dist"] = cos_dist

        # ── Phase 3: majority vote + deduplication ────────────────────────────
        seen_ids: dict = {}

        for tid, box, state in confirmed_tracks:
            if state["confirmed_id"] is None and len(state["votes"]) >= _effective_vw:
                tally: dict = defaultdict(list)
                for v_id, v_dist in state["votes"][-_effective_vw:]:
                    tally[v_id].append(v_dist)
                best_id = max(tally, key=lambda k: (len(tally[k]), -np.mean(tally[k])))
                if len(tally[best_id]) >= VOTE_REQUIRED:
                    state["confirmed_id"]   = best_id
                    state["confirmed_dist"] = float(np.mean(tally[best_id]))

            if state["confirmed_id"]:
                eid   = state["confirmed_id"]
                entry = {
                    "employeeId": eid,
                    "box":        box,
                    "time":       now,
                    "distance":   state["confirmed_dist"],
                    "track_id":   tid,
                }
                if eid not in seen_ids or entry["distance"] < seen_ids[eid]["distance"]:
                    seen_ids[eid] = entry

        # ── Prune dead tracks → populate re-entry cache ───────────────────────
        dead_tids = [t for t in self._track_state if t not in active_ids]
        for tid in dead_tids:
            state = self._track_state.pop(tid)
            if state["confirmed_id"] and state["last_emb"] is not None:
                # Prefer EMA embedding for the cache — it's more stable than the last raw frame
                cache_emb = state["avg_emb"] if state["avg_emb"] is not None else state["last_emb"]
                self._identity_cache[state["confirmed_id"]] = {
                    "emb":    cache_emb.copy(),
                    "dist":   state["confirmed_dist"],
                    "expire": now + IDENTITY_CACHE_SECS,
                }

        return list(seen_ids.values())


# ═══════════════════════════════════════════════════════════════════════════════
#  GPUInferenceEngine – single shared GPU processor for all cameras
# ═══════════════════════════════════════════════════════════════════════════════

class GPUInferenceEngine:
    """
    Owns ONE YOLO + ONE FaceNet, shared by all cameras.

    Walk-through specific behaviour:
      · Three-tier YOLO confidence (normal / motion / high-motion)
      · Three-tier blur gate  (static / motion / high-motion)
      · Wider crop padding during motion (WALK_CROP_PADDING / HIGH_MOTION_PADDING)
      · Selective per-face lighting enhancement for dark/overexposed crops

    Queue protocol:
      frame_queues  (maxsize=2)  newest frame wins; stale frames dropped
      result_queues (maxsize=8)  per-camera detection lists
    """

    def __init__(self):
        print("GPUInferenceEngine: loading shared YOLOv8 model…")
        self._yolo = build_yolo_face_detector()
        print("GPUInferenceEngine: loading shared FaceNet model…")
        self._embedder, self._device = build_face_embedder()

        self._frame_queues:  dict = {}
        self._result_queues: dict = {}
        self._running = True
        self._thread  = threading.Thread(
            target=self._inference_loop, name="gpu-engine", daemon=True
        )
        self._thread.start()
        print("GPUInferenceEngine: inference thread started.")

    @property
    def yolo(self) -> YOLO:
        return self._yolo

    @property
    def embedder(self):
        return self._embedder

    @property
    def embed_device(self) -> str:
        return self._device

    def register_camera(self, cam_id: str):
        self._frame_queues[cam_id]  = queue.Queue(maxsize=2)
        self._result_queues[cam_id] = queue.Queue(maxsize=8)

    def submit_frame(
        self,
        cam_id:        str,
        proc_frame:    np.ndarray,
        orig_frame:    np.ndarray,
        motion_score:  float,
    ):
        """Non-blocking submit. Drops the oldest frame when queue is full."""
        q = self._frame_queues[cam_id]
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait((proc_frame, orig_frame, motion_score))
        except queue.Full:
            pass

    def get_detections(self, cam_id: str) -> Optional[list]:
        try:
            return self._result_queues[cam_id].get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._running = False

    # ── inference loop ───────────────────────────────────────────────────────

    def _inference_loop(self):
        while self._running:
            batch: dict = {}
            for cam_id, q in self._frame_queues.items():
                try:
                    batch[cam_id] = q.get_nowait()
                except queue.Empty:
                    pass

            if not batch:
                time.sleep(0.002)
                continue

            all_crops: list = []
            all_meta:  list = []   # (cam_id, candidate_dict, motion_score)

            for cam_id, (proc_frame, orig_frame, motion_score) in batch.items():
                fh, fw = proc_frame.shape[:2]

                # Three-tier YOLO confidence based on motion level
                # High motion → very low conf floor so fast walkers aren't missed
                if motion_score > HIGH_MOTION_THRESHOLD:
                    yolo_conf = max(HIGH_MOTION_CONF_FLOOR, YOLO_FACE_CONF - 0.13)
                elif motion_score > MOTION_THRESHOLD:
                    yolo_conf = max(MOTION_CONF_FLOOR, YOLO_FACE_CONF - 0.10)
                else:
                    yolo_conf = YOLO_FACE_CONF

                yolo_results = self._yolo.predict(
                    proc_frame, conf=yolo_conf, verbose=False,
                    imgsz=YOLO_IMGSZ, half=_USE_GPU,
                )

                candidates: list = []
                for r in yolo_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        w, h = x2 - x1, y2 - y1
                        if w < MIN_FACE_PX or h < MIN_FACE_PX:
                            continue
                        candidates.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "w":  w,  "h":  h,
                            "det_score": float(box.conf[0]),
                        })
                candidates.sort(key=lambda c: c["det_score"], reverse=True)
                candidates = candidates[:MAX_FACES_PER_FRAME]

                for c in candidates:
                    crop = self._extract_crop(
                        proc_frame, orig_frame, c, fh, fw, motion_score
                    )
                    all_crops.append(crop)
                    all_meta.append((cam_id, c, motion_score))

            # Three-tier motion-adaptive blur gate
            gated_crops: list = []
            for crop, (_, _, ms) in zip(all_crops, all_meta):
                if ms > HIGH_MOTION_THRESHOLD:
                    thresh = BLUR_THRESHOLD_HIGH_MOT
                elif ms > MOTION_THRESHOLD:
                    thresh = BLUR_THRESHOLD_MOTION
                else:
                    thresh = BLUR_THRESHOLD
                gated_crops.append(None if is_blurry(crop, thresh) else crop)

            # Single batched FaceNet pass for ALL cameras
            embs = extract_face_embeddings_batch(gated_crops, self._embedder, self._device)

            # Route detections to per-camera result queues
            cam_detections: dict = {cam_id: [] for cam_id in batch}
            for (cam_id, c, _), emb in zip(all_meta, embs):
                if emb is not None:
                    cam_detections[cam_id].append({
                        "box":       [c["x1"], c["y1"], c["w"], c["h"]],
                        "emb":       emb,
                        "det_score": c["det_score"],
                    })

            for cam_id, detections in cam_detections.items():
                rq = self._result_queues[cam_id]
                if rq.full():
                    try:
                        rq.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    rq.put_nowait(detections)
                except queue.Full:
                    pass

    def _extract_crop(
        self,
        proc_frame:   np.ndarray,
        orig_frame:   np.ndarray,
        c:            dict,
        fh:           int,
        fw:           int,
        motion_score: float,
    ) -> Optional[np.ndarray]:
        """
        Extract a face crop with motion-adaptive padding and lighting enhancement.

        Walk-through padding tiers:
          static       → no extra padding
          motion       → WALK_CROP_PADDING (0.25) – mid-stride misalignment
          high motion  → HIGH_MOTION_PADDING (0.35) – fast walkers
        """
        x1, y1, x2, y2 = c["x1"], c["y1"], c["x2"], c["y2"]
        w, h = c["w"], c["h"]

        if motion_score > HIGH_MOTION_THRESHOLD:
            padding = HIGH_MOTION_PADDING
        elif motion_score > MOTION_THRESHOLD:
            padding = WALK_CROP_PADDING
        else:
            padding = 0.0

        if padding > 0.0:
            px   = int(w * padding)
            py   = int(h * padding)
            crop = proc_frame[max(0, y1 - py):min(fh, y2 + py),
                              max(0, x1 - px):min(fw, x2 + px)]
        else:
            crop = proc_frame[max(0, y1):min(fh, y2),
                              max(0, x1):min(fw, x2)]

        if crop is None or crop.size == 0:
            return None

        # Selective per-face lighting enhancement
        # Triggered for smaller faces too (FACE_REPROCESS_MIN_PX=40, was 60)
        # to help with distant/dark faces during walk-through
        if w >= FACE_REPROCESS_MIN_PX and h >= FACE_REPROCESS_MIN_PX:
            mean_l = float(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
            if mean_l < FACE_DARK_THRESHOLD or mean_l > FACE_BRIGHT_THRESHOLD:
                pad       = max(10, int(max(w, h) * 0.15))
                orig_crop = orig_frame[max(0, y1 - pad):min(fh, y2 + pad),
                                       max(0, x1 - pad):min(fw, x2 + pad)]
                if orig_crop.size > 0:
                    enhanced = preprocess_face_crop_enhanced(orig_crop)
                    re_found = False
                    for rr in self._yolo.predict(
                        enhanced, conf=0.20, verbose=False, imgsz=160, half=_USE_GPU,
                    ):
                        for rb in rr.boxes:
                            rx1, ry1, rx2, ry2 = [int(v) for v in rb.xyxy[0].tolist()]
                            re_crop = enhanced[max(0, ry1):ry2, max(0, rx1):rx2]
                            if re_crop.size > 0:
                                crop     = re_crop
                                re_found = True
                                break
                        if re_found:
                            break
                    if not re_found:
                        crop = enhanced

        return crop


# ═══════════════════════════════════════════════════════════════════════════════
#  API helpers
# ═══════════════════════════════════════════════════════════════════════════════

def download_registered_photos():
    print("Downloading registered personnel photos from API…")
    try:
        resp = requests.get(
            f"{API_URL}/api/logs/personnel-photos",
            headers={"x-api-key": API_KEY}, timeout=30,
        )
        if resp.status_code != 200:
            print(f"  Warning: could not fetch photos ({resp.status_code}): {resp.text[:200]}")
            return
        import base64
        count = 0
        for entry in resp.json():
            emp_id    = entry["employeeId"]
            view_type = entry.get("viewType", "front")
            photo_b64 = entry.get("photoBase64", "")
            if not photo_b64 or not photo_b64.startswith("data:image"):
                continue
            _, b64data = photo_b64.split(",", 1)
            filename = f"{emp_id}_{view_type}.jpg"
            with open(os.path.join(REGISTERED_DIR, filename), "wb") as f:
                f.write(base64.b64decode(b64data))
            count += 1
        print(f"  Downloaded {count} personnel photo(s) to {REGISTERED_DIR}/")
    except Exception as e:
        print(f"  Error downloading photos: {e}")


def submit_log(employee_id: str, log_type: str = "entry", camera_name: str = ""):
    try:
        payload = {"employeeId": employee_id, "logType": log_type}
        if camera_name:
            payload["cameraName"] = camera_name
        resp = requests.post(
            f"{API_URL}/api/logs",
            headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
            json=payload, timeout=10,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            if data.get("duplicate"):
                print(f"  [Cooldown]  {employee_id} – skipped [{camera_name}]")
            else:
                ts = datetime.now().strftime("%I:%M %p")
                print(f"  [Logged]    {employee_id} | {camera_name} | {ts} | TIME IN")
        else:
            print(f"  [API Error] {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [Submit Error] ({camera_name}) {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HikvisionStream – per-camera RTSP reader with auto-reconnect
# ═══════════════════════════════════════════════════════════════════════════════

class HikvisionStream:
    def __init__(self, config: CameraConfig):
        self.config          = config
        self.src             = config.stream_url
        self.name            = config.name
        self.cap             = None
        self.stopped         = False
        self.frame: Optional[np.ndarray] = None
        self.connected       = False
        self.reconnect_count = 0
        self.max_reconnects  = 10
        self._lock           = threading.Lock()
        self._frozen_sig     = None
        self._frozen_count   = 0
        self.connect()

    def connect(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            print(f"[{self.name}] Connecting (attempt {self.reconnect_count + 1})…")
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_ZOOM, 0)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self.frame     = frame
                        self.connected = True
                    self.reconnect_count = 0
                    h, w = frame.shape[:2]
                    print(f"[{self.name}] Connected at {w}×{h}  {self.config.safe_url()}")
                    return True
            print(f"[{self.name}] Failed to connect")
            self.connected = False
            return False
        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            self.connected = False
            return False

    def start(self):
        threading.Thread(target=self._update_loop, name=f"stream-{self.name}",
                         daemon=True).start()
        return self

    def _update_loop(self):
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.connected = False
                    if self.reconnect_count < self.max_reconnects:
                        self.reconnect_count += 1
                        if self.connect():
                            continue
                    else:
                        print(f"[{self.name}] Max reconnects; waiting 30 s…")
                        time.sleep(30)
                        self.reconnect_count = 0
                        continue

                ret, frame = self.cap.read()
                if ret and frame is not None and frame.ndim == 3 and frame.size > 0:
                    thumb = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_NEAREST)
                    sig   = thumb.tobytes()
                    if sig == self._frozen_sig:
                        self._frozen_count += 1
                        if self._frozen_count >= FROZEN_FRAME_LIMIT:
                            print(f"[{self.name}] Frozen stream – reconnecting…")
                            self._frozen_count = 0
                            self._frozen_sig   = None
                            self.connect()
                            continue
                    else:
                        self._frozen_count = 0
                        self._frozen_sig   = sig
                    with self._lock:
                        self.frame = frame
                    self.connected       = True
                    self.reconnect_count = 0
                else:
                    self.connected        = False
                    self.reconnect_count += 1
                    if self.reconnect_count > 5:
                        self.connect()
            except Exception:
                self.connected = False
                time.sleep(1)
                self.connect()
            time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.frame is None:
                return None
            try:
                return self.frame.copy()
            except Exception:
                self.frame = None
                return None

    def stop(self):
        self.stopped = True
        time.sleep(0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print(f"[{self.name}] Stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CameraWorker – RTSP reading + motion estimation + DeepSORT tracking
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWorker:
    """
    Per-camera worker. GPU inference is handled by the shared GPUInferenceEngine.

    Walk-through pipeline:
      1. Read RTSP frame
      2. Compute motion score (cheap CPU op at 25% scale)
      3. preprocess_frame() on CPU – parallel across camera threads
      4. Submit (proc_frame, orig_frame, motion_score) to engine
      5. Poll engine result queue → DeepSORT update → attendance log
    """

    def __init__(
        self,
        config:             CameraConfig,
        system_running_ref: list,
        db:                 EmbeddingDatabase,
        engine:             GPUInferenceEngine,
    ):
        self.config        = config
        self.name          = config.name
        self._running      = system_running_ref
        self.stream        = HikvisionStream(config)
        self.tracker       = DeepSORTFaceTracker()
        self.db            = db
        self._engine       = engine
        self.display_faces: list = []
        self._display_lock = threading.Lock()
        self.motion_score       = MOTION_THRESHOLD + 1.0
        self.is_processing      = False   # guard: prevents overlapping AI loop iterations
        self._already_logged: dict = {}
        self._last_motion_time    = 0.0
        self._last_detection_time = 0.0

        engine.register_camera(self.name)
        print(f"[{self.name}] Camera worker ready (shared GPU engine).")

    def start(self):
        self.stream.start()
        threading.Thread(target=self._ai_loop, name=f"ai-{self.name}",
                         daemon=True).start()
        return self

    def stop(self):
        self.stream.stop()

    @property
    def connected(self) -> bool:
        return self.stream.connected

    def _ai_loop(self):
        last_submit_time = time.time()
        prev_gray: Optional[np.ndarray] = None

        while self._running[0]:
            now = time.time()

            # Poll engine for completed detections
            confirmed: list = []
            detections = self._engine.get_detections(self.name)
            if detections is not None:
                confirmed = self.tracker.update(detections, self.db)
                with self._display_lock:
                    self.display_faces = confirmed
                for person in confirmed:
                    emp_id = person["employeeId"]
                    if now - self._already_logged.get(emp_id, 0) > COOLDOWN_SECS:
                        self._already_logged[emp_id] = now
                        _log_executor.submit(submit_log, emp_id, "entry", self.name)

            # Update hysteresis timers every iteration (motion_score refreshed at submit time)
            if self.motion_score > MOTION_THRESHOLD:
                self._last_motion_time = now
            if self.tracker._track_state or confirmed:
                self._last_detection_time = now

            # Adaptive submission rate — only go idle when truly quiet:
            #   · no motion recently (hysteresis)
            #   · no active tracks or confirmed faces recently
            #   · all known tracks already have an identity
            all_confirmed = not any(
                s["confirmed_id"] is None
                for s in self.tracker._track_state.values()
            )
            in_idle = (
                prev_gray is not None
                and self.motion_score < MOTION_THRESHOLD
                and now - self._last_motion_time    > MOTION_HYSTERESIS_SECS
                and now - self._last_detection_time > FACE_ACTIVITY_SECS
                and all_confirmed
            )
            effective_delay = IDLE_DETECTION_DELAY if in_idle else DETECTION_DELAY

            if self.stream.connected and now - last_submit_time > effective_delay:
                frame = self.stream.get_frame()
                if frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    if w > 1280:
                        frame = cv2.resize(frame, (1280, int(h * 1280 / w)),
                                           interpolation=cv2.INTER_LINEAR)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None and prev_gray.shape == curr_gray.shape:
                        self.motion_score = estimate_motion(prev_gray, curr_gray)
                    prev_gray = curr_gray

                    proc_frame = preprocess_frame(frame)
                    self._engine.submit_frame(
                        self.name, proc_frame, frame, self.motion_score
                    )
                    last_submit_time = now

            time.sleep(0.003)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-camera display renderer – main thread only
# ═══════════════════════════════════════════════════════════════════════════════

def render_camera_window(worker: CameraWorker, fps: int):
    window_name = worker.name
    frame = worker.stream.get_frame()

    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"{worker.name}: Connecting…",
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        if _web_mode:
            _update_web_frame(window_name, placeholder)
        else:
            cv2.imshow(window_name, placeholder)
        return

    display = frame
    now     = time.time()

    with worker._display_lock:
        faces_snapshot = list(worker.display_faces)

    for person in faces_snapshot:
        if now - person["time"] < 5.0:
            x, y, w, h = person["box"]
            dist  = person["distance"]
            color = (
                (0, 255, 0)   if dist < 0.30 else
                (0, 255, 255) if dist < 0.40 else
                (0, 165, 255)
            )
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            confidence = (1 - dist) * 100
            tid        = person.get("track_id", "?")
            label      = f"{person['employeeId']} ({confidence:.1f}%)  T{tid}"
            cv2.putText(display, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    cam_ok       = worker.connected
    ms           = worker.motion_score
    motion_str   = (
        f"HIGH-MOT:{ms:.1f}" if ms > HIGH_MOTION_THRESHOLD else
        f"MOT:{ms:.1f}"       if ms > MOTION_THRESHOLD else
        f"STATIC:{ms:.1f}"
    )
    status_color = (0, 255, 0) if cam_ok else (0, 0, 255)
    status_text  = (
        f"FPS:{fps} | {worker.name} | "
        f"{'ONLINE' if cam_ok else 'OFFLINE'} | "
        f"Faces:{len(faces_snapshot)} | {motion_str}"
    )
    cv2.putText(display, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    if _web_mode:
        _update_web_frame(window_name, display)
    else:
        cv2.imshow(window_name, display)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY environment variable is not set.")
        print("  Add it to your .env file:  API_KEY=your-key-here")
        sys.exit(1)

    if _web_mode:
        _start_mjpeg_server(port=5001)

    cameras = load_cameras_from_env()

    print("\n" + "=" * 68)
    print("  BSU Personnel Monitoring – Facial Recognition Service")
    print("  Multi-Camera  |  YOLOv8 + DeepSORT  |  Walk-Through Edition")
    print("=" * 68)
    print(f"  API URL            : {API_URL}")
    print(f"  Cosine threshold   : {COSINE_THRESHOLD}")
    print(f"  Instant confirm    : {INSTANT_CONFIRM_DIST}  (angled walk-through faces)")
    print(f"  Vote window        : {VOTE_WINDOW} frames")
    print(f"  Motion tiers       : >{MOTION_THRESHOLD} = motion  >{HIGH_MOTION_THRESHOLD} = high-motion")
    print(f"  YOLO imgsz         : {YOLO_IMGSZ}")
    print(f"  Target FPS         : {int(1 / DETECTION_DELAY)}")
    print(f"  DB lighting aug    : 4 variants per photo (std/bright/dark/clahe)")
    print(f"  Cameras            : {len(cameras)}")
    for i, cam in enumerate(cameras, 1):
        print(f"    [{i}] {cam.name}  {cam.safe_url()}")
    print("=" * 68 + "\n")

    download_registered_photos()

    print("Initialising shared GPU inference engine…")
    engine = GPUInferenceEngine()

    print("\nBuilding multi-angle + multi-lighting personnel embedding database…")
    shared_db = EmbeddingDatabase(
        engine.yolo, engine.embedder, engine.embed_device, REGISTERED_DIR
    )
    print(
        f"  Personnel : {shared_db.person_count} registered  |  "
        f"{shared_db.embedding_count} total embeddings (incl. lighting variants)  |  "
        f"ANN: {'cKDTree' if shared_db._ann_index is not None else 'flat matrix'}"
    )

    system_running = [True]

    if not _web_mode:
        for cam in cameras:
            cv2.namedWindow(cam.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam.name, 960, 540)

    print("\nStarting camera workers…")
    workers: list = []
    for cam_cfg in cameras:
        workers.append(CameraWorker(cam_cfg, system_running, shared_db, engine).start())

    time.sleep(3)

    frame_count = 0
    fps_time    = time.time()
    fps         = 0

    if _web_mode:
        print("System running in web mode.  Streams → http://127.0.0.1:5001")
        print("Press Ctrl+C to quit.\n")
        try:
            while True:
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps         = frame_count
                    frame_count = 0
                    fps_time    = time.time()

                for worker in workers:
                    try:
                        render_camera_window(worker, fps)
                    except Exception as exc:
                        print(f"[{worker.name}] Render error: {exc}")

                time.sleep(0.033)   # ~30 fps cap for web streaming
        except KeyboardInterrupt:
            print("\nCtrl+C received.")
    else:
        print("System running. Press 'q' in any camera window to quit.\n")
        try:
            while True:
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps         = frame_count
                    frame_count = 0
                    fps_time    = time.time()

                for worker in workers:
                    try:
                        render_camera_window(worker, fps)
                    except Exception as exc:
                        print(f"[{worker.name}] Render error: {exc}")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C received.")

    system_running[0] = False
    engine.stop()
    print("Shutting down…")
    for worker in workers:
        worker.stop()
    if not _web_mode:
        cv2.destroyAllWindows()
    print("System closed.")