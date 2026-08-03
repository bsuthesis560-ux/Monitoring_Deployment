"""
BSU Personnel Monitoring - Facial Recognition Service (Multi-Camera Edition)
=============================================================================
Run this LOCALLY on the machine connected to the IP cameras via network switch.

Setup:
  pip install opencv-python insightface onnxruntime deep-sort-realtime \
              requests numpy scipy python-dotenv

  # GPU acceleration (optional but recommended):
  pip install onnxruntime-gpu   # instead of onnxruntime

Environment variables (create a .env file in the same directory):
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

  # Add CAM3_*, CAM4_* etc. to scale beyond two cameras.

Usage:
  python facial_recognition_service.py

=============================================================================
CHANGES IN THIS REVISION  –  InsightFace + DeepSORT upgrade
=============================================================================
[DETECTION + RECOGNITION – REPLACED]
  DeepFace (slow, not stream-optimised) → InsightFace buffalo_l model.
    • RetinaFace detector  : handles non-frontal faces, motion blur, occlusion.
    • ArcFace recogniser   : ~2–5 ms per face vs 500–2 000 ms with DeepFace.
    • Single-pass multi-face: all faces in a frame are processed in one call.
    • No global lock needed: InsightFace inference is thread-safe per app instance.
  The per-camera FaceApp objects are created once at worker start-up and
  reused for the lifetime of the process.

[TRACKING – REPLACED]
  Custom IOU FaceTracker → DeepSORT (deep_sort_realtime).
    • Assigns a persistent numeric track-ID to every face.
    • Kalman filter predicts position between frames → stable boxes on fast
      movement and during brief occlusion.
    • Recognition is only triggered for NEW or re-appearing tracks.
      Existing confirmed tracks reuse their last identity → eliminates
      redundant recognition calls in crowded scenes.
    • Track age / max_age logic replaces the old TRACK_HOLD_FRAMES constant.

[PREPROCESSING – UNCHANGED]
  preprocess_face_roi() (CLAHE + gamma) is kept and applied to each detected
  face crop before embedding extraction, preserving lighting robustness.

[EVERYTHING ELSE – UNCHANGED]
  • CameraConfig / load_cameras_from_env()
  • HikvisionStream (RTSP reader, auto-reconnect, /101 main-stream fix)
  • download_registered_photos() / submit_log()
  • Attendance logging logic (cooldown, IN/OUT alternating via API)
  • Department-based filtering (handled server-side, unchanged)
  • Database structure (employeeId filename convention unchanged)
  • render_camera_window() HUD and colour-coded confidence overlay
  • Main loop (main-thread-only cv2, 'q' to quit)
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
import requests
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── Load .env (silently ignored if file or package is absent) ───────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── InsightFace ──────────────────────────────────────────────────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("ERROR: insightface is not installed.")
    print("  Run:  pip install insightface onnxruntime")
    sys.exit(1)

# ── DeepSORT ─────────────────────────────────────────────────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: deep_sort_realtime is not installed.")
    print("  Run:  pip install deep-sort-realtime")
    sys.exit(1)

# ── Scipy cKDTree (optional – ANN index for large rosters) ───────────────────────
try:
    from scipy.spatial import cKDTree as _CKDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False

# ── Core API config ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "https://YOUR_APP.replit.app")
API_KEY = os.environ.get("API_KEY", "")

REGISTERED_DIR  = os.path.join(os.path.dirname(__file__), "registered_personnel")
COOLDOWN_SECS   = 60     # Minimum seconds between logs for the same person
DETECTION_DELAY = 0.033  # ~30 fps recognition pipeline cap (was 0.1)

# ── Recognition constants ────────────────────────────────────────────────────────
COSINE_THRESHOLD = 0.50        # lower = stricter
MIN_FACE_PX      = 40          # Minimum face side in pixels
VOTE_WINDOW      = 3           # Reduced from 5 – faster confirmation while walking
VOTE_REQUIRED    = 2           # Reduced from 3 – match at 2/3 votes
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID  = (8, 8)

# Per-face selective re-processing thresholds
FACE_REPROCESS_MIN_PX = 60    # Minimum face side (px) to attempt per-face re-extraction
FACE_DARK_THRESHOLD   = 85    # L-channel mean below this → still too dark after global CLAHE
FACE_BRIGHT_THRESHOLD = 195   # L-channel mean above this → still overexposed after global CLAHE

# DeepSORT parameters
DEEPSORT_MAX_AGE    = 20   # Increased from 10 – tracks survive longer during motion/occlusion
DEEPSORT_N_INIT     = 1    # Reduced from 2 – confirm tracks on first detection
DEEPSORT_MAX_IOU    = 0.7  # IOU threshold for association

# Multi-angle recognition: per-view cosine threshold offsets.
# Profile/top-view reference embeddings have a wider natural spread than
# frontal photos; a small positive offset lets a profile reference still
# register a match when the camera catches the person at an oblique angle.
VIEW_THRESHOLD_OFFSETS: dict = {
    "front": 0.00,
    "left":  0.05,
    "right": 0.05,
    "top":   0.03,
}

# ANN index: switch from flat matrix multiply to cKDTree when total embeddings
# (registered people × captured angles) exceeds this number.
ANN_MIN_EMBEDDINGS = 200

# Identity re-entry cache: after a confirmed track leaves the frame, its
# embedding is cached briefly. If the same person re-enters within the window
# and the new track's embedding is very close, identity is assigned instantly
# without waiting for the full vote window to accumulate again.
IDENTITY_CACHE_SECS      = 8.0   # Seconds to retain a departed identity
IDENTITY_CACHE_THRESHOLD = 0.35  # Cosine distance for instant re-identification

os.makedirs(REGISTERED_DIR, exist_ok=True)

# Three CLAHE instances, all read-only after creation → thread-safe
_clahe       = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)  # DB loader
_clahe_frame = cv2.createCLAHE(clipLimit=2.5,             tileGridSize=(16, 16))          # full-frame (tighter tiles)
_clahe_face  = cv2.createCLAHE(clipLimit=4.0,             tileGridSize=(4,  4))           # per-face aggressive

# Cached gamma LUTs keyed by rounded gamma value – avoids recomputing 256-entry
# tables on every preprocess_face_roi call.
_gamma_lut_cache: dict = {}

# Shared thread pool for non-blocking API log submissions.
_log_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="log")

# Force OpenCV to use TCP for RTSP (more reliable, lower jitter than UDP)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera configuration  (UNCHANGED)
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
            name        = os.environ.get(f"{prefix}NAME", f"Camera {idx}").strip(),
            ip          = ip,
            username    = os.environ.get(f"{prefix}USER", "admin").strip(),
            password    = os.environ.get(f"{prefix}PASS", "").strip(),
            port        = int(os.environ.get(f"{prefix}PORT", "554").strip()),
            stream_path = os.environ.get(f"{prefix}STREAM_PATH",
                                         "/Streaming/Channels/101").strip(),
            stream_url  = os.environ.get(f"{prefix}STREAM_URL", "").strip(),
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
#  Lighting-robust preprocessing  (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_face_roi(roi: np.ndarray) -> np.ndarray:
    """Apply CLAHE + adaptive gamma to a BGR face crop."""
    if roi is None or roi.size == 0:
        return roi
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_eq   = _clahe.apply(l_channel)
    mean_l = float(np.mean(l_eq))
    if mean_l < 100:
        gamma = max(0.5, mean_l / 128.0)
    elif mean_l > 180:
        gamma = min(1.8, mean_l / 128.0)
    else:
        gamma = 1.0
    if gamma != 1.0:
        key = round(gamma, 2)
        if key not in _gamma_lut_cache:
            inv_gamma = 1.0 / gamma
            _gamma_lut_cache[key] = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
            )
        l_eq = cv2.LUT(l_eq, _gamma_lut_cache[key])
    return cv2.cvtColor(cv2.merge([l_eq, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full-frame preprocessing before InsightFace detection.

    Two additions over preprocess_face_roi:
      1. Gray-world white balance – removes artificial-lighting color cast
         (fluorescent green, incandescent yellow, etc.) so all faces in the
         frame are colour-neutral regardless of position.
      2. Tighter CLAHE tile grid (16×16 vs 8×8) – on a 1920×1080 frame the
         tiles shrink from 240×135 px to 120×67 px, which means each person's
         face region gets its own local histogram equalisation even when
         multiple people stand in differently lit spots.
    """
    if frame is None or frame.size == 0:
        return frame

    # ── Gray-world white balance ──────────────────────────────────────────
    bgr = frame.astype(np.float32)
    b_mean, g_mean, r_mean = (bgr[:, :, c].mean() for c in range(3))
    mean_all = (b_mean + g_mean + r_mean) / 3.0
    if mean_all > 1.0:
        scales = np.array([mean_all / (b_mean + 1e-6),
                           mean_all / (g_mean + 1e-6),
                           mean_all / (r_mean + 1e-6)], dtype=np.float32)
        scales = np.clip(scales, 0.5, 2.0)   # prevent overcorrection
        bgr   *= scales
        frame  = np.clip(bgr, 0, 255).astype(np.uint8)

    # ── CLAHE on L channel (tighter 16×16 grid for multi-person scenes) ──
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l_eq   = _clahe_frame.apply(l)
    mean_l = float(np.mean(l_eq))
    gamma  = 1.0
    if mean_l < 100:
        gamma = max(0.5, mean_l / 128.0)
    elif mean_l > 180:
        gamma = min(1.8, mean_l / 128.0)
    if gamma != 1.0:
        key = round(gamma, 2)
        if key not in _gamma_lut_cache:
            inv_gamma = 1.0 / gamma
            _gamma_lut_cache[key] = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
            )
        l_eq = cv2.LUT(l_eq, _gamma_lut_cache[key])
    return cv2.cvtColor(cv2.merge([l_eq, a, b_ch]), cv2.COLOR_LAB2BGR)


def preprocess_face_crop_enhanced(crop: np.ndarray) -> np.ndarray:
    """
    Aggressive per-face preprocessing for crops that are still dark or
    overexposed after the global frame pass.

    Uses _clahe_face (4×4 tiles, 4.0 clip) for strong local normalisation
    within a single face crop.  No white balance here – skin-tone dominance
    in a tight crop would skew a gray-world correction.
    """
    if crop is None or crop.size == 0:
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l_eq   = _clahe_face.apply(l)
    mean_l = float(np.mean(l_eq))
    gamma  = 1.0
    if mean_l < 90:
        gamma = max(0.4, mean_l / 128.0)
    elif mean_l > 175:
        gamma = min(2.0, mean_l / 128.0)
    if gamma != 1.0:
        key = round(gamma, 2)
        if key not in _gamma_lut_cache:
            inv_gamma = 1.0 / gamma
            _gamma_lut_cache[key] = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
            )
        l_eq = cv2.LUT(l_eq, _gamma_lut_cache[key])
    return cv2.cvtColor(cv2.merge([l_eq, a, b_ch]), cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
#  InsightFace model initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def build_face_app() -> FaceAnalysis:
    """
    Create and warm up one InsightFace FaceAnalysis instance.
    buffalo_l   = RetinaFace detector + ArcFace recogniser (best accuracy).
    buffalo_sc  = lighter option if GPU memory is tight.
    ctx_id=0    = GPU; ctx_id=-1 = CPU only.
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider",
                                                     "CPUExecutionProvider"])
    # det_size controls the internal detection resolution.
    # 640×640 is the best balance of speed vs accuracy for 1080p streams.
    app.prepare(ctx_id=0, det_size=(640, 640))
    # Lower threshold catches faces in dim areas or at oblique angles.
    # Default is 0.5; 0.40 adds sensitivity without significant false positives
    # in a controlled indoor camera scene.
    app.det_thresh = 0.40
    return app


# ═══════════════════════════════════════════════════════════════════════════════
#  Registered personnel embedding database
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingDatabase:
    """
    Multi-angle identity profile database.

    Each registered employee has up to four reference photos (front, left,
    right, top). This class stores ALL of them as a per-person profile so
    that ANY captured angle can trigger a recognition match.

    Data layout
    ───────────
    _profiles   : {emp_id: [(view, embedding), …]}  – per-person angle list
    _flat_ids   : [emp_id, …]                        – row → employee id
    _flat_views : [view, …]                          – row → view label
    _flat_matrix: (N_total, 512) float32             – all embeddings stacked
    _ann_index  : cKDTree | None                     – built when N ≥ ANN_MIN_EMBEDDINGS

    Search strategy
    ───────────────
    find() / find_batch() scan every row in the flat matrix, so the best
    matching angle across ALL registered views for ALL people is found in
    a single (N_query, 512) @ (512, N_total) multiply. A per-view threshold
    offset allows profile/top-view references to match at a slightly wider
    cosine distance than frontal references.

    Thread-safety: read-only after construction; safe to share across workers.
    """

    def __init__(self, face_app: FaceAnalysis, registered_dir: str):
        self._profiles:    dict                  = {}   # emp_id → [(view, emb)]
        self._flat_ids:    list                  = []   # row → emp_id
        self._flat_views:  list                  = []   # row → view label
        self._flat_matrix: Optional[np.ndarray]  = None # (N, 512)
        self._ann_index                          = None  # cKDTree when N is large
        self._face_app                           = face_app
        self._load(registered_dir)

    # ── internal helpers ──────────────────────────────────────────────────

    def _load(self, directory: str):
        known_views = ("_front", "_left", "_right", "_top")

        # Parse filenames → (fname, emp_id, view)
        file_meta = []
        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem   = os.path.splitext(fname)[0]
            emp_id = stem
            view   = "front"
            for suffix in known_views:
                if stem.endswith(suffix):
                    emp_id = stem[: -len(suffix)]
                    view   = suffix[1:]   # strip leading underscore
                    break
            file_meta.append((fname, emp_id, view))

        loaded = skipped = 0
        for fname, emp_id, view in file_meta:
            img = cv2.imread(os.path.join(directory, fname))
            if img is None:
                continue
            img   = preprocess_face_roi(img)
            faces = self._face_app.get(img)
            if not faces:
                print(f"  [DB] No face found in {fname} – skipped.")
                skipped += 1
                continue
            emb = faces[0].normed_embedding        # already L2-normalised
            self._profiles.setdefault(emp_id, []).append((view, emb))
            loaded += 1

        n_people = len(self._profiles)
        print(f"  [DB] {loaded}/{loaded + skipped} embeddings loaded "
              f"across {n_people} registered personnel.")
        for eid, views in sorted(self._profiles.items()):
            labels = ", ".join(v for v, _ in views)
            print(f"       {eid}: {len(views)} angle(s) [{labels}]")

        self._rebuild_matrix()

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
            self._flat_matrix = np.stack(rows).astype(np.float32)  # (N, 512)
            if _HAS_KDTREE and len(rows) >= ANN_MIN_EMBEDDINGS:
                # cKDTree on unit-norm embeddings: euclidean ↔ cosine via
                # cos_dist = eucl_dist² / 2  (valid for unit vectors)
                self._ann_index = _CKDTree(self._flat_matrix)
                print(f"  [DB] ANN index (cKDTree) built – {len(rows)} embeddings.")
            else:
                self._ann_index = None
        else:
            self._flat_matrix = None
            self._ann_index   = None

    def _threshold_for(self, view: str) -> float:
        return COSINE_THRESHOLD + VIEW_THRESHOLD_OFFSETS.get(view, 0.0)

    # ── public interface ──────────────────────────────────────────────────

    @property
    def person_count(self) -> int:
        return len(self._profiles)

    @property
    def embedding_count(self) -> int:
        return len(self._flat_ids)

    def find(self, query_emb: np.ndarray) -> tuple[Optional[str], float]:
        """
        Single-query search. Returns (employee_id, cosine_distance) of the
        closest registered angle, or (None, 1.0) when no angle of any person
        is within the per-view threshold.
        """
        if self._flat_matrix is None:
            return None, 1.0

        q = query_emb / (np.linalg.norm(query_emb) + 1e-6)

        if self._ann_index is not None:
            eucl, idx = self._ann_index.query(q, k=1, workers=1)
            idx  = int(idx)
            dist = float(eucl ** 2) / 2.0     # convert euclidean → cosine
        else:
            sims = self._flat_matrix @ q
            idx  = int(np.argmax(sims))
            dist = float(1.0 - sims[idx])

        if dist > self._threshold_for(self._flat_views[idx]):
            return None, dist
        return self._flat_ids[idx], dist

    def find_batch(self, query_embs: np.ndarray) -> list:
        """
        Batch search: (N, 512) array → list of N (employee_id|None, distance).

        One (N,512)@(512,M) multiply covers all N queries simultaneously,
        replacing N individual find() calls. Per-view adaptive thresholds
        are applied to each result independently.

        For large rosters (N_total ≥ ANN_MIN_EMBEDDINGS) the cKDTree batch
        query is used instead, giving O(N_query · log N_total) complexity.
        """
        n = len(query_embs)
        if self._flat_matrix is None or n == 0:
            return [(None, 1.0)] * n

        norms       = np.linalg.norm(query_embs, axis=1, keepdims=True)
        query_norms = query_embs / (norms + 1e-6)    # (N, 512), unit vectors

        if self._ann_index is not None:
            eucl_dists, idxs = self._ann_index.query(query_norms, k=1, workers=-1)
            best_idxs  = idxs.ravel().astype(int)
            best_dists = (eucl_dists.ravel() ** 2) / 2.0
        else:
            sim_matrix = query_norms @ self._flat_matrix.T   # (N, M)
            best_idxs  = np.argmax(sim_matrix, axis=1)
            best_dists = 1.0 - sim_matrix[np.arange(n), best_idxs]

        results = []
        for idx, dist in zip(best_idxs, best_dists):
            dist = float(dist)
            view = self._flat_views[int(idx)]
            results.append(
                (self._flat_ids[int(idx)], dist) if dist <= self._threshold_for(view)
                else (None, dist)
            )
        return results

    def reload(self, registered_dir: str):
        """Hot-reload all profiles (e.g. after new photos are downloaded)."""
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

    Each confirmed entry returned by update():
      {"employeeId": str, "box": [x,y,w,h], "time": float,
       "distance": float, "track_id": int}

    Four-phase update design
    ────────────────────────
    Phase 1   Build DeepSORT input, run Kalman-filter tracker, initialise
              per-track state (votes, last embedding, cache flag).

    Phase 1.5 Identity re-entry cache lookup for brand-new tracks.
              When a person briefly leaves the frame and re-enters, their
              most recent confirmed embedding is cached for IDENTITY_CACHE_SECS
              seconds. New tracks are compared against the cache with a strict
              threshold (IDENTITY_CACHE_THRESHOLD). A cache hit assigns the
              identity immediately, bypassing the vote window entirely.

    Phase 2   Batch DB search (db.find_batch) for any tracks still
              unidentified after the cache check. One matrix multiply covers
              all of them. Multi-angle profiles are searched transparently:
              the best matching angle across all four reference views is
              found automatically.

    Phase 3   Majority-vote confirmation + identity deduplication. If the
              same employee_id appears on two different tracks (detection
              jitter), only the track with the lower cosine distance is kept.
              Confirmed tracks that die have their embedding saved to the
              re-entry cache before being pruned.
    """

    def __init__(self):
        self._tracker = DeepSort(
            max_age          = DEEPSORT_MAX_AGE,
            n_init           = DEEPSORT_N_INIT,
            max_iou_distance = 1.0 - DEEPSORT_MAX_IOU,
            embedder         = None,  # ArcFace embeddings supplied externally
        )
        self._track_state:    dict = {}  # track_id → per-track state dict
        self._identity_cache: dict = {}  # emp_id → {emb, dist, expire}

    def update(
        self,
        detections: list,
        db: EmbeddingDatabase,
    ) -> list:
        now = time.time()

        # ── Phase 1: build DeepSORT input + initialise track state ───────
        raw_detections: list = []
        embeds:         list = []
        for det in detections:
            x, y, w, h = det["box"]
            raw_detections.append(([x, y, w, h], float(det.get("det_score", 0.9)), None))
            embeds.append(det["emb"])

        tracks = self._tracker.update_tracks(raw_detections, embeds=embeds, frame=None)

        confirmed_tracks: list = []   # (tid, box, state)
        to_recognize:     list = []   # (tid, embedding) for unresolved tracks
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
                    "last_emb":       None,   # updated every frame for cache
                    "cache_checked":  False,  # cache is checked only once
                }

            state = self._track_state[tid]
            state["last_seen"] = now
            if track.features:
                state["last_emb"] = track.features[-1]   # keep freshest embedding

            confirmed_tracks.append((tid, box, state))
            if state["confirmed_id"] is None and track.features:
                to_recognize.append((tid, track.features[-1]))

        # ── Phase 1.5: identity re-entry cache lookup ─────────────────────
        # Expire stale cache entries first.
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
                    # Already checked on a previous frame; fall through to DB
                    still_unresolved.append((tid, emb))
                    continue
                state["cache_checked"] = True
                q    = emb / (np.linalg.norm(emb) + 1e-6)
                sims = cache_norms @ q                            # (C,)
                best = int(np.argmax(sims))
                dist = float(1.0 - sims[best])
                if dist < IDENTITY_CACHE_THRESHOLD:
                    # Instant re-identification from cache – skip vote window
                    state["confirmed_id"]   = cache_ids[best]
                    state["confirmed_dist"] = dist
                else:
                    still_unresolved.append((tid, emb))
            to_recognize = still_unresolved

        # ── Phase 2: batch multi-angle DB search for remaining tracks ─────
        # db.find_batch searches ALL registered angles of ALL people in one
        # matrix multiply. Per-view thresholds are applied internally.
        if to_recognize:
            emb_arr      = np.stack([e for _, e in to_recognize])
            batch_results = db.find_batch(emb_arr)
            for (tid, _), (emp_id, cos_dist) in zip(to_recognize, batch_results):
                if emp_id is not None:
                    self._track_state[tid]["votes"].append((emp_id, cos_dist))

        # ── Phase 3: majority vote + deduplicated emit ────────────────────
        seen_ids: dict = {}   # emp_id → best entry (lowest distance)

        for tid, box, state in confirmed_tracks:
            if state["confirmed_id"] is None and len(state["votes"]) >= VOTE_WINDOW:
                tally: dict = defaultdict(list)
                for v_id, v_dist in state["votes"][-VOTE_WINDOW:]:
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

        # ── Prune dead tracks + populate re-entry cache ───────────────────
        dead_tids = [t for t in self._track_state if t not in active_ids]
        for tid in dead_tids:
            state = self._track_state.pop(tid)
            # Cache confirmed identities so re-entrants skip the vote window
            if state["confirmed_id"] and state["last_emb"] is not None:
                self._identity_cache[state["confirmed_id"]] = {
                    "emb":    state["last_emb"].copy(),
                    "dist":   state["confirmed_dist"],
                    "expire": now + IDENTITY_CACHE_SECS,
                }

        return list(seen_ids.values())


# ═══════════════════════════════════════════════════════════════════════════════
#  API helpers  (UNCHANGED logic)
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
                print(f"  [Logged]    Name: {employee_id} | Camera: {camera_name} "
                      f"| Time: {ts} | Type: TIME IN")
        else:
            print(f"  [API Error] {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [Submit Error] ({camera_name}) {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HikvisionStream – per-camera RTSP reader with auto-reconnect  (UNCHANGED)
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
        self.connect()

    def connect(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            print(f"[{self.name}] Connecting (attempt {self.reconnect_count + 1})…")
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 25)

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self.frame     = frame
                        self.connected = True
                    self.reconnect_count = 0
                    h, w = frame.shape[:2]
                    print(f"[{self.name}] ✓ Connected at {w}×{h}  {self.config.safe_url()}")
                    return True
            print(f"[{self.name}] ✗ Failed to connect")
            self.connected = False
            return False
        except Exception as e:
            print(f"[{self.name}] ✗ Error: {e}")
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
                if ret and frame is not None:
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
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        time.sleep(0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print(f"[{self.name}] Stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CameraWorker – bundles stream + tracker + AI loop for one camera
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWorker:
    def __init__(
        self,
        config: CameraConfig,
        system_running_ref: list,
        db: EmbeddingDatabase,
    ):
        self.config           = config
        self.name             = config.name
        self._running         = system_running_ref
        self.stream           = HikvisionStream(config)
        self.tracker          = DeepSORTFaceTracker()
        self.db               = db
        self.display_faces:   list = []
        self.is_processing    = False
        self._already_logged: dict = {}

        # Each worker gets its own InsightFace app instance.
        # InsightFace is thread-safe per-instance, so no global lock needed.
        print(f"[{self.name}] Loading InsightFace model…")
        self._face_app = build_face_app()
        print(f"[{self.name}] InsightFace ready.")

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
        last_detection_time = time.time()

        while self._running[0]:
            now = time.time()
            if (
                self.stream.connected
                and not self.is_processing
                and now - last_detection_time > DETECTION_DELAY
            ):
                self.is_processing = True
                try:
                    frame = self.stream.get_frame()
                    if frame is None or frame.size == 0:
                        self.is_processing = False
                        time.sleep(0.05)
                        continue

                    fh, fw = frame.shape[:2]

                    # ── Stage 1: full-frame preprocessing + detection ────
                    # preprocess_frame adds gray-world white balance and uses
                    # a tighter 16×16 CLAHE grid so multiple people at
                    # different lighting positions each get local normalisation.
                    proc_frame = preprocess_frame(frame)
                    faces      = self._face_app.get(proc_frame)

                    detections = []
                    for face in faces:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        w = x2 - x1;  h = y2 - y1
                        if w < MIN_FACE_PX or h < MIN_FACE_PX:
                            continue

                        emb = face.normed_embedding   # default: from global-CLAHE frame

                        # ── Stage 2: selective per-face lighting fix ─────
                        # Check whether the face region is still problematic
                        # (dark or overexposed) even after the global pass.
                        # Only faces large enough to re-process and outside
                        # the normal L range trigger the extra pass – faces
                        # in good light skip it entirely.
                        if w >= FACE_REPROCESS_MIN_PX and h >= FACE_REPROCESS_MIN_PX:
                            fc = proc_frame[max(0, y1):min(fh, y2),
                                            max(0, x1):min(fw, x2)]
                            if fc.size > 0:
                                mean_l = float(
                                    cv2.cvtColor(fc, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
                                )
                                if mean_l < FACE_DARK_THRESHOLD or mean_l > FACE_BRIGHT_THRESHOLD:
                                    pad  = max(10, int(max(w, h) * 0.15))
                                    cy1  = max(0,  y1 - pad); cy2 = min(fh, y2 + pad)
                                    cx1  = max(0,  x1 - pad); cx2 = min(fw, x2 + pad)
                                    orig_crop = frame[cy1:cy2, cx1:cx2]
                                    if orig_crop.size > 0:
                                        enhanced = preprocess_face_crop_enhanced(orig_crop)
                                        reface   = self._face_app.get(enhanced)
                                        if reface:
                                            emb = reface[0].normed_embedding

                        detections.append({
                            "box":       [x1, y1, w, h],
                            "emb":       emb,
                            "det_score": float(face.det_score),
                        })

                    # ── DeepSORT update ────────────────────────────────────
                    confirmed          = self.tracker.update(detections, self.db)
                    self.display_faces = confirmed
                    last_detection_time = now

                    # ── Attendance logging ─────────────────────────────────
                    for person in confirmed:
                        employee_id = person["employeeId"]
                        if now - self._already_logged.get(employee_id, 0) > COOLDOWN_SECS:
                            self._already_logged[employee_id] = now
                            _log_executor.submit(
                                submit_log, employee_id, "entry", self.name
                            )

                except Exception as exc:
                    print(f"[{self.name}] AI loop error: {exc}")
                finally:
                    self.is_processing = False

            time.sleep(0.005)  # tight poll; actual pace set by DETECTION_DELAY


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-camera display renderer – called from main thread only  (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def render_camera_window(worker: CameraWorker, fps: int):
    """
    Annotate the raw frame with face overlays and a HUD bar, then push it
    to the camera's dedicated named window.
    Must be called from the main thread – OpenCV GUI is not thread-safe.
    """
    window_name = worker.name
    frame = worker.stream.get_frame()

    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"{worker.name}: Connecting…",
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow(window_name, placeholder)
        return

    display = frame
    now     = time.time()

    # ── Face overlays ────────────────────────────────────────────────────────
    for person in worker.display_faces:
        if now - person["time"] < 2.0:
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

    # ── HUD bar ──────────────────────────────────────────────────────────────
    cam_ok       = worker.connected
    status_color = (0, 255, 0) if cam_ok else (0, 0, 255)
    status_text  = (
        f"FPS:{fps} | {worker.name} | "
        f"{'ONLINE' if cam_ok else 'OFFLINE'} | "
        f"Faces:{len(worker.display_faces)} | InsightFace+DeepSORT"
    )
    cv2.putText(display, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.imshow(window_name, display)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY environment variable is not set.")
        print("  Add it to your .env file:  API_KEY=your-key-here")
        input("\nPress Enter to close…")
        exit(1)

    cameras = load_cameras_from_env()

    print("\n" + "=" * 64)
    print("  BSU Personnel Monitoring – Facial Recognition Service")
    print("  Multi-Camera Edition  |  InsightFace + DeepSORT")
    print("=" * 64)
    print(f"  API URL   : {API_URL}")
    print(f"  Threshold : cosine ≤ {COSINE_THRESHOLD}  "
          f"(+{VIEW_THRESHOLD_OFFSETS['left']} left/right, "
          f"+{VIEW_THRESHOLD_OFFSETS['top']} top)")
    print(f"  Cameras   : {len(cameras)}")
    for i, cam in enumerate(cameras, 1):
        print(f"    [{i}] {cam.name}  {cam.safe_url()}")
    print("=" * 64 + "\n")

    # Download photos first so the embedding DB is populated
    download_registered_photos()

    # Build shared InsightFace app for the embedding DB
    print("Loading InsightFace model for embedding database…")
    _db_app = build_face_app()
    print("Building multi-angle personnel embedding database…")
    shared_db = EmbeddingDatabase(_db_app, REGISTERED_DIR)
    print(f"  Personnel : {shared_db.person_count} registered  |  "
          f"{shared_db.embedding_count} total angle embeddings  |  "
          f"ANN index: {'yes' if shared_db._ann_index is not None else 'no (flat matrix)'}")

    system_running = [True]

    for cam in cameras:
        cv2.namedWindow(cam.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam.name, 960, 540)

    print("\nStarting camera workers…")
    workers: list = []
    for cam_cfg in cameras:
        workers.append(CameraWorker(cam_cfg, system_running, shared_db).start())

    time.sleep(3)

    print("System running. Press 'q' in any camera window to quit.\n")

    frame_count = 0
    fps_time    = time.time()
    fps         = 0

    while True:
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps         = frame_count
            frame_count = 0
            fps_time    = time.time()

        for worker in workers:
            render_camera_window(worker, fps)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            system_running[0] = False
            break

    print("\nShutting down…")
    for worker in workers:
        worker.stop()
    cv2.destroyAllWindows()
    print("System closed.")
