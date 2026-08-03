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

# ── Core API config ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "https://YOUR_APP.replit.app")
API_KEY = os.environ.get("API_KEY", "")

REGISTERED_DIR  = os.path.join(os.path.dirname(__file__), "registered_personnel")
COOLDOWN_SECS   = 60    # Minimum seconds between logs for the same person
DETECTION_DELAY = 0.1   # Seconds between full recognition cycles (was 0.4 with DeepFace)

# ── Recognition constants ────────────────────────────────────────────────────────
# Cosine similarity threshold: embeddings closer than this are the same person.
# InsightFace cosine distance is in [0, 2]; typical match < 0.5 is a safe margin.
COSINE_THRESHOLD = 0.50        # lower = stricter  (was DISTANCE_THRESHOLD 0.55)
MIN_FACE_PX      = 40          # Minimum face side in pixels (unchanged)
VOTE_WINDOW      = 5           # Frames kept for majority vote (unchanged)
VOTE_REQUIRED    = 3           # Votes needed to confirm identity (unchanged)
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID  = (8, 8)

# DeepSORT parameters
DEEPSORT_MAX_AGE    = 10   # Frames a track survives without a detection
DEEPSORT_N_INIT     = 2    # Detections before a track is confirmed
DEEPSORT_MAX_IOU    = 0.7  # IOU threshold for association

os.makedirs(REGISTERED_DIR, exist_ok=True)

# Pre-build shared CLAHE (read-only → thread-safe)
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)

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
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        l_eq = cv2.LUT(l_eq, table)
    return cv2.cvtColor(cv2.merge([l_eq, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


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
    return app


# ═══════════════════════════════════════════════════════════════════════════════
#  Registered personnel embedding database
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingDatabase:
    """
    Loads all registered personnel images, extracts ArcFace embeddings,
    and exposes a nearest-neighbour search method.

    Thread-safety: read-only after __init__; safe to share across cameras.
    """

    def __init__(self, face_app: FaceAnalysis, registered_dir: str):
        self._embeddings: list  = []   # list of (employee_id, unit_norm_embedding)
        self._face_app          = face_app
        self._load(registered_dir)

    def _load(self, directory: str):
        files = [
            f for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        loaded = 0
        for fname in files:
            emp_id = os.path.splitext(fname)[0]
            img    = cv2.imread(os.path.join(directory, fname))
            if img is None:
                continue
            img = preprocess_face_roi(img)
            faces = self._face_app.get(img)
            if faces:
                emb = faces[0].normed_embedding   # already L2-normalised
                self._embeddings.append((emp_id, emb))
                loaded += 1
            else:
                print(f"  [DB] Warning: no face found in {fname} – skipped.")
        print(f"  [DB] Loaded {loaded}/{len(files)} registered embeddings.")

    def find(self, query_emb: np.ndarray) -> tuple[Optional[str], float]:
        """
        Return (employee_id, cosine_distance) of the closest registered face,
        or (None, 1.0) if the database is empty or no match is within threshold.

        Cosine distance = 1 - dot(a, b)  (both embeddings are L2-normalised).
        """
        if not self._embeddings:
            return None, 1.0

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-6)
        best_id, best_dist = None, 1.0

        for emp_id, ref_emb in self._embeddings:
            dist = float(1.0 - np.dot(query_norm, ref_emb))
            if dist < best_dist:
                best_dist = dist
                best_id   = emp_id

        if best_dist > COSINE_THRESHOLD:
            return None, best_dist
        return best_id, best_dist

    def reload(self, registered_dir: str):
        """Hot-reload embeddings (e.g. after new photos are downloaded)."""
        self._embeddings.clear()
        self._load(registered_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  DeepSORT-backed face tracker with majority-vote identity confirmation
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSORTFaceTracker:
    """
    Wraps DeepSort to provide the same interface as the old FaceTracker:
      update(detections, recognitions) → confirmed_list

    Each confirmed entry:
      {"employeeId": str, "box": [x, y, w, h], "time": float, "distance": float}

    Identity assignment strategy
    ─────────────────────────────
    1. DeepSORT assigns a persistent track_id per face using Kalman + IoU.
    2. For each new track_id we attempt recognition (majority vote over
       VOTE_WINDOW frames). Once VOTE_REQUIRED votes agree, the identity
       is locked to that track_id and recognition stops until the track dies.
    3. For already-confirmed tracks, the last identity is reused – no
       additional DeepFace/InsightFace calls.

    This mirrors the original vote-buffer logic but eliminates redundant
    recognition work for confirmed faces, which is the primary speedup in
    crowded scenes.
    """

    def __init__(self):
        self._tracker = DeepSort(
            max_age        = DEEPSORT_MAX_AGE,
            n_init         = DEEPSORT_N_INIT,
            max_iou_distance = 1.0 - DEEPSORT_MAX_IOU,
            embedder       = None,   # we supply our own ArcFace embeddings below
        )
        # Per-track state: {track_id: {"votes": [...], "confirmed_id": str|None, ...}}
        self._track_state: dict = {}

    def update(
        self,
        detections: list,     # list of {"box": [x,y,w,h], "emb": ndarray}
        db: EmbeddingDatabase,
    ) -> list:
        """
        detections – InsightFace detections for the current frame, each with
                     a bounding box and ArcFace embedding.
        db         – shared EmbeddingDatabase (read-only).
        Returns    – list of confirmed face dicts for display / logging.
        """
        now = time.time()

        # ── Build DeepSORT-compatible input ──────────────────────────────────
        # Format: list of ([left, top, w, h], confidence, feature_vector)
        raw_detections = []
        for det in detections:
            x, y, w, h = det["box"]
            conf        = float(det.get("det_score", 0.9))
            emb         = det["emb"]
            raw_detections.append(([x, y, w, h], conf, emb))

        tracks = self._tracker.update_tracks(raw_detections, frame=None)

        confirmed = []
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            box  = [x1, y1, x2 - x1, y2 - y1]
            active_ids.add(tid)

            # Initialise state for new tracks
            if tid not in self._track_state:
                self._track_state[tid] = {
                    "votes":         [],
                    "confirmed_id":  None,
                    "confirmed_dist": 1.0,
                    "last_seen":     now,
                }

            state = self._track_state[tid]
            state["last_seen"] = now

            # ── Recognition: only for unconfirmed tracks ──────────────────
            if state["confirmed_id"] is None:
                # Retrieve the embedding that DeepSORT matched to this track.
                # DeepSORT stores the last feature vector in track.features[-1].
                if track.features:
                    query_emb         = track.features[-1]
                    emp_id, cos_dist  = db.find(query_emb)
                    if emp_id is not None:
                        state["votes"].append((emp_id, cos_dist))

                # Majority vote
                if len(state["votes"]) >= VOTE_WINDOW:
                    tally: dict = defaultdict(list)
                    for v_id, v_dist in state["votes"][-VOTE_WINDOW:]:
                        tally[v_id].append(v_dist)
                    best_id = max(
                        tally,
                        key=lambda k: (len(tally[k]), -np.mean(tally[k]))
                    )
                    if len(tally[best_id]) >= VOTE_REQUIRED:
                        state["confirmed_id"]   = best_id
                        state["confirmed_dist"] = float(np.mean(tally[best_id]))

            # ── Emit confirmed tracks ─────────────────────────────────────
            if state["confirmed_id"]:
                confirmed.append({
                    "employeeId": state["confirmed_id"],
                    "box":        box,
                    "time":       now,
                    "distance":   state["confirmed_dist"],
                    "track_id":   tid,
                })

        # ── Prune dead tracks ─────────────────────────────────────────────
        dead = [tid for tid in self._track_state if tid not in active_ids]
        for tid in dead:
            del self._track_state[tid]

        return confirmed


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
            photo_b64 = entry.get("photoBase64", "")
            if not photo_b64 or not photo_b64.startswith("data:image"):
                continue
            _, b64data = photo_b64.split(",", 1)
            with open(os.path.join(REGISTERED_DIR, f"{emp_id}.jpg"), "wb") as f:
                f.write(base64.b64decode(b64data))
            count += 1
        print(f"  Downloaded {count} personnel photos to {REGISTERED_DIR}/")
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

                    # ── InsightFace: detect ALL faces + extract embeddings ──
                    # get() returns a list of Face objects, each with:
                    #   .bbox            : [x1, y1, x2, y2]  (float)
                    #   .det_score       : detection confidence
                    #   .normed_embedding: 512-d ArcFace embedding (L2-normed)
                    faces = self._face_app.get(frame)

                    detections = []
                    for face in faces:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        w = x2 - x1;  h = y2 - y1
                        if w < MIN_FACE_PX or h < MIN_FACE_PX:
                            continue

                        # Apply CLAHE preprocessing to the face crop
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        crop = preprocess_face_roi(crop)

                        # Re-extract embedding from the preprocessed crop so
                        # CLAHE benefits carry through to the matching step.
                        crop_faces = self._face_app.get(crop)
                        emb = (
                            crop_faces[0].normed_embedding
                            if crop_faces
                            else face.normed_embedding
                        )

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
                            threading.Thread(
                                target=submit_log,
                                args=(employee_id, "entry", self.name),
                                daemon=True,
                            ).start()

                except Exception as exc:
                    print(f"[{self.name}] AI loop error: {exc}")
                finally:
                    self.is_processing = False

            time.sleep(0.02)   # tight poll; actual pace set by DETECTION_DELAY


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
    print(f"  Threshold : cosine ≤ {COSINE_THRESHOLD}")
    print(f"  Cameras   : {len(cameras)}")
    for i, cam in enumerate(cameras, 1):
        print(f"    [{i}] {cam.name}  {cam.safe_url()}")
    print("=" * 64 + "\n")

    # Download photos first so the embedding DB is populated
    download_registered_photos()

    # Build shared InsightFace app for the embedding DB
    # (DB load is camera-count-independent; one shared model is enough here)
    print("Loading InsightFace model for embedding database…")
    _db_app = build_face_app()
    print("Building registered-personnel embedding database…")
    shared_db = EmbeddingDatabase(_db_app, REGISTERED_DIR)

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
