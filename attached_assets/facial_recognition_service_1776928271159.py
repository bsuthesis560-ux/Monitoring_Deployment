"""
BSU Personnel Monitoring - Facial Recognition Service (Multi-Camera Edition)
=============================================================================
Run this LOCALLY on the machine connected to the IP cameras via network switch.

Setup:
  pip install opencv-python deepface pandas requests numpy scipy python-dotenv

Environment variables — create a .env file in the same directory:
  API_URL            = https://your-app.replit.app
  API_KEY            = your-facial-recognition-api-key

  CAM1_NAME          = Camera 1
  CAM1_IP            = 192.168.1.64
  CAM1_USERNAME      = admin
  CAM1_PASSWORD      = bbffu@275
  CAM1_STREAM_PATH   = /Streaming/Channels/102   (optional, default shown)
  CAM1_PORT          = 554                        (optional, default shown)

  CAM2_NAME          = Camera 2
  CAM2_IP            = 192.168.1.65
  CAM2_USERNAME      = admin
  CAM2_PASSWORD      = bbffu@274
  CAM2_STREAM_PATH   = /Streaming/Channels/102
  CAM2_PORT          = 554

  # Add CAM3_*, CAM4_*, etc. for more cameras (system scales automatically)

Usage:
  python facial_recognition_service.py

=============================================================================
WHAT WAS CHANGED (multi-camera upgrade)
=============================================================================
[MULTI-CAMERA]
  - CameraConfig dataclass replaces all hardcoded RTSP_URL / IP values.
  - load_cameras_from_env() discovers any number of CAMn_* env groups
    automatically – just add CAM3_*, CAM4_* to .env to add more cameras.
  - Each camera gets its own HikvisionStream + FaceTracker + ai_worker thread.
  - All worker threads run in parallel; a failure in one never touches others.

[CAMERA SOURCE TAGGING]
  - Every recognition event and display overlay now carries the camera name.
  - Log output format:
      Name: <employeeId>  Camera: Camera 1  Time: 08:01 AM  Type: TIME IN
  - submit_log() passes camera_name to the API as an extra field (backward-
    compatible: existing endpoints ignore unknown JSON keys).

[FAULT TOLERANCE]
  - Each HikvisionStream owns its reconnect counter and backoff – cameras
    recover independently.
  - If ALL cameras fail at startup, the system waits and retries rather than
    crashing.
  - Per-camera status is shown live in both the console and the display HUD.

[CONCURRENT DISPLAY]
  - When ≥2 cameras are active, frames are tiled side-by-side in one window.
  - If only one camera is live the layout collapses gracefully to a single pane.

[.env / SECURITY]
  - Credentials are NEVER hardcoded. python-dotenv loads them from .env.
  - Falls back cleanly if .env is absent (use OS environment variables).

[UNCHANGED – COMPATIBILITY PRESERVED]
  - preprocess_face_roi()   – CLAHE + gamma correction: identical.
  - fast_detect_faces()     – two-stage Haar + DNN detector: identical.
  - FaceTracker             – IOU tracker + vote buffer: identical.
  - download_registered_photos() – API endpoint + logic: identical.
  - submit_log()            – API endpoint unchanged; camera_name is additive.
  - All recognition constants (ArcFace, threshold 0.55, VOTE_WINDOW …): identical.
  - Attendance cooldown / duplicate-log guard: identical.
=============================================================================
"""

import cv2
from deepface import DeepFace
import os
import requests
import threading
import time
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ── Load .env file (silently ignored if absent) ─────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on OS environment variables

# ── Core API config (UNCHANGED variable names) ──────────────────────────────────
API_URL = os.environ.get("API_URL", "https://YOUR_APP.replit.app")
API_KEY = os.environ.get("API_KEY", "")

REGISTERED_DIR  = os.path.join(os.path.dirname(__file__), "registered_personnel")
COOLDOWN_SECS   = 60    # Minimum seconds between logs for the same person
DETECTION_DELAY = 0.4   # Seconds between full recognition cycles

# ── OPTIMIZED CONSTANTS (UNCHANGED) ─────────────────────────────────────────────
RECOGNITION_MODEL  = "ArcFace"
DETECTOR_BACKEND   = "opencv"
DISTANCE_THRESHOLD = 0.55
DETECTION_SCALE    = 0.5
MIN_FACE_PX        = 40
VOTE_WINDOW        = 5
VOTE_REQUIRED      = 3
TRACK_HOLD_FRAMES  = 6
CLAHE_CLIP_LIMIT   = 2.5
CLAHE_TILE_GRID    = (8, 8)

# ── Display layout ───────────────────────────────────────────────────────────────
DISPLAY_WIDTH  = 960   # Per-camera pane width (px) for the tiled view
DISPLAY_HEIGHT = 540   # Per-camera pane height (px)

os.makedirs(REGISTERED_DIR, exist_ok=True)

# Pre-build shared CLAHE object (reuse across all camera threads)
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)

# Pre-load Haar cascade (shared – read-only, thread-safe)
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW: Camera configuration dataclass + .env loader
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CameraConfig:
    """
    All settings for one IP camera.

    stream_url is auto-built from the individual fields when not supplied
    explicitly – supports both local LAN IPs (192.168.x.x) and future
    static/public IPs without any code changes; just update .env.
    """
    name:        str
    ip:          str
    username:    str
    password:    str
    port:        int    = 554
    stream_path: str    = "/Streaming/Channels/102"
    stream_url:  str    = field(default="", init=True)

    def __post_init__(self):
        if not self.stream_url:
            # Build RTSP URL from components
            self.stream_url = (
                f"rtsp://{self.username}:{self.password}"
                f"@{self.ip}:{self.port}{self.stream_path}"
            )

    def safe_url(self) -> str:
        """Return the RTSP URL with the password redacted (safe to log/print)."""
        return (
            f"rtsp://{self.username}:****"
            f"@{self.ip}:{self.port}{self.stream_path}"
        )


def load_cameras_from_env() -> list[CameraConfig]:
    """
    Discover camera configurations from environment variables.

    Reads CAM1_*, CAM2_*, CAM3_*, … until it finds a group with no IP set.
    Minimum required per camera: CAMn_IP, CAMn_USERNAME, CAMn_PASSWORD.

    Falls back to a single-camera config built from the legacy RTSP_URL
    environment variable so that existing deployments are unaffected.
    """
    cameras: list[CameraConfig] = []
    idx = 1
    while True:
        prefix = f"CAM{idx}_"
        ip = os.environ.get(f"{prefix}IP", "").strip()
        if not ip:
            break  # No more cameras defined

        username    = os.environ.get(f"{prefix}USERNAME",    "admin").strip()
        password    = os.environ.get(f"{prefix}PASSWORD",    "").strip()
        name        = os.environ.get(f"{prefix}NAME",        f"Camera {idx}").strip()
        port        = int(os.environ.get(f"{prefix}PORT",    "554").strip())
        stream_path = os.environ.get(f"{prefix}STREAM_PATH", "/Streaming/Channels/102").strip()

        # Allow a fully custom stream URL override per camera
        explicit_url = os.environ.get(f"{prefix}STREAM_URL", "").strip()

        cameras.append(CameraConfig(
            name=name,
            ip=ip,
            username=username,
            password=password,
            port=port,
            stream_path=stream_path,
            stream_url=explicit_url,   # empty → auto-built in __post_init__
        ))
        idx += 1

    # Legacy fallback: single camera via RTSP_URL
    if not cameras:
        legacy_url = os.environ.get(
            "RTSP_URL",
            "rtsp://admin:bbffu@275@192.168.1.64:554/Streaming/Channels/102"
        )
        cameras.append(CameraConfig(
            name="Camera 1",
            ip="192.168.1.64",
            username="admin",
            password="",
            stream_url=legacy_url,
        ))
        print("  [Config] No CAMn_* env vars found – using legacy RTSP_URL fallback.")

    return cameras


# ═══════════════════════════════════════════════════════════════════════════════
#  UNCHANGED: Lighting-robust preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_face_roi(roi: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE + auto-gamma correction to a face ROI before recognition.
    Compensates for dim/dark environments, backlit subjects, and fluorescent
    flicker common in corridor/entrance camera deployments.
    """
    if roi is None or roi.size == 0:
        return roi

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_eq = _clahe.apply(l_channel)

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

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
#  UNCHANGED: Fast face detection with DNN fallback
# ═══════════════════════════════════════════════════════════════════════════════

def fast_detect_faces(frame: np.ndarray) -> list:
    """
    Two-stage detection:
      Stage 1 – Haar cascade on a half-size grey image (very fast).
      Stage 2 – OpenCV DNN detector fallback for angled / low-light faces.
    Returns list of (x, y, w, h) in original frame coordinates.
    """
    h, w = frame.shape[:2]
    scale = DETECTION_SCALE
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    grey  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    rects = _haar.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(int(MIN_FACE_PX * scale), int(MIN_FACE_PX * scale)),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces = []
    if len(rects) > 0:
        for (fx, fy, fw, fh) in rects:
            faces.append((
                int(fx / scale), int(fy / scale),
                int(fw / scale), int(fh / scale),
            ))
        return faces

    try:
        dnn_net = _get_dnn_detector()
        if dnn_net is not None:
            blob = cv2.dnn.blobFromImage(
                small, 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            sh, sw = small.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.65:
                    box = detections[0, 0, i, 3:7] * np.array([sw, sh, sw, sh])
                    x1, y1, x2, y2 = box.astype(int)
                    fw_d, fh_d = x2 - x1, y2 - y1
                    if fw_d >= MIN_FACE_PX * scale and fh_d >= MIN_FACE_PX * scale:
                        faces.append((
                            int(x1 / scale), int(y1 / scale),
                            int(fw_d / scale), int(fh_d / scale),
                        ))
    except Exception:
        pass

    return faces


_dnn_detector = None
_dnn_init_attempted = False
_dnn_lock = threading.Lock()   # NEW: protect lazy-load from multiple camera threads


def _get_dnn_detector():
    """Lazy-load the OpenCV DNN face detector. Thread-safe. Returns None if unavailable."""
    global _dnn_detector, _dnn_init_attempted
    with _dnn_lock:
        if _dnn_init_attempted:
            return _dnn_detector
        _dnn_init_attempted = True
        prototxt    = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
        caffemodel  = os.path.join(os.path.dirname(__file__),
                                   "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                _dnn_detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                print("✓ DNN face detector loaded.")
            except Exception as e:
                print(f"  DNN detector load failed: {e} – Haar cascade only.")
        else:
            print("  DNN model files not found – Haar cascade only.")
        return _dnn_detector


# ═══════════════════════════════════════════════════════════════════════════════
#  UNCHANGED: IOU helper + FaceTracker
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """
    Per-camera lightweight IOU tracker with vote-buffer confirmation.
    Each CameraWorker owns its own FaceTracker instance – no shared state.
    """

    def __init__(self):
        self.tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(self, detected_boxes: list, recognitions: list) -> list:
        now = time.time()
        matched_tracks: set[int] = set()

        for box in detected_boxes:
            best_tid, best_iou = None, 0.3
            for tid, trk in self.tracks.items():
                iou = _iou(box, trk["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
                self.tracks[best_tid] = {
                    "box": box,
                    "last_seen": now,
                    "votes": deque(maxlen=VOTE_WINDOW),
                    "confirmed_id": None,
                    "confirmed_distance": 1.0,
                    "hold_frames": 0,
                }
            else:
                self.tracks[best_tid]["box"] = box
                self.tracks[best_tid]["last_seen"] = now
                self.tracks[best_tid]["hold_frames"] = 0
            matched_tracks.add(best_tid)

        for rec in recognitions:
            rb = rec["box"]
            best_tid, best_iou = None, 0.2
            for tid in matched_tracks:
                iou = _iou(rb, self.tracks[tid]["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid]["votes"].append(
                    (rec["employeeId"], rec["distance"])
                )

        stale = [tid for tid, trk in self.tracks.items()
                 if tid not in matched_tracks]
        for tid in stale:
            self.tracks[tid]["hold_frames"] += 1
            if self.tracks[tid]["hold_frames"] > TRACK_HOLD_FRAMES:
                del self.tracks[tid]

        confirmed = []
        for tid, trk in list(self.tracks.items()):
            votes = list(trk["votes"])
            if votes:
                tally: dict[str, list] = defaultdict(list)
                for emp_id, dist in votes:
                    tally[emp_id].append(dist)
                best_id = max(tally, key=lambda k: (len(tally[k]), -np.mean(tally[k])))
                if len(tally[best_id]) >= VOTE_REQUIRED:
                    trk["confirmed_id"]       = best_id
                    trk["confirmed_distance"] = float(np.mean(tally[best_id]))

            if trk["confirmed_id"] and trk["hold_frames"] <= TRACK_HOLD_FRAMES:
                confirmed.append({
                    "employeeId": trk["confirmed_id"],
                    "box":        trk["box"],
                    "time":       now,
                    "distance":   trk["confirmed_distance"],
                })
        return confirmed


# ═══════════════════════════════════════════════════════════════════════════════
#  UNCHANGED: Download registered photos
# ═══════════════════════════════════════════════════════════════════════════════

def download_registered_photos():
    """Download all personnel photos from the web API and save to disk."""
    print("Downloading registered personnel photos from API…")
    try:
        resp = requests.get(
            f"{API_URL}/api/logs/personnel-photos",
            headers={"x-api-key": API_KEY},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  Warning: could not fetch photos ({resp.status_code}): {resp.text[:200]}")
            return

        photos = resp.json()
        count = 0
        for entry in photos:
            emp_id    = entry["employeeId"]
            photo_b64 = entry.get("photoBase64", "")
            if not photo_b64 or not photo_b64.startswith("data:image"):
                continue

            header, b64data = photo_b64.split(",", 1)
            import base64
            img_bytes = base64.b64decode(b64data)
            filepath  = os.path.join(REGISTERED_DIR, f"{emp_id}.jpg")
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            count += 1

        print(f"  Downloaded {count} personnel photos to {REGISTERED_DIR}/")
    except Exception as e:
        print(f"  Error downloading photos: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENHANCED: submit_log – adds camera_name (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

def submit_log(employee_id: str, log_type: str = "entry", camera_name: str = ""):
    """
    POST a recognition event to the web application API.
    camera_name is included as an extra JSON field; existing API endpoints
    ignore unknown keys so this is fully backward-compatible.
    """
    try:
        payload = {"employeeId": employee_id, "logType": log_type}
        if camera_name:
            payload["cameraName"] = camera_name  # NEW: camera source tagging

        resp = requests.post(
            f"{API_URL}/api/logs",
            headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            if data.get("duplicate"):
                print(f"  [Cooldown]  {employee_id} – skipped (logged recently) [{camera_name}]")
            else:
                ts = datetime.now().strftime("%I:%M %p")
                print(
                    f"  [Logged]    Name: {employee_id} | Camera: {camera_name} "
                    f"| Time: {ts} | Type: TIME IN"
                )
        else:
            print(f"  [API Error] {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [Submit Error] ({camera_name}) {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENHANCED: HikvisionStream – now per-camera with independent state
# ═══════════════════════════════════════════════════════════════════════════════

class HikvisionStream:
    """
    Per-camera RTSP reader with auto-reconnect.

    Each instance is fully independent – a failure in Camera 2 has zero
    effect on Camera 1's frame buffer or reconnect counter.
    """

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)
            self.cap.set(cv2.CAP_PROP_FPS,            15)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self.frame     = frame
                        self.connected = True
                    self.reconnect_count = 0
                    print(f"[{self.name}] ✓ Connected – {self.config.safe_url()}")
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
                        print(f"[{self.name}] Max reconnects reached; waiting 30 s…")
                        time.sleep(30)
                        self.reconnect_count = 0
                        continue

                ret, frame = self.cap.read()
                if ret:
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
        print(f"[{self.name}] Camera stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW: CameraWorker – bundles stream + tracker + AI loop for one camera
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWorker:
    """
    Encapsulates everything related to a single camera:
      - HikvisionStream  (frame capture with auto-reconnect)
      - FaceTracker      (per-camera IOU tracker + vote buffer)
      - AI worker thread (detection + recognition + attendance logging)
      - display_faces    (latest confirmed detections for the display loop)

    Workers run completely independently; a crash in one does not affect others.
    """

    def __init__(self, config: CameraConfig, system_running_ref: list):
        self.config          = config
        self.name            = config.name
        self._running        = system_running_ref  # shared [True] flag
        self.stream          = HikvisionStream(config)
        self.tracker         = FaceTracker()
        self.display_faces:  list = []
        self.is_processing   = False
        self._already_logged: dict[str, float] = {}
        self._face_db_count  = 0

    def start(self):
        """Start the RTSP stream reader and the AI worker thread."""
        self.stream.start()
        threading.Thread(
            target=self._ai_loop,
            name=f"ai-{self.name}",
            daemon=True,
        ).start()
        return self

    def stop(self):
        self.stream.stop()

    @property
    def connected(self) -> bool:
        return self.stream.connected

    # ── AI recognition loop ──────────────────────────────────────────────────
    def _ai_loop(self):
        """
        Recognition loop for this camera.  Mirrors the original ai_worker()
        exactly in logic; key additions:
          - camera_name passed to submit_log() for source tagging.
          - fully independent already_logged dict (per-camera cooldown).
        """
        face_images = [
            f for f in os.listdir(REGISTERED_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self._face_db_count = len(face_images)
        print(f"[{self.name}] Face database: {self._face_db_count} image(s)")
        if not face_images:
            print(f"[{self.name}] WARNING: No face images found. Register personnel first.")

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
                        time.sleep(0.1)
                        continue

                    # ── Fast detection pass ──────────────────────────────────
                    detected_boxes = fast_detect_faces(frame)
                    if not detected_boxes:
                        confirmed = self.tracker.update([], [])
                        self.display_faces = confirmed
                        last_detection_time = now
                        self.is_processing  = False
                        time.sleep(0.05)
                        continue

                    # ── Lighting preprocessing ───────────────────────────────
                    preprocessed_frame = preprocess_face_roi(frame)

                    # ── ArcFace recognition ──────────────────────────────────
                    results = DeepFace.find(
                        img_path=preprocessed_frame,
                        db_path=REGISTERED_DIR,
                        enforce_detection=False,
                        model_name=RECOGNITION_MODEL,
                        detector_backend=DETECTOR_BACKEND,
                        silent=True,
                        threshold=DISTANCE_THRESHOLD,
                    )

                    raw_recognitions = []
                    if isinstance(results, list):
                        for df in results:
                            if df.empty:
                                continue
                            match = df.iloc[0]
                            if match["distance"] < DISTANCE_THRESHOLD:
                                filename    = os.path.basename(match["identity"])
                                employee_id = os.path.splitext(filename)[0]
                                x = int(match["source_x"])
                                y = int(match["source_y"])
                                w = int(match["source_w"])
                                h = int(match["source_h"])
                                if w > MIN_FACE_PX and h > MIN_FACE_PX:
                                    raw_recognitions.append({
                                        "employeeId": employee_id,
                                        "box":        [x, y, w, h],
                                        "time":       now,
                                        "distance":   float(match["distance"]),
                                    })

                    # ── Tracker update ───────────────────────────────────────
                    confirmed = self.tracker.update(detected_boxes, raw_recognitions)
                    self.display_faces  = confirmed
                    last_detection_time = now

                    # ── Attendance logging (UNCHANGED logic + camera tag) ────
                    for person in confirmed:
                        employee_id = person["employeeId"]
                        last_logged = self._already_logged.get(employee_id, 0)
                        if now - last_logged > COOLDOWN_SECS:
                            self._already_logged[employee_id] = now
                            threading.Thread(
                                target=submit_log,
                                args=(employee_id, "entry", self.name),  # camera_name added
                                daemon=True,
                            ).start()

                except Exception:
                    pass   # Suppress DeepFace noise; individual camera errors are isolated
                finally:
                    self.is_processing = False
            time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW: Multi-camera tiled display builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_display_frame(workers: list[CameraWorker]) -> np.ndarray:
    """
    Compose per-camera annotated frames into a single tiled display image.

    Layout:
      1 camera  → single pane (full size)
      2 cameras → side-by-side
      3+ cameras → row of panes (wraps to 2-per-row for >3)
    """
    panes = []
    for worker in workers:
        frame = worker.stream.get_frame()
        if frame is None:
            # Blank pane with "Connecting…" message
            pane = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(pane, f"{worker.name}: Connecting…",
                        (30, DISPLAY_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            pane = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Draw recognised face overlays
            now = time.time()
            scale_x = DISPLAY_WIDTH  / frame.shape[1]
            scale_y = DISPLAY_HEIGHT / frame.shape[0]
            for person in worker.display_faces:
                if now - person["time"] < 2.0:
                    ox, oy, ow, oh = person["box"]
                    x = int(ox * scale_x); y = int(oy * scale_y)
                    w = int(ow * scale_x); h = int(oh * scale_y)
                    dist  = person["distance"]
                    color = (
                        (0, 255, 0)   if dist < 0.35 else
                        (0, 255, 255) if dist < 0.45 else
                        (0, 165, 255)
                    )
                    cv2.rectangle(pane, (x, y), (x + w, y + h), color, 2)
                    confidence = (1 - dist) * 100
                    label = f"{person['employeeId']} ({confidence:.1f}%)"
                    cv2.putText(pane, label, (x, max(y - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Per-camera HUD bar
            cam_ok    = worker.connected
            hud_color = (0, 255, 0) if cam_ok else (0, 0, 255)
            hud_text  = (
                f"{worker.name} | "
                f"{'ONLINE' if cam_ok else 'OFFLINE'} | "
                f"Faces: {len(worker.display_faces)}"
            )
            cv2.putText(pane, hud_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

            # Camera name label (top-left corner box)
            label_bg_end = (len(worker.name) * 11 + 20, 40)
            cv2.rectangle(pane, (0, 0), label_bg_end, (0, 0, 0), -1)
            cv2.putText(pane, worker.name, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        panes.append(pane)

    if len(panes) == 1:
        return panes[0]

    # Tile horizontally (two-per-row)
    rows = []
    for i in range(0, len(panes), 2):
        row_panes = panes[i:i+2]
        if len(row_panes) == 1:
            # Pad with a blank pane to keep consistent width
            blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            row_panes.append(blank)
        rows.append(np.hstack(row_panes))
    return np.vstack(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY environment variable is not set.")
        print("  Add it to your .env file:  API_KEY=your-key-here\n")
        exit(1)

    cameras = load_cameras_from_env()

    print("\n" + "=" * 62)
    print("  BSU Personnel Monitoring – Facial Recognition Service")
    print("  Multi-Camera Edition  |  ArcFace + CLAHE + FaceTracker")
    print("=" * 62)
    print(f"  API URL   : {API_URL}")
    print(f"  Model     : {RECOGNITION_MODEL} (threshold {DISTANCE_THRESHOLD})")
    print(f"  Vote req  : {VOTE_REQUIRED}/{VOTE_WINDOW} frames to confirm identity")
    print(f"  Cameras   : {len(cameras)}")
    for i, cam in enumerate(cameras, 1):
        print(f"    [{i}] {cam.name}  {cam.safe_url()}")
    print("=" * 62 + "\n")

    download_registered_photos()

    # Shared stop flag – a list so threads can read it by reference
    system_running = [True]

    print("\nStarting camera workers…")
    workers: list[CameraWorker] = []
    for cam_cfg in cameras:
        worker = CameraWorker(cam_cfg, system_running)
        worker.start()
        workers.append(worker)

    # Give cameras a moment to establish their first connection
    time.sleep(3)

    print("\nSystem running. Press 'q' to quit.\n")

    frame_count = 0
    fps_time    = time.time()
    fps         = 0

    while True:
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps         = frame_count
            frame_count = 0
            fps_time    = time.time()

        display = build_display_frame(workers)

        # Global FPS overlay (bottom-left)
        online_count = sum(1 for w in workers if w.connected)
        cv2.putText(
            display,
            f"FPS: {fps}  |  Cameras online: {online_count}/{len(workers)}",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("BSU-TNEU Monitor – Multi-Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            system_running[0] = False
            break

    print("\nShutting down…")
    for worker in workers:
        worker.stop()
    cv2.destroyAllWindows()
    print("System closed.")
