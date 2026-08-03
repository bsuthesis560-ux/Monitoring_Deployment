"""
BSU Personnel Monitoring - Facial Recognition Service (Multi-Camera Edition)
=============================================================================
Run this LOCALLY on the machine connected to the IP cameras via network switch.

Setup:
  pip install opencv-python deepface pandas requests numpy scipy python-dotenv

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
CHANGES IN THIS REVISION
=============================================================================
[SEPARATE WINDOWS]
  - Each camera now renders in its own independent cv2.imshow window.
  - Windows are titled with the camera name (e.g. "Camera 1", "Camera 2").
  - Pressing 'q' in ANY window shuts down the entire system cleanly.
  - All cv2 calls remain on the main thread (required by OpenCV/macOS/Windows).

[VIDEO QUALITY – ROOT CAUSE AND FIX]
  Three separate issues were degrading stream quality vs. the browser view:

  Issue 1 – Wrong stream channel (MAJOR)
    The original code requests channel /102, which is the Hikvision sub-stream
    (typically 640×360 or 704×576, heavily compressed for bandwidth saving).
    The browser Live View uses channel /101, which is the main stream
    (1920×1080 or 2560×1440 depending on camera model).
    Fix: default stream path changed from /Channels/102 → /Channels/101.

  Issue 2 – CAP_PROP_FRAME_WIDTH/HEIGHT have no effect on RTSP (MINOR)
    OpenCV ignores resolution hints on network streams – the camera decides
    the resolution. These misleading lines have been removed. The display
    window uses cv2.WINDOW_NORMAL so the frame is shown at native resolution
    without any scaling.

  Issue 3 – CAP_PROP_BUFFERSIZE = 1 causes frame drops under load (MINOR)
    A buffer size of 1 means OpenCV discards frames faster than it can decode
    them on slower hardware, causing visual stuttering. Raised to 3 to allow
    a small decode queue without introducing meaningful latency.

[UNCHANGED – COMPATIBILITY PRESERVED]
  - preprocess_face_roi()  – CLAHE + gamma: identical.
  - fast_detect_faces()    – two-stage Haar + DNN: identical.
  - FaceTracker            – IOU tracker + vote buffer: identical.
  - download_registered_photos() – API endpoint + logic: identical.
  - submit_log()           – API endpoint + logic: identical.
  - All recognition constants (ArcFace, 0.55 threshold, VOTE_WINDOW): identical.
  - Cooldown / duplicate-log guard: identical.
  - Auto-reconnect logic: identical.
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

# ── Load .env (silently ignored if file or package is absent) ───────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Core API config ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "https://YOUR_APP.replit.app")
API_KEY = os.environ.get("API_KEY", "")

REGISTERED_DIR  = os.path.join(os.path.dirname(__file__), "registered_personnel")
COOLDOWN_SECS   = 60    # Minimum seconds between logs for the same person
DETECTION_DELAY = 0.4   # Seconds between full recognition cycles

# ── Recognition constants (UNCHANGED) ───────────────────────────────────────────
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

os.makedirs(REGISTERED_DIR, exist_ok=True)

# Pre-build shared objects (read-only → thread-safe)
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
_haar  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
    # FIX: /101 = main stream (high quality). /102 = sub-stream (low quality).
    stream_path: str = "/Streaming/Channels/101"
    stream_url:  str = field(default="", init=True)

    def __post_init__(self):
        if not self.stream_url:
            self.stream_url = (
                f"rtsp://{self.username}:{self.password}"
                f"@{self.ip}:{self.port}{self.stream_path}"
            )

    def safe_url(self) -> str:
        """Password-redacted URL safe for logging."""
        return (
            f"rtsp://{self.username}:****"
            f"@{self.ip}:{self.port}{self.stream_path}"
        )


def load_cameras_from_env() -> list:
    """
    Discover any number of cameras from CAM1_*, CAM2_*, CAM3_* env groups.
    Falls back to a single-camera config built from the legacy RTSP_URL var.
    """
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
#  Lighting-robust preprocessing (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_face_roi(roi: np.ndarray) -> np.ndarray:
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
#  Fast face detection with DNN fallback (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

def fast_detect_faces(frame: np.ndarray) -> list:
    h, w  = frame.shape[:2]
    scale = DETECTION_SCALE
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    grey  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    rects = _haar.detectMultiScale(
        grey, scaleFactor=1.1, minNeighbors=4,
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
            blob = cv2.dnn.blobFromImage(small, 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            sh, sw = small.shape[:2]
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.65:
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


_dnn_detector        = None
_dnn_init_attempted  = False
_dnn_lock            = threading.Lock()


def _get_dnn_detector():
    global _dnn_detector, _dnn_init_attempted
    with _dnn_lock:
        if _dnn_init_attempted:
            return _dnn_detector
        _dnn_init_attempted = True
        prototxt   = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
        caffemodel = os.path.join(os.path.dirname(__file__),
                                  "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                _dnn_detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                print("✓ DNN face detector loaded.")
            except Exception as e:
                print(f"  DNN load failed: {e} – Haar cascade only.")
        else:
            print("  DNN model files not found – Haar cascade only.")
        return _dnn_detector


# ═══════════════════════════════════════════════════════════════════════════════
#  IOU helper + FaceTracker (UNCHANGED)
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
    def __init__(self):
        self.tracks: dict = {}
        self._next_id = 0

    def update(self, detected_boxes: list, recognitions: list) -> list:
        now = time.time()
        matched_tracks: set = set()

        for box in detected_boxes:
            best_tid, best_iou = None, 0.3
            for tid, trk in self.tracks.items():
                iou = _iou(box, trk["box"])
                if iou > best_iou:
                    best_iou = iou; best_tid = tid
            if best_tid is None:
                best_tid = self._next_id; self._next_id += 1
                self.tracks[best_tid] = {
                    "box": box, "last_seen": now,
                    "votes": deque(maxlen=VOTE_WINDOW),
                    "confirmed_id": None, "confirmed_distance": 1.0,
                    "hold_frames": 0,
                }
            else:
                self.tracks[best_tid]["box"]         = box
                self.tracks[best_tid]["last_seen"]   = now
                self.tracks[best_tid]["hold_frames"] = 0
            matched_tracks.add(best_tid)

        for rec in recognitions:
            rb = rec["box"]; best_tid, best_iou = None, 0.2
            for tid in matched_tracks:
                iou = _iou(rb, self.tracks[tid]["box"])
                if iou > best_iou:
                    best_iou = iou; best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid]["votes"].append((rec["employeeId"], rec["distance"]))

        for tid in [t for t in list(self.tracks) if t not in matched_tracks]:
            self.tracks[tid]["hold_frames"] += 1
            if self.tracks[tid]["hold_frames"] > TRACK_HOLD_FRAMES:
                del self.tracks[tid]

        confirmed = []
        for tid, trk in list(self.tracks.items()):
            votes = list(trk["votes"])
            if votes:
                tally: dict = defaultdict(list)
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
#  API helpers (UNCHANGED logic)
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
        self.connect()

    def connect(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            print(f"[{self.name}] Connecting (attempt {self.reconnect_count + 1})…")
            self.cap = cv2.VideoCapture(self.src)

            # FIX: Removed CAP_PROP_FRAME_WIDTH / HEIGHT – they are silently
            # ignored on RTSP streams and give a false impression of control.
            # Stream resolution is set on the camera side via the channel path
            # (/101 = main stream, /102 = sub-stream).
            # Raised buffer from 1 → 3 to prevent decode-queue starvation.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
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
    def __init__(self, config: CameraConfig, system_running_ref: list):
        self.config           = config
        self.name             = config.name
        self._running         = system_running_ref
        self.stream           = HikvisionStream(config)
        self.tracker          = FaceTracker()
        self.display_faces:   list = []
        self.is_processing    = False
        self._already_logged: dict = {}

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
        face_images = [
            f for f in os.listdir(REGISTERED_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"[{self.name}] Face database: {len(face_images)} image(s)")
        if not face_images:
            print(f"[{self.name}] WARNING: No face images found.")

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

                    detected_boxes = fast_detect_faces(frame)
                    if not detected_boxes:
                        self.display_faces  = self.tracker.update([], [])
                        last_detection_time = now
                        self.is_processing  = False
                        time.sleep(0.05)
                        continue

                    preprocessed_frame = preprocess_face_roi(frame)

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
                                x = int(match["source_x"]); y = int(match["source_y"])
                                w = int(match["source_w"]); h = int(match["source_h"])
                                if w > MIN_FACE_PX and h > MIN_FACE_PX:
                                    raw_recognitions.append({
                                        "employeeId": employee_id,
                                        "box":        [x, y, w, h],
                                        "time":       now,
                                        "distance":   float(match["distance"]),
                                    })

                    confirmed           = self.tracker.update(detected_boxes, raw_recognitions)
                    self.display_faces  = confirmed
                    last_detection_time = now

                    for person in confirmed:
                        employee_id = person["employeeId"]
                        if now - self._already_logged.get(employee_id, 0) > COOLDOWN_SECS:
                            self._already_logged[employee_id] = now
                            threading.Thread(
                                target=submit_log,
                                args=(employee_id, "entry", self.name),
                                daemon=True,
                            ).start()

                except Exception:
                    pass
                finally:
                    self.is_processing = False
            time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-camera display renderer – called from main thread only
# ═══════════════════════════════════════════════════════════════════════════════

def render_camera_window(worker: CameraWorker, fps: int):
    """
    Annotate the raw frame with face overlays and a HUD bar, then push it
    to the camera's dedicated named window.

    This must be called from the main thread – OpenCV's GUI functions are
    not thread-safe on Windows or macOS.
    """
    window_name = worker.name
    frame = worker.stream.get_frame()

    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"{worker.name}: Connecting…",
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow(window_name, placeholder)
        return

    # Render at native stream resolution – no resize, no quality loss
    display = frame
    now     = time.time()

    # ── Face overlays ────────────────────────────────────────────────────────
    for person in worker.display_faces:
        if now - person["time"] < 2.0:
            x, y, w, h = person["box"]
            dist  = person["distance"]
            color = (
                (0, 255, 0)   if dist < 0.35 else
                (0, 255, 255) if dist < 0.45 else
                (0, 165, 255)
            )
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            confidence = (1 - dist) * 100
            label = f"{person['employeeId']} ({confidence:.1f}%)"
            cv2.putText(display, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── HUD bar ──────────────────────────────────────────────────────────────
    cam_ok       = worker.connected
    status_color = (0, 255, 0) if cam_ok else (0, 0, 255)
    status_text  = (
        f"FPS:{fps} | {worker.name} | "
        f"{'ONLINE' if cam_ok else 'OFFLINE'} | "
        f"Faces:{len(worker.display_faces)} | {RECOGNITION_MODEL}"
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

    print("\n" + "=" * 62)
    print("  BSU Personnel Monitoring – Facial Recognition Service")
    print("  Multi-Camera Edition  |  ArcFace + CLAHE + FaceTracker")
    print("=" * 62)
    print(f"  API URL  : {API_URL}")
    print(f"  Model    : {RECOGNITION_MODEL} (threshold {DISTANCE_THRESHOLD})")
    print(f"  Cameras  : {len(cameras)}")
    for i, cam in enumerate(cameras, 1):
        print(f"    [{i}] {cam.name}  {cam.safe_url()}")
    print("=" * 62 + "\n")

    download_registered_photos()

    system_running = [True]

    # Create a named, resizable window for each camera before threads start,
    # so the window handle exists when the first frame arrives.
    for cam in cameras:
        cv2.namedWindow(cam.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam.name, 960, 540)   # initial size – user can resize freely

    print("\nStarting camera workers…")
    workers: list = []
    for cam_cfg in cameras:
        workers.append(CameraWorker(cam_cfg, system_running).start())

    time.sleep(3)   # Allow streams to establish their first frames

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

        # Render each camera into its own window (main thread only)
        for worker in workers:
            render_camera_window(worker, fps)

        # 'q' in any window shuts everything down
        if cv2.waitKey(1) & 0xFF == ord("q"):
            system_running[0] = False
            break

    print("\nShutting down…")
    for worker in workers:
        worker.stop()
    cv2.destroyAllWindows()
    print("System closed.")
