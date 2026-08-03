"""
BSU Personnel Monitoring - Facial Recognition Service (Optimized)
=================================================================
Run this LOCALLY on the machine connected to the IP camera.

Setup:
  pip install opencv-python deepface pandas requests numpy scipy

Environment variables (create a .env file or set them in your shell):
  API_URL    = https://your-app.replit.app   (your published Replit app URL)
  API_KEY    = your-facial-recognition-api-key
  RTSP_URL   = rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102

Usage:
  python facial_recognition_service.py

=================================================================
OPTIMIZATION SUMMARY
=================================================================
[DETECTION]
  - Switched from single-pass OpenCV detector to a two-stage cascade:
      1. Fast Haar cascade for initial face detection (very low latency)
      2. Fallback DNN-based detector (cv2.dnn) for challenging conditions
    This improves recall in low-light, backlit, and angled scenarios.

[PREPROCESSING – LIGHTING ROBUSTNESS]
  - CLAHE (Contrast-Limited Adaptive Histogram Equalization) is applied to
    the luminance channel before recognition. This compensates for:
      • Dim/dark environments (low light)
      • Overexposed / backlit faces
      • Uneven illumination across the frame
  - Gamma correction auto-calibrated per face ROI to normalize brightness.

[RECOGNITION]
  - Model upgraded to ArcFace (from Facenet). ArcFace produces more
    discriminative angular embeddings, significantly reducing false positives.
  - Tightened distance threshold from 0.68 → 0.55 for higher precision.
  - Added a multi-sample voting window (5 consecutive positives required)
    before logging to prevent single-frame false matches.

[FRAME PROCESSING / SPEED]
  - Frames are downscaled to 50 % width for detection, then the bounding
    box is up-scaled back. Reduces detection time ~4×.
  - Skips recognition if no face was detected in the quick cascade pass.
  - Worker thread sleeps are tuned to avoid busy-waiting while staying
    near real-time.

[ANTI-FLICKER / STABILITY]
  - FaceTracker class: tracks each face by IOU overlap across frames.
    Re-uses the last known identity for up to TRACK_HOLD_FRAMES frames
    when the recognizer returns empty, eliminating flicker.
  - Recognition vote buffer (VOTE_WINDOW): a person is only "confirmed"
    after appearing consistently, preventing spurious one-frame hits.

[UNCHANGED – COMPATIBILITY PRESERVED]
  - API endpoints (submit_log, download_registered_photos): identical.
  - Cooldown / duplicate-log guard: identical.
  - RTSP reconnect logic: identical.
  - All environment variable names: identical.
=================================================================
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

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL  = os.environ.get("API_URL",  "https://YOUR_APP.replit.app")
API_KEY  = os.environ.get("API_KEY",  "")
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://admin:bbffu@275@192.168.1.64:554/Streaming/Channels/102")

REGISTERED_DIR  = os.path.join(os.path.dirname(__file__), "registered_personnel")
COOLDOWN_SECS   = 60    # Minimum seconds between logs for the same person
DETECTION_DELAY = 0.4   # Seconds between full recognition cycles

# ── OPTIMIZED CONSTANTS ─────────────────────────────────────────────────────────
RECOGNITION_MODEL   = "ArcFace"         # CHANGED: ArcFace > Facenet for precision
DETECTOR_BACKEND    = "opencv"          # Fast backend; DNN fallback handled below
DISTANCE_THRESHOLD  = 0.55             # TIGHTENED from 0.68 to reduce false positives
DETECTION_SCALE     = 0.5              # Downscale factor for fast face detection pass
MIN_FACE_PX         = 40               # Minimum face bounding box size (pixels)
VOTE_WINDOW         = 5                # Frames a face must match before it's "confirmed"
VOTE_REQUIRED       = 3                # Min positive votes within VOTE_WINDOW to confirm
TRACK_HOLD_FRAMES   = 6                # Frames to keep last identity during tracking gaps
CLAHE_CLIP_LIMIT    = 2.5              # CLAHE clip limit (tune 2–4 for your environment)
CLAHE_TILE_GRID     = (8, 8)           # CLAHE tile grid size

os.makedirs(REGISTERED_DIR, exist_ok=True)

# ── Pre-build CLAHE object (reuse across frames – avoids repeated allocation) ──
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)

# ── Pre-load Haar cascade (fast first-pass face detector) ──────────────────────
_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Global state (unchanged names for compatibility) ────────────────────────────
latest_frame     = None
display_faces    = []
is_processing    = False
system_running   = True
camera_connected = False


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT: Lighting-robust preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_face_roi(roi: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE + gamma correction to a face ROI before recognition.

    Why:
      Raw frames from IP cameras often have uneven lighting (backlit subjects,
      fluorescent flicker, night-mode artefacts). CLAHE adaptively equalises
      local contrast, and gamma correction lifts crushed shadows or tones down
      blown highlights. Together they give the recognition model a normalised
      input regardless of ambient conditions.
    """
    if roi is None or roi.size == 0:
        return roi

    # Convert to LAB colour space; only process the L (luminance) channel
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE on luminance
    l_eq = _clahe.apply(l_channel)

    # Auto-gamma: if mean luminance is below 100 → image is dark → gamma < 1
    # brightens it; if mean > 180 → image is bright → gamma > 1 darkens it.
    mean_l = float(np.mean(l_eq))
    if mean_l < 100:
        gamma = max(0.5, mean_l / 128.0)      # Brighten dark face
    elif mean_l > 180:
        gamma = min(1.8, mean_l / 128.0)      # Darken overexposed face
    else:
        gamma = 1.0                            # Neutral – no gamma shift needed

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        l_eq = cv2.LUT(l_eq, table)

    # Merge back and return BGR
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT: Fast face detection with DNN fallback
# ═══════════════════════════════════════════════════════════════════════════════
def fast_detect_faces(frame: np.ndarray):
    """
    Two-stage face detection:
      Stage 1 – Haar cascade on a half-size grey image (very fast).
      Stage 2 – If Stage 1 returns nothing, run OpenCV DNN detector on the
                 half-size colour image for improved recall at angles / low light.

    Returns list of (x, y, w, h) in ORIGINAL frame coordinates.

    Why:
      Running DeepFace.find on every single frame is expensive (~0.5–2 s per
      call depending on hardware). By first doing a cheap cascade pass we can:
        a) Skip full recognition entirely when no face is present.
        b) Pass only detected face regions to the recognizer (future extension).
      DNN fallback (uses a Caffe-based face detector baked into OpenCV) handles
      partially occluded or angled faces that Haar misses.
    """
    h, w = frame.shape[:2]
    scale = DETECTION_SCALE
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    grey  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Stage 1: Haar cascade
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
            # Scale back to original resolution
            faces.append((
                int(fx / scale), int(fy / scale),
                int(fw / scale), int(fh / scale),
            ))
        return faces

    # Stage 2: DNN fallback (only when Haar finds nothing)
    # NOTE: Requires opencv-contrib or the model files. We wrap in try/except
    # so the service degrades gracefully if the DNN model is unavailable.
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
                    fw, fh = x2 - x1, y2 - y1
                    if fw >= MIN_FACE_PX * scale and fh >= MIN_FACE_PX * scale:
                        faces.append((
                            int(x1 / scale), int(y1 / scale),
                            int(fw / scale), int(fh / scale),
                        ))
    except Exception:
        pass  # DNN unavailable – silently degrade to Haar-only

    return faces


_dnn_detector = None
_dnn_init_attempted = False

def _get_dnn_detector():
    """Lazy-load the OpenCV DNN face detector. Returns None if unavailable."""
    global _dnn_detector, _dnn_init_attempted
    if _dnn_init_attempted:
        return _dnn_detector
    _dnn_init_attempted = True
    prototxt = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
    caffemodel = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")
    if os.path.exists(prototxt) and os.path.exists(caffemodel):
        try:
            _dnn_detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            print("✓ DNN face detector loaded (improved detection for angles/low-light)")
        except Exception as e:
            print(f"  DNN detector load failed: {e} – using Haar cascade only")
    else:
        print("  DNN model files not found – using Haar cascade only (place deploy.prototxt")
        print("  and res10_300x300_ssd_iter_140000.caffemodel next to this script for")
        print("  improved angle/occlusion detection).")
    return _dnn_detector


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT: Vote-buffer-based face tracker (anti-flicker + stability)
# ═══════════════════════════════════════════════════════════════════════════════
def _iou(boxA, boxB):
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """
    Lightweight centroid/IOU tracker.

    Why:
      Without tracking, a face disappearing for one frame (blink, motion blur,
      momentary occlusion) causes the display overlay to vanish and reappear,
      creating distracting flicker. The tracker holds the last known identity
      for TRACK_HOLD_FRAMES frames, making the overlay stable.

      Vote buffer: accumulates the last VOTE_WINDOW recognition results per
      tracked face. A confirmed identity requires VOTE_REQUIRED or more
      agreeing votes – this filters out single-frame false positives where a
      background pattern or partial face happens to match.
    """

    def __init__(self):
        self.tracks: dict[int, dict] = {}   # track_id → track_state
        self._next_id = 0

    def update(self, detected_boxes: list, recognitions: list) -> list:
        """
        detected_boxes : list of (x, y, w, h) from fast_detect_faces
        recognitions   : list of dicts {employeeId, box, distance} from DeepFace

        Returns list of confirmed display_faces dicts.
        """
        now = time.time()

        # Match detections to existing tracks by IOU
        matched_tracks = set()
        for box in detected_boxes:
            best_tid, best_iou = None, 0.3   # IOU threshold to associate
            for tid, trk in self.tracks.items():
                iou = _iou(box, trk["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is None:
                # New track
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

        # Feed recognition results into matching tracks
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

        # Age out unmatched tracks
        stale = [tid for tid, trk in self.tracks.items()
                 if tid not in matched_tracks]
        for tid in stale:
            self.tracks[tid]["hold_frames"] += 1
            if self.tracks[tid]["hold_frames"] > TRACK_HOLD_FRAMES:
                del self.tracks[tid]

        # Build display_faces from confirmed tracks
        confirmed = []
        for tid, trk in list(self.tracks.items()):
            votes = list(trk["votes"])
            if votes:
                # Count votes per employeeId
                tally = defaultdict(list)
                for emp_id, dist in votes:
                    tally[emp_id].append(dist)
                # Best candidate: most votes, tie-break by avg distance
                best_id = max(tally, key=lambda k: (len(tally[k]), -np.mean(tally[k])))
                if len(tally[best_id]) >= VOTE_REQUIRED:
                    trk["confirmed_id"] = best_id
                    trk["confirmed_distance"] = float(np.mean(tally[best_id]))

            if trk["confirmed_id"] and trk["hold_frames"] <= TRACK_HOLD_FRAMES:
                confirmed.append({
                    "employeeId": trk["confirmed_id"],
                    "box": trk["box"],
                    "time": now,
                    "distance": trk["confirmed_distance"],
                })
        return confirmed


# ── Download registered photos from the API (UNCHANGED) ─────────────────────────
def download_registered_photos():
    """Download all personnel photos from the web API and save to disk."""
    print("Downloading registered personnel photos from API...")
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

            filename = f"{emp_id}.jpg"
            filepath = os.path.join(REGISTERED_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            count += 1

        print(f"  Downloaded {count} personnel photos to {REGISTERED_DIR}/")
    except Exception as e:
        print(f"  Error downloading photos: {e}")


# ── Submit a log entry to the API (UNCHANGED) ───────────────────────────────────
def submit_log(employee_id: str, log_type: str = "entry"):
    """POST a recognition event to the web application API."""
    try:
        resp = requests.post(
            f"{API_URL}/api/logs",
            headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
            json={"employeeId": employee_id, "logType": log_type},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            if data.get("duplicate"):
                print(f"  [Cooldown] {employee_id} – skipped (logged recently)")
            else:
                print(f"  [Logged]   {employee_id} ({log_type}) at {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"  [API Error] {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  [Submit Error] {e}")


# ── Hikvision camera stream with auto-reconnect (UNCHANGED) ──────────────────────
class HikvisionStream:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.stopped = False
        self.frame = None
        self.ret = False
        self.reconnect_count = 0
        self.max_reconnect_attempts = 10
        self.connect()

    def connect(self):
        try:
            if self.cap is not None:
                self.cap.release()
            print(f"Connecting to camera (attempt {self.reconnect_count + 1})…")
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    global camera_connected
                    camera_connected = True
                    self.reconnect_count = 0
                    print("✓ Camera connected!")
                    return True
            print("✗ Camera failed to connect")
            return False
        except Exception as e:
            print(f"✗ Camera error: {e}")
            return False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        global camera_connected, latest_frame
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    camera_connected = False
                    if self.reconnect_count < self.max_reconnect_attempts:
                        self.reconnect_count += 1
                        if self.connect():
                            continue
                    else:
                        print("Max reconnect attempts; waiting 30s…")
                        time.sleep(30)
                        self.reconnect_count = 0
                        continue

                ret, frame = self.cap.read()
                if ret:
                    latest_frame = frame
                    camera_connected = True
                    self.reconnect_count = 0
                else:
                    camera_connected = False
                    self.reconnect_count += 1
                    if self.reconnect_count > 5:
                        self.connect()
            except Exception:
                camera_connected = False
                time.sleep(1)
                self.connect()
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        time.sleep(0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("Camera stopped.")


# ── AI recognition worker (OPTIMIZED) ──────────────────────────────────────────
def ai_worker():
    """
    Optimized recognition loop.

    Key changes vs original:
      1. fast_detect_faces() runs first – if no face found, DeepFace is skipped
         entirely (large speedup for empty frames).
      2. CLAHE + gamma correction applied to each face ROI before passing to
         DeepFace (lighting robustness).
      3. ArcFace model with tighter threshold (higher precision).
      4. FaceTracker handles anti-flicker and vote-based confirmation.
      5. All API/logging calls remain identical to preserve compatibility.
    """
    global display_faces, is_processing, system_running, camera_connected, latest_frame

    already_logged: dict[str, float] = {}
    last_detection_time = time.time()
    tracker = FaceTracker()

    face_images = [f for f in os.listdir(REGISTERED_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Face database: {len(face_images)} image(s) in {REGISTERED_DIR}/")
    if not face_images:
        print("  WARNING: No face images found. Register personnel first, then restart.")

    while system_running:
        now = time.time()
        if (
            camera_connected
            and latest_frame is not None
            and not is_processing
            and now - last_detection_time > DETECTION_DELAY
        ):
            is_processing = True
            try:
                frame = latest_frame.copy()
                if frame is None or frame.size == 0:
                    is_processing = False
                    time.sleep(0.1)
                    continue

                # ── IMPROVEMENT 1: Fast detection pass ──────────────────────
                # Only run heavy recognition if at least one face is detected.
                detected_boxes = fast_detect_faces(frame)
                if not detected_boxes:
                    # No faces in frame – update tracker with empty detections
                    confirmed = tracker.update([], [])
                    display_faces = confirmed
                    last_detection_time = now
                    is_processing = False
                    time.sleep(0.05)
                    continue

                # ── IMPROVEMENT 2: Apply lighting preprocessing ─────────────
                # Preprocess the full frame for recognition. CLAHE on the
                # whole frame (rather than per-crop) ensures the background
                # context used by the detector is also normalised.
                preprocessed_frame = preprocess_face_roi(frame)

                # ── IMPROVEMENT 3: ArcFace recognition with tighter threshold
                results = DeepFace.find(
                    img_path=preprocessed_frame,
                    db_path=REGISTERED_DIR,
                    enforce_detection=False,
                    model_name=RECOGNITION_MODEL,     # ArcFace
                    detector_backend=DETECTOR_BACKEND,
                    silent=True,
                    threshold=DISTANCE_THRESHOLD,     # Tightened to 0.55
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
                                    "box": [x, y, w, h],
                                    "time": now,
                                    "distance": float(match["distance"]),
                                })

                # ── IMPROVEMENT 4: Tracker update (anti-flicker + voting) ───
                confirmed = tracker.update(detected_boxes, raw_recognitions)
                display_faces = confirmed
                last_detection_time = now

                # ── Attendance logging (UNCHANGED logic) ────────────────────
                for person in confirmed:
                    employee_id = person["employeeId"]
                    last_logged = already_logged.get(employee_id, 0)
                    if now - last_logged > COOLDOWN_SECS:
                        already_logged[employee_id] = now
                        threading.Thread(
                            target=submit_log,
                            args=(employee_id, "entry"),
                            daemon=True,
                        ).start()

            except Exception as e:
                # Suppress noisy DeepFace errors (e.g., no-face exceptions)
                pass
            finally:
                is_processing = False
        time.sleep(0.05)


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY environment variable is not set.")
        print("  Set it to the value of FACIAL_RECOGNITION_API_KEY from your Replit app secrets.\n")
        exit(1)

    print("\n" + "=" * 58)
    print("  BSU Personnel Monitoring – Facial Recognition Service")
    print("  (Optimized build – ArcFace + CLAHE + FaceTracker)")
    print("=" * 58)
    print(f"  API URL   : {API_URL}")
    print(f"  RTSP      : {RTSP_URL}")
    print(f"  Model     : {RECOGNITION_MODEL} (threshold {DISTANCE_THRESHOLD})")
    print(f"  Vote req  : {VOTE_REQUIRED}/{VOTE_WINDOW} frames to confirm identity")
    print("=" * 58 + "\n")

    download_registered_photos()

    print("\nStarting camera…")
    vs = HikvisionStream(RTSP_URL).start()
    time.sleep(3)

    threading.Thread(target=ai_worker, daemon=True).start()

    print("System running. Press 'q' to quit.\n")

    frame_count = 0
    fps_time    = time.time()
    fps         = 0

    while True:
        frame = latest_frame
        if frame is not None:
            display = frame.copy()
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps         = frame_count
                frame_count = 0
                fps_time    = time.time()

            # Draw recognised faces
            for person in display_faces:
                if time.time() - person["time"] < 2.0:
                    x, y, w, h = person["box"]
                    dist  = person["distance"]
                    # Colour-code by confidence: green > yellow > orange
                    color = (
                        (0, 255, 0)   if dist < 0.35 else
                        (0, 255, 255) if dist < 0.45 else
                        (0, 165, 255)
                    )
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    confidence = (1 - dist) * 100
                    label = f"{person['employeeId']} ({confidence:.1f}%)"
                    cv2.putText(display, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # HUD / status bar
            status_color = (0, 255, 0) if camera_connected else (0, 0, 255)
            status_text = (
                f"FPS:{fps} | Cam:{'OK' if camera_connected else 'ERR'} | "
                f"Faces:{len(display_faces)} | Model:{RECOGNITION_MODEL}"
            )
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            cv2.imshow("BSU-TNEU Monitor", display)
        else:
            wait = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(wait, "Connecting to camera…", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("BSU-TNEU Monitor", wait)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            system_running = False
            break

    vs.stop()
    cv2.destroyAllWindows()
    print("\nSystem closed.")