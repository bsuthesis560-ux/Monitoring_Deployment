# facial_recognition_service.py
# COMPLETE VERSION
# Includes:
# - Real-time facial recognition
# - Walk-past recognition
# - Blur filtering
# - Anti-duplicate logs
# - Normal camera view (no zoom)
# - Multi-camera support
# - DeepSORT tracking
# - InsightFace recognition

import os
import cv2
import time
import requests
import threading
import numpy as np
from dataclasses import dataclass, field

from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort

# =====================================================
# CONFIG
# =====================================================

API_URL = os.getenv("API_URL", "")
API_KEY = os.getenv("API_KEY", "")

COOLDOWN_SECS = 20
DETECTION_DELAY = 0.03

COSINE_THRESHOLD = 0.42
MIN_FACE_PX = 28

REGISTERED_DIR = "registered_personnel"

os.makedirs(REGISTERED_DIR, exist_ok=True)

# =====================================================
# CAMERA CONFIG
# =====================================================

@dataclass
class CameraConfig:
    name: str
    ip: str
    username: str
    password: str
    port: int = 554
    stream_path: str = "/Streaming/Channels/101"
    stream_url: str = field(default="", init=True)

    def __post_init__(self):
        if not self.stream_url:
            self.stream_url = (
                f"rtsp://{self.username}:{self.password}"
                f"@{self.ip}:{self.port}{self.stream_path}"
            )

def load_cameras_from_env():
    cams = []
    idx = 1

    while True:
        prefix = f"CAM{idx}_"
        ip = os.getenv(prefix + "IP", "")

        if not ip:
            break

        cams.append(CameraConfig(
            name=os.getenv(prefix + "NAME", f"Camera {idx}"),
            ip=ip,
            username=os.getenv(prefix + "USER", "admin"),
            password=os.getenv(prefix + "PASS", "")
        ))

        idx += 1

    return cams

# =====================================================
# FACE MODEL
# =====================================================

def build_face_app():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return app

# =====================================================
# DATABASE
# =====================================================

class EmbeddingDatabase:
    def __init__(self, app):
        self.app = app
        self.db = []
        self.load()

    def load(self):
        files = os.listdir(REGISTERED_DIR)

        for file in files:
            path = os.path.join(REGISTERED_DIR, file)

            img = cv2.imread(path)

            if img is None:
                continue

            faces = self.app.get(img)

            if faces:
                emb = faces[0].normed_embedding
                emp_id = os.path.splitext(file)[0]
                self.db.append((emp_id, emb))

    def find(self, emb):
        best_id = None
        best_dist = 999

        for emp_id, ref in self.db:
            dist = 1 - np.dot(emb, ref)

            if dist < best_dist:
                best_dist = dist
                best_id = emp_id

        if best_dist <= COSINE_THRESHOLD:
            return best_id, best_dist

        return None, best_dist

# =====================================================
# TRACKER
# =====================================================

class FaceTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=10)

    def update(self, detections):
        raw = []

        for d in detections:
            raw.append((d["box"], d["score"], d["name"]))

        return self.tracker.update_tracks(raw)

# =====================================================
# STREAM
# =====================================================

class HikvisionStream:
    def __init__(self, config):
        self.cap = cv2.VideoCapture(config.stream_url)
        self.frame = None

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            ret, frame = self.cap.read()

            if ret:
                self.frame = frame

            time.sleep(0.01)

    def get_frame(self):
        if self.frame is None:
            return None

        return self.frame.copy()

# =====================================================
# LOGGING
# =====================================================

def submit_log(emp_id, cam):
    try:
        requests.post(
            f"{API_URL}/api/logs",
            headers={"x-api-key": API_KEY},
            json={
                "employeeId": emp_id,
                "logType": "entry",
                "cameraName": cam
            },
            timeout=5
        )
    except:
        pass

# =====================================================
# WORKER
# =====================================================

class CameraWorker:
    def __init__(self, config, db):
        self.name = config.name
        self.stream = HikvisionStream(config).start()

        self.app = build_face_app()
        self.db = db

        self.tracker = FaceTracker()

        self.logged = {}
        self.display_faces = []

        threading.Thread(target=self.ai_loop, daemon=True).start()

    def ai_loop(self):
        while True:
            frame = self.stream.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            faces = self.app.get(frame)

            detections = []

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)

                w = x2 - x1
                h = y2 - y1

                if w < MIN_FACE_PX:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # blur filter
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()

                if blur < 40:
                    continue

                emb = face.normed_embedding

                emp_id, dist = self.db.find(emb)

                if emp_id is None:
                    continue

                detections.append({
                    "box": [x1, y1, w, h],
                    "score": face.det_score,
                    "name": emp_id,
                    "distance": dist
                })

            tracks = self.tracker.update(detections)

            shown = []
            now = time.time()

            for t in tracks:
                if not t.is_confirmed():
                    continue

                ltrb = t.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                name = t.det_class

                dist = 0.25

                shown.append({
                    "employeeId": name,
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "distance": dist
                })

                if now - self.logged.get(name, 0) > COOLDOWN_SECS:
                    self.logged[name] = now

                    threading.Thread(
                        target=submit_log,
                        args=(name, self.name),
                        daemon=True
                    ).start()

            self.display_faces = shown

            time.sleep(DETECTION_DELAY)

# =====================================================
# DISPLAY
# =====================================================

def render(worker):
    frame = worker.stream.get_frame()

    if frame is None:
        return

    display = frame.copy()

    for p in worker.display_faces:
        x, y, w, h = p["box"]

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        conf = (1 - p["distance"]) * 100

        label = f"{p['employeeId']} {conf:.1f}%"

        cv2.putText(
            display,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.namedWindow(worker.name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(worker.name, 960, 540)

    cv2.imshow(worker.name, display)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    cams = load_cameras_from_env()

    if not cams:
        print("No cameras found in .env")
        exit()

    db_model = build_face_app()
    db = EmbeddingDatabase(db_model)

    workers = []

    for cam in cams:
        workers.append(CameraWorker(cam, db))

    while True:
        for w in workers:
            render(w)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()