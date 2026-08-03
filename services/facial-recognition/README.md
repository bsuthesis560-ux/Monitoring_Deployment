# Facial Recognition Service — Local Setup (Multi-Camera Edition)

This service runs **on your local machine** (the one connected to the IP cameras).
It detects faces from one or more RTSP streams and logs attendance to the web app.

This version uses **YOLO11 + ArcFace (InsightFace buffalo_l) + DeepSORT** for fast,
lighting-robust recognition with walk-through optimisation.

---

## What's new in this version

| Feature | Details |
|---|---|
| **YOLO11 face detector** | `yolo11n-face.pt` — faster and more accurate than YOLOv8 |
| **ArcFace embeddings** | 512-dim via InsightFace `buffalo_l` (replaces FaceNet) |
| **4× lighting augmentation** | Each registered photo generates standard / bright / dark / CLAHE embeddings |
| **Walk-through detection** | Three-tier motion thresholds; instant confirm at 0.38 cosine distance |
| **EMA embedding smoothing** | Stable recognition across blurry mid-stride frames |

---

## Control Panel Launcher (Recommended)

Instead of running the service manually, use the graphical **Control Panel** which provides:

- One-click start / stop of the recognition service
- Automatic browser launch to the monitoring web app
- Visual camera configuration editor (no manual `.env` editing)
- Live RTSP camera preview with real-time feed display
- Scrollable service log output with save option

### Option A — Run directly with Python

```bash
pip install -r requirements_launcher.txt
python control_panel.py
```

### Option B — Build a standalone Windows `.exe`

1. Open a command prompt in this folder (`services\facial-recognition\`)
2. Run:

```bat
build_exe.bat
```

This produces `dist\BSU_FaceRec_Launcher.exe`.

> **Important:** The `.exe` is only the launcher GUI. The machine must still have Python
> installed with the facial recognition dependencies (`requirements.txt`).

**Files to copy alongside the `.exe` when distributing:**

| File / Folder | Purpose |
|---|---|
| `facial_recognition_service.py` | Recognition service (started by launcher) |
| `.env` | Camera IPs, credentials, API key |
| `registered_personnel/` | Enrolled personnel photos |
| `yolo11n-face.pt` | YOLO11 face model weights *(required)* |

---

## Manual Setup (without the launcher)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For NVIDIA GPU acceleration also install the CUDA-enabled PyTorch wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Configure the `.env` file

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `API_URL` | Full URL of your deployed web app |
| `API_KEY` | Must match `FACIAL_RECOGNITION_API_KEY` in the web app secrets |
| `CAM1_IP` | IP address of camera 1 |
| `CAM1_USER` | Camera username (usually `admin`) |
| `CAM1_PASS` | Camera password |
| `CAM1_PORT` | RTSP port (default `554`) |
| `CAM1_STREAM_PATH` | RTSP path (e.g. `/Streaming/Channels/101`) |

Add `CAM2_*`, `CAM3_*` … for additional cameras.

Optional:

| Variable | Default | Description |
|---|---|---|
| `YOLO_FACE_MODEL` | `yolo11n-face.pt` | Path to YOLO face model weights |
| `YOLO_IMGSZ` | `640` | YOLO input size (use `480` for ~40 % faster inference) |

### 3. Run the service

```bash
python facial_recognition_service.py
```

Press **q** in any camera window to quit.

---

## InsightFace model download (first run)

On first run InsightFace will automatically download the `buffalo_l` recognition pack
(~300 MB) to `~/.insightface/models/buffalo_l/`. An internet connection is required
for this one-time download. Subsequent runs use the cached model.

---

## API compatibility

The service communicates with the web app through two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/logs/personnel-photos` | GET | Download registered personnel photos |
| `/api/logs` | POST | Submit an attendance log entry |

Both require the `x-api-key` header set to your `API_KEY` value.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `yolo11n-face.pt not found` | Place `yolo11n-face.pt` next to `facial_recognition_service.py` |
| `insightface not installed` | Run `pip install insightface onnxruntime-gpu` |
| CUDA not detected | Install the correct PyTorch CUDA wheels (see Step 1) |
| Faces not recognised | Delete `registered_personnel/embeddings_cache.pkl` to force a full rebuild |
| Camera not connecting | Check IP, port, username, and password in `.env`; verify RTSP is enabled on camera |
