# Facial Recognition Service — Local Setup (Multi-Camera Edition)

This service runs **on your local machine** (the one connected to the IP cameras).
It detects faces from one or more RTSP streams and logs attendance to the web app.

This version uses **YOLOv8 + FaceNet + DeepSORT** for faster, more accurate recognition
compared to the previous InsightFace-based implementation.

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

This installs dependencies and produces:

```
dist\BSU_FaceRec_Launcher.exe
```

Double-click the `.exe` to launch the control panel on any Windows machine.

> **Important:** The `.exe` is only the launcher GUI. The machine must still have Python
> installed with the facial recognition dependencies (`requirements.txt`) so the launcher
> can start the recognition service process.

**Files to copy alongside the `.exe` when distributing:**

| File / Folder | Purpose |
|---|---|
| `facial_recognition_service.py` | Recognition service (started by launcher) |
| `dashboard_server.py` | Local web dashboard for service control |
| `.env` | Camera IPs, credentials, API key |
| `registered_personnel/` | Enrolled personnel photos |
| `yolov8n-face.pt` | YOLO model weights |

---

## Manual Setup (without the launcher)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Optional — GPU acceleration (highly recommended if you have an NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
GPU acceleration makes recognition 5–10× faster. The service auto-detects CUDA and
uses fp16 + torch.compile when available.

### 2. Configure cameras and API

Copy the template and fill in your values:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
API_URL=https://your-app.replit.app
API_KEY=your-facial-recognition-api-key

# Camera 1
CAM1_NAME=Camera 1
CAM1_IP=192.168.1.64
CAM1_USER=admin
CAM1_PASS=your_password

# Camera 2 (add as many CAMn_* groups as needed)
CAM2_NAME=Camera 2
CAM2_IP=192.168.1.65
CAM2_USER=admin
CAM2_PASS=your_password
```

- `API_URL` — your published Replit app URL
- `API_KEY` — the value of `FACIAL_RECOGNITION_API_KEY` from your Replit Secrets
- `CAMn_*` — one block per camera. The system auto-discovers all defined cameras

> **Important:** Add `.env` to `.gitignore` — it contains passwords.

### 3. Set the API key in Replit

In your Replit project, add a Secret:

| Key | Value |
|-----|-------|
| `FACIAL_RECOGNITION_API_KEY` | any strong random string |

### 4. Run the service

```bash
python facial_recognition_service.py
```

On first run, the YOLOv8 face model (`yolov8n-face.pt`) is downloaded automatically
if not already present. FaceNet weights are also downloaded on first run via
`facenet-pytorch`. Subsequent startups use a local embedding cache for instant loading.

---

## Dashboard Server (Local Web Control Panel)

`dashboard_server.py` is a lightweight Flask server that runs locally alongside the
recognition service. It provides:

- Web dashboard at `http://localhost:5000`
- Start / stop the facial recognition service as a managed subprocess
- Live log streaming to the browser via SSE
- Camera configuration editor (reads and writes `.env`)
- Camera RTSP connection tester

```bash
pip install flask python-dotenv opencv-python
python dashboard_server.py
```

The dashboard server is also started automatically by `control_panel.py`.

---

## How it works

1. On startup, downloads all registered personnel photos from the web API into `registered_personnel/`
2. Builds a multi-angle embedding database (front, left, right, top views per person)
3. Connects to **every** configured camera in parallel (each on its own thread)
4. Runs **YOLOv8** face detection + **FaceNet** embedding + **DeepSORT** tracking continuously
5. When a face matches, POSTs the Employee ID + camera name to `/api/logs`
   (60-second cooldown per person, alternating TIME IN / TIME OUT)
6. A window per camera shows live feeds with face overlays and a status HUD
7. The Staff Monitoring page polls `/api/logs` and updates automatically

## What changed from the previous version

| Component | Before | After |
|-----------|--------|-------|
| Face detection | InsightFace (buffalo_l) | **YOLOv8** face detector |
| Face recognition | ArcFace (InsightFace) | **FaceNet** (VGGFace2, 512-dim) |
| Face tracking | Custom IOU tracker | **DeepSORT** (Kalman filter) |
| GPU acceleration | ONNX Runtime GPU | **PyTorch CUDA + fp16 + torch.compile** |
| Embedding cache | None | **Disk cache** (instant restarts after first run) |
| Re-entry speed | Full vote window every time | **Identity cache** (instant re-ID within 8s) |

## Multi-camera behavior

- Each camera runs independently — a network drop on one camera never affects the others
- Console output tags each event with the camera name:
  `[Logged] Name: Juan Dela Cruz | Camera: Camera 1 | Time: 08:01 AM | Type: TIME IN`
- One window per camera with live overlays; press `q` in any window to quit
- To switch from home network to campus network, just change the IP values in `.env`

## Department-based access control

- **Admin** users see all logs from all departments
- **User** accounts only see logs from their own department

## Notes

- All camera IPs must be reachable from the machine running this script
- Photos uploaded during registration are used directly — no manual file copying needed
- Duplicate logging within the cooldown window is automatically suppressed
- The embedding cache (`registered_personnel/embeddings_cache.pkl`) is invalidated
  automatically when personnel photos are added, removed, or changed
