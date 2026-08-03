"""
BSU Personnel Monitoring — Dashboard Server
============================================
Lightweight Flask web server that:
  · Serves the admin dashboard at http://localhost:5000
  · Starts / stops the facial recognition service as a managed subprocess
  · Streams service stdout to the browser via SSE
  · Reads and writes camera configuration in .env
  · Provides a REST endpoint for camera connection testing

Run directly:
  python dashboard_server.py

Or launched automatically by BSU_FaceRec_Launcher.exe (control_panel.py).

Requirements (in the same Python env as the recognition service):
  pip install flask python-dotenv opencv-python
"""

import atexit
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote as urlquote

try:
    from flask import Flask, Response, jsonify, render_template, request, stream_with_context
except ImportError:
    print("ERROR: Flask not installed.  Run: pip install flask")
    sys.exit(1)

try:
    from dotenv import dotenv_values
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR           = Path(__file__).parent.resolve()
ENV_FILE       = _DIR / ".env"
SERVICE_SCRIPT = _DIR / "facial_recognition_service.py"
STREAM_PORT    = 5001   # port the recognition service uses for MJPEG

# ── shared state ──────────────────────────────────────────────────────────────
_service_proc: Optional[subprocess.Popen] = None
_service_logs: list = []
_log_lock      = threading.Lock()
_MAX_LOGS      = 500

_log_subscribers: list = []   # list[queue.Queue]
_sub_lock = threading.Lock()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(_DIR / "templates"))
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_python() -> str:
    candidates = [
        str(_DIR / ".venv" / "Scripts" / "python.exe"),
        str(_DIR / "venv"  / "Scripts" / "python.exe"),
        sys.executable,
        shutil.which("python") or "",
        shutil.which("python3") or "",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return "python"


def _load_env() -> dict:
    if _HAS_DOTENV and ENV_FILE.exists():
        return dict(dotenv_values(ENV_FILE))
    env: dict = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _save_env(data: dict):
    lines = [f"{k}={v}\n" for k, v in data.items()]
    ENV_FILE.write_text("".join(lines), encoding="utf-8")


def _append_log(line: str):
    with _log_lock:
        _service_logs.append(line)
        if len(_service_logs) > _MAX_LOGS:
            _service_logs.pop(0)
    with _sub_lock:
        for q in list(_log_subscribers):
            try:
                q.put_nowait(line)
            except Exception:
                pass


def _stream_proc_output(proc: subprocess.Popen):
    try:
        for raw in proc.stdout:
            _append_log(raw.rstrip())
    except Exception:
        pass
    _append_log("[Dashboard] Recognition service process ended.")


def _stop_service_proc():
    global _service_proc
    if _service_proc and _service_proc.poll() is None:
        _service_proc.terminate()
        try:
            _service_proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            _service_proc.kill()


atexit.register(_stop_service_proc)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("index.html", stream_port=STREAM_PORT)


@app.route("/api/service/start", methods=["POST"])
def start_service():
    global _service_proc
    if _service_proc and _service_proc.poll() is None:
        return jsonify({"ok": False, "message": "Service is already running."})

    python = _find_python()
    if not Path(python).is_file() and not shutil.which(python):
        return jsonify({"ok": False, "message": f"Python executable not found: {python}"})
    if not SERVICE_SCRIPT.is_file():
        return jsonify({"ok": False, "message": f"Service script not found: {SERVICE_SCRIPT}"})

    try:
        _service_proc = subprocess.Popen(
            [python, str(SERVICE_SCRIPT), "--web-mode"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(_DIR),
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        threading.Thread(
            target=_stream_proc_output,
            args=(_service_proc,),
            daemon=True,
        ).start()
        _append_log("[Dashboard] Recognition service starting…")
        return jsonify({"ok": True, "message": "Service starting…"})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/service/stop", methods=["POST"])
def stop_service():
    global _service_proc
    if _service_proc and _service_proc.poll() is None:
        _stop_service_proc()
        _append_log("[Dashboard] Recognition service stopped.")
        return jsonify({"ok": True, "message": "Service stopped."})
    return jsonify({"ok": False, "message": "Service was not running."})


@app.route("/api/service/status")
def service_status():
    running = bool(_service_proc and _service_proc.poll() is None)
    return jsonify({"running": running})


@app.route("/api/service/logs")
def get_logs():
    with _log_lock:
        return jsonify({"logs": list(_service_logs)})


@app.route("/api/service/logs/stream")
def stream_logs():
    q: queue.Queue = queue.Queue(maxsize=200)
    with _sub_lock:
        _log_subscribers.append(q)

    def generate():
        with _log_lock:
            for line in _service_logs[-100:]:
                yield f"data: {json.dumps(line)}\n\n"
        try:
            while True:
                try:
                    line = q.get(timeout=25)
                    yield f"data: {json.dumps(line)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sub_lock:
                try:
                    _log_subscribers.remove(q)
                except ValueError:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    env = _load_env()
    cameras = []
    for i in range(1, 9):
        ip = env.get(f"CAM{i}_IP", "").strip()
        if not ip:
            break
        name = env.get(f"CAM{i}_NAME", f"Camera {i}").strip()
        cameras.append({
            "index":       i,
            "name":        name,
            "ip":          ip,
            "user":        env.get(f"CAM{i}_USER",        "admin").strip(),
            "port":        env.get(f"CAM{i}_PORT",        "554").strip(),
            "stream_path": env.get(f"CAM{i}_STREAM_PATH", "/Streaming/Channels/101").strip(),
            "stream_url":  f"http://localhost:{STREAM_PORT}/stream/{urlquote(name)}",
        })
    return jsonify({
        "cameras":    cameras,
        "api_url":    env.get("API_URL",    ""),
        "api_key":    env.get("API_KEY",    ""),
        "yolo_imgsz": env.get("YOLO_IMGSZ", "640"),
    })


@app.route("/api/cameras", methods=["POST"])
def save_cameras():
    data = request.get_json(force=True)
    env  = _load_env()

    # Clear previous per-camera keys
    for i in range(1, 9):
        for key in [f"CAM{i}_NAME", f"CAM{i}_IP", f"CAM{i}_USER",
                    f"CAM{i}_PASS", f"CAM{i}_PORT", f"CAM{i}_STREAM_PATH"]:
            env.pop(key, None)

    for cam in data.get("cameras", []):
        i = int(cam.get("index", 0))
        if not i:
            continue
        for field, env_key in [
            ("name",        f"CAM{i}_NAME"),
            ("ip",          f"CAM{i}_IP"),
            ("user",        f"CAM{i}_USER"),
            ("pass",        f"CAM{i}_PASS"),
            ("port",        f"CAM{i}_PORT"),
            ("stream_path", f"CAM{i}_STREAM_PATH"),
        ]:
            val = str(cam.get(field, "")).strip()
            if val:
                env[env_key] = val

    if "api_url"    in data: env["API_URL"]    = str(data["api_url"]).strip()
    if "api_key"    in data: env["API_KEY"]    = str(data["api_key"]).strip()
    if "yolo_imgsz" in data: env["YOLO_IMGSZ"] = str(data["yolo_imgsz"]).strip() or "640"

    _save_env(env)
    return jsonify({"ok": True})


@app.route("/api/cameras/test", methods=["POST"])
def test_camera():
    data = request.get_json(force=True)
    url  = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"ok": False, "message": "No URL provided."})
    try:
        import cv2
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.time() + 5.0
        ok = False
        while time.time() < deadline:
            ret, _ = cap.read()
            if ret:
                ok = True
                break
        cap.release()
        msg = "Connection successful." if ok else "Stream opened but no frames received within 5 s."
        return jsonify({"ok": ok, "message": msg})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BSU FaceRec Dashboard Server")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"[Dashboard] Starting BSU FaceRec Dashboard at http://localhost:{args.port}")
    print(f"[Dashboard] Service script : {SERVICE_SCRIPT}")
    print(f"[Dashboard] Stream port    : {STREAM_PORT}")
    print("[Dashboard] Press Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
