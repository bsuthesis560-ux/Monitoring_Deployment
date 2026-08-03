#!/usr/bin/env python3
"""
BSU Personnel Monitoring — Facial Recognition Control Panel
===========================================================
A local desktop launcher for the facial recognition service.

Requirements:
    pip install customtkinter Pillow opencv-python python-dotenv requests

To build the Windows .exe (run build_exe.bat or manually):
    pip install pyinstaller
    pyinstaller build_spec.spec

Usage:
    python control_panel.py
    OR double-click the built dist/BSU_FaceRec_Launcher.exe
"""

import sys
import os
import re
import cv2
import time
import json
import queue
import socket
import threading
import webbrowser
import subprocess
import traceback
import shutil
from pathlib import Path
from datetime import datetime

# ── Pillow ─────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("WARNING: Pillow not installed. Camera preview disabled.")
    print("         Run: pip install Pillow")

# ── customtkinter ──────────────────────────────────────────────────────────
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    print("ERROR: customtkinter not installed.")
    print("       Run: pip install customtkinter")
    sys.exit(1)

# ── dotenv ─────────────────────────────────────────────────────────────────
try:
    from dotenv import dotenv_values, set_key
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

# ── Constants ──────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent.resolve()
ENV_FILE = _DIR / ".env"
SERVICE_SCRIPT = _DIR / "facial_recognition_service.py"
CONFIG_FILE = _DIR / "launcher_config.json"

BSU_RED  = "#CC0000"
BSU_BLUE = "#0A2463"
DARK_BG  = "#1a1a2e"
CARD_BG  = "#16213e"
ACCENT   = "#3d5af1"

PREVIEW_W = 640
PREVIEW_H = 360
MAX_LOG_LINES = 400


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _find_python() -> str:
    """Find a usable Python executable (prefers venv/conda in the service dir)."""
    candidates = [
        str(_DIR / ".venv" / "Scripts" / "python.exe"),
        str(_DIR / "venv" / "Scripts" / "python.exe"),
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
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _save_env(env: dict):
    lines = []
    for k, v in env.items():
        lines.append(f"{k}={v}\n")
    ENV_FILE.write_text("".join(lines), encoding="utf-8")


def _load_config() -> dict:
    defaults = {
        "web_url": "http://localhost:5000",
        "python_path": _find_python(),
        "preview_cam_index": 0,
        "preview_rtsp": "",
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def _save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _test_rtsp(url: str, timeout: float = 4.0) -> tuple:
    """Try to open an RTSP stream and read one frame. Returns (ok, message)."""
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.time() + timeout
        ok = False
        while time.time() < deadline:
            ret, _ = cap.read()
            if ret:
                ok = True
                break
        cap.release()
        return (True, "Connection successful.") if ok else (False, "Stream opened but no frames received.")
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Application Window
# ═══════════════════════════════════════════════════════════════════════════

class ControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("BSU Personnel Monitoring — Facial Recognition Control Panel")
        self.geometry("1200x760")
        self.minsize(1000, 680)
        self.configure(fg_color=DARK_BG)

        # App state
        self._service_proc:  subprocess.Popen | None = None
        self._preview_cap:   cv2.VideoCapture | None = None
        self._preview_active = False
        self._log_queue:     queue.Queue = queue.Queue()
        self._config = _load_config()
        self._env    = _load_env()

        # Build UI
        self._build_header()
        self._build_main()
        self._build_statusbar()

        # Start log consumer
        self._consume_logs()

        # Restore fields from saved data
        self._populate_fields()

        # Graceful close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Header ──────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=BSU_BLUE, height=64, corner_radius=0)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="BATANGAS STATE UNIVERSITY",
            font=ctk.CTkFont(family="Georgia", size=18, weight="bold"),
            text_color="white",
        ).pack(side="left", padx=20, pady=4)

        ctk.CTkLabel(
            hdr,
            text="THE NATIONAL ENGINEERING UNIVERSITY  ·  Personnel Monitoring System",
            font=ctk.CTkFont(size=10),
            text_color="#adc4ff",
        ).pack(side="left", padx=0)

        # Red accent bar
        red_bar = ctk.CTkFrame(self, fg_color=BSU_RED, height=6, corner_radius=0)
        red_bar.pack(fill="x", side="top")

    # ── Main content ────────────────────────────────────────────────────────

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=DARK_BG, corner_radius=0)
        main.pack(fill="both", expand=True, padx=0, pady=0)
        main.columnconfigure(0, weight=0, minsize=260)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_right_tabs(main)

    def _build_left_panel(self, parent):
        left = ctk.CTkFrame(parent, fg_color=CARD_BG, width=260, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew")
        left.pack_propagate(False)

        # Section title
        ctk.CTkLabel(
            left, text="SERVICE CONTROL",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#7090d0",
        ).pack(pady=(16, 6), padx=16, anchor="w")

        # Status indicator
        status_row = ctk.CTkFrame(left, fg_color="transparent")
        status_row.pack(fill="x", padx=16, pady=(0, 12))

        self._status_dot = ctk.CTkLabel(
            status_row, text="●", font=ctk.CTkFont(size=18),
            text_color="#555577", width=24,
        )
        self._status_dot.pack(side="left")

        self._status_label = ctk.CTkLabel(
            status_row, text="Service Stopped",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888",
        )
        self._status_label.pack(side="left", padx=6)

        # Start button
        self._btn_start = ctk.CTkButton(
            left, text="▶  Start Recognition Service",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1a6e2e", hover_color="#25913e",
            height=42, corner_radius=8,
            command=self._start_service,
        )
        self._btn_start.pack(fill="x", padx=16, pady=(0, 6))

        # Stop button
        self._btn_stop = ctk.CTkButton(
            left, text="■  Stop Service",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#6e1a1a", hover_color="#912525",
            height=42, corner_radius=8,
            state="disabled",
            command=self._stop_service,
        )
        self._btn_stop.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkFrame(left, fg_color="#2a2a4a", height=1).pack(fill="x", padx=16, pady=(0, 12))

        # Open browser
        ctk.CTkLabel(
            left, text="WEB INTERFACE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#7090d0",
        ).pack(pady=(0, 6), padx=16, anchor="w")

        url_frame = ctk.CTkFrame(left, fg_color="transparent")
        url_frame.pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkLabel(url_frame, text="URL:", font=ctk.CTkFont(size=11), text_color="#999").pack(side="left")
        self._url_entry = ctk.CTkEntry(url_frame, font=ctk.CTkFont(size=10), height=28)
        self._url_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._url_entry.insert(0, self._config.get("web_url", "http://localhost:5000"))

        self._btn_browser = ctk.CTkButton(
            left, text="🌐  Open in Browser",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=ACCENT, hover_color="#2d45d0",
            height=38, corner_radius=8,
            command=self._open_browser,
        )
        self._btn_browser.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkFrame(left, fg_color="#2a2a4a", height=1).pack(fill="x", padx=16, pady=(0, 12))

        # Python path
        ctk.CTkLabel(
            left, text="PYTHON EXECUTABLE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#7090d0",
        ).pack(pady=(0, 4), padx=16, anchor="w")

        self._py_entry = ctk.CTkEntry(left, font=ctk.CTkFont(size=9), height=28)
        self._py_entry.pack(fill="x", padx=16, pady=(0, 4))
        self._py_entry.insert(0, self._config.get("python_path", _find_python()))

        ctk.CTkButton(
            left, text="Auto-detect", height=26,
            font=ctk.CTkFont(size=10),
            fg_color="#333355", hover_color="#44447a",
            command=self._autodetect_python,
        ).pack(fill="x", padx=16, pady=(0, 16))

        # Spacer
        ctk.CTkFrame(left, fg_color="transparent").pack(fill="both", expand=True)

        # Version / info
        ctk.CTkLabel(
            left, text="BSU FaceRec Control Panel v1.0\nLipa Campus · CET Department",
            font=ctk.CTkFont(size=9),
            text_color="#444466",
            justify="center",
        ).pack(pady=10)

    def _build_right_tabs(self, parent):
        right = ctk.CTkFrame(parent, fg_color=DARK_BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=0)

        tabs = ctk.CTkTabview(right, fg_color=CARD_BG, segmented_button_fg_color=BSU_BLUE,
                               segmented_button_selected_color=ACCENT,
                               segmented_button_unselected_color=BSU_BLUE,
                               text_color="white")
        tabs.pack(fill="both", expand=True, padx=12, pady=12)

        tabs.add("📋  Service Log")
        tabs.add("📷  Camera Config")
        tabs.add("🎥  Live Preview")

        self._build_log_tab(tabs.tab("📋  Service Log"))
        self._build_camera_tab(tabs.tab("📷  Camera Config"))
        self._build_preview_tab(tabs.tab("🎥  Live Preview"))

    # ── Service Log Tab ─────────────────────────────────────────────────────

    def _build_log_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self._log_box = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Courier New", size=11),
            fg_color="#0d0d1a",
            text_color="#c8f0c8",
            corner_radius=6,
            wrap="word",
        )
        self._log_box.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 6))

        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew")

        ctk.CTkButton(
            btn_row, text="Clear Log", width=100, height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#333344", hover_color="#44445a",
            command=lambda: self._log_box.delete("1.0", "end"),
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Save Log…", width=100, height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#333344", hover_color="#44445a",
            command=self._save_log,
        ).pack(side="left")

        self._log("[System] Control panel ready.")
        self._log(f"[System] Service script: {SERVICE_SCRIPT}")
        self._log(f"[System] Python: {self._config.get('python_path', '(not set)')}")

    # ── Camera Config Tab ───────────────────────────────────────────────────

    def _build_camera_tab(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll.columnconfigure((0, 1, 2, 3), weight=1)

        self._cam_fields: list[dict] = []

        for i in range(1, 5):  # Support up to 4 cameras
            ctk.CTkLabel(
                scroll,
                text=f"Camera {i}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#9ab0ff",
            ).grid(row=(i - 1) * 4, column=0, columnspan=4, sticky="w", padx=8, pady=(16, 2))

            fields: dict = {}

            label_col = [
                ("Name",        f"CAM{i}_NAME",        f"Camera {i}",                         0, 1),
                ("IP Address",  f"CAM{i}_IP",          f"192.168.1.{63 + i}",                 2, 3),
                ("Username",    f"CAM{i}_USER",        "admin",                                0, 1),
                ("Password",    f"CAM{i}_PASS",        "",                                     2, 3),
                ("Port",        f"CAM{i}_PORT",        "554",                                  0, 1),
                ("Stream Path", f"CAM{i}_STREAM_PATH", "/Streaming/Channels/101",              2, 3),
            ]

            for li, (lbl, key, placeholder, c_label, c_entry) in enumerate(label_col):
                row = (i - 1) * 4 + 1 + li // 2
                ctk.CTkLabel(scroll, text=lbl + ":", font=ctk.CTkFont(size=11), text_color="#888").grid(
                    row=row, column=c_label, sticky="w", padx=(8, 2), pady=2)
                ent = ctk.CTkEntry(scroll, height=30, placeholder_text=placeholder,
                                   font=ctk.CTkFont(size=11), show="*" if "Pass" in lbl else "")
                ent.grid(row=row, column=c_entry, sticky="ew", padx=(0, 8), pady=2)
                val = self._env.get(key, "")
                if val:
                    ent.insert(0, val)
                fields[key] = ent

            # Enable / disable toggle
            enable_var = ctk.BooleanVar(value=bool(self._env.get(f"CAM{i}_IP", "")))
            fields[f"_enabled_{i}"] = enable_var
            ctk.CTkCheckBox(
                scroll,
                text=f"Enable Camera {i}",
                variable=enable_var,
                font=ctk.CTkFont(size=11),
                checkbox_height=18, checkbox_width=18,
            ).grid(row=(i - 1) * 4 + 3, column=0, sticky="w", padx=8, pady=(2, 0))

            # Test button
            def _make_test(idx=i, flds=fields):
                return lambda: self._test_camera(idx, flds)

            ctk.CTkButton(
                scroll, text=f"Test Camera {i}", height=28,
                font=ctk.CTkFont(size=11),
                fg_color="#1e3a5f", hover_color="#2a5280",
                command=_make_test(),
            ).grid(row=(i - 1) * 4 + 3, column=2, columnspan=2, sticky="e", padx=8, pady=(2, 0))

            self._cam_fields.append(fields)

        # ── API Config ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            scroll, text="API Connection",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#9ab0ff",
        ).grid(row=17, column=0, columnspan=4, sticky="w", padx=8, pady=(20, 2))

        ctk.CTkLabel(scroll, text="API URL:", font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=18, column=0, sticky="w", padx=(8, 2), pady=2)
        self._api_url_entry = ctk.CTkEntry(scroll, height=30, font=ctk.CTkFont(size=11))
        self._api_url_entry.grid(row=18, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=2)
        self._api_url_entry.insert(0, self._env.get("API_URL", ""))

        ctk.CTkLabel(scroll, text="API Key:", font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=19, column=0, sticky="w", padx=(8, 2), pady=2)
        self._api_key_entry = ctk.CTkEntry(scroll, height=30, font=ctk.CTkFont(size=11), show="*")
        self._api_key_entry.grid(row=19, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=2)
        self._api_key_entry.insert(0, self._env.get("API_KEY", ""))

        # ── YOLO Config ────────────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="YOLO Image Size:", font=ctk.CTkFont(size=11), text_color="#888").grid(
            row=20, column=0, sticky="w", padx=(8, 2), pady=2)
        self._yolo_entry = ctk.CTkEntry(scroll, height=30, font=ctk.CTkFont(size=11), width=80)
        self._yolo_entry.grid(row=20, column=1, sticky="w", padx=(0, 8), pady=2)
        self._yolo_entry.insert(0, self._env.get("YOLO_IMGSZ", "640"))

        ctk.CTkLabel(
            scroll, text="(set 480 for ~40% faster YOLO; lower recall on small faces)",
            font=ctk.CTkFont(size=9), text_color="#555577",
        ).grid(row=20, column=2, columnspan=2, sticky="w", pady=2)

        # Save button
        ctk.CTkButton(
            scroll, text="💾  Save Configuration", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1a5c3a", hover_color="#237a4e",
            command=self._save_camera_config,
        ).grid(row=21, column=0, columnspan=4, sticky="ew", padx=8, pady=16)

        self._cam_status_label = ctk.CTkLabel(
            scroll, text="", font=ctk.CTkFont(size=11), text_color="#4caf50",
        )
        self._cam_status_label.grid(row=22, column=0, columnspan=4, pady=(0, 8))

    # ── Live Preview Tab ────────────────────────────────────────────────────

    def _build_preview_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # Controls row
        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(ctrl, text="RTSP URL:", font=ctk.CTkFont(size=11), text_color="#999").pack(side="left", padx=(0, 6))
        self._preview_url = ctk.CTkEntry(ctrl, height=30, font=ctk.CTkFont(size=11), width=380,
                                          placeholder_text="rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101")
        self._preview_url.pack(side="left", padx=(0, 6))
        self._preview_url.insert(0, self._config.get("preview_rtsp", ""))

        self._btn_preview_start = ctk.CTkButton(
            ctrl, text="▶ Start Preview", width=120, height=30,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1a6e2e", hover_color="#25913e",
            command=self._start_preview,
        )
        self._btn_preview_start.pack(side="left", padx=(0, 4))

        self._btn_preview_stop = ctk.CTkButton(
            ctrl, text="■ Stop", width=80, height=30,
            font=ctk.CTkFont(size=11),
            fg_color="#6e1a1a", hover_color="#912525",
            state="disabled",
            command=self._stop_preview,
        )
        self._btn_preview_stop.pack(side="left")

        # Camera canvas
        canvas_frame = ctk.CTkFrame(parent, fg_color="#0a0a1a", corner_radius=8)
        canvas_frame.grid(row=1, column=0, sticky="nsew")

        self._canvas = ctk.CTkCanvas(canvas_frame, bg="#0a0a1a", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        self._canvas_placeholder()

        # Preview stats bar
        self._preview_info = ctk.CTkLabel(
            parent, text="Preview stopped",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color="#444466",
        )
        self._preview_info.grid(row=2, column=0, sticky="w", pady=(4, 0))

    # ── Status bar ──────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color="#0d0d1a", height=26, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._statusbar_label = ctk.CTkLabel(
            bar, text="Ready",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color="#445566",
            anchor="w",
        )
        self._statusbar_label.pack(side="left", padx=12, fill="y")

    def _set_status(self, text: str):
        self._statusbar_label.configure(text=f"[{_ts()}] {text}")

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log(self, text: str):
        self._log_queue.put(f"[{_ts()}] {text}")

    def _consume_logs(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log_box.insert("end", msg + "\n")
                self._log_box.see("end")
                # Trim old lines
                line_count = int(self._log_box.index("end-1c").split(".")[0])
                if line_count > MAX_LOG_LINES:
                    self._log_box.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        except queue.Empty:
            pass
        self.after(120, self._consume_logs)

    def _save_log(self):
        import tkinter.filedialog as fd
        path = fd.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")],
            initialfile=f"bsu_facerec_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        if path:
            content = self._log_box.get("1.0", "end")
            Path(path).write_text(content, encoding="utf-8")
            self._set_status(f"Log saved → {path}")

    # ── Service management ───────────────────────────────────────────────────

    def _start_service(self):
        if self._service_proc and self._service_proc.poll() is None:
            self._log("[Warning] Service is already running.")
            return

        python = self._py_entry.get().strip() or _find_python()
        if not Path(python).is_file() and not shutil.which(python):
            self._log(f"[Error] Python not found: {python}")
            self._set_status("Error: Python executable not found.")
            return

        if not SERVICE_SCRIPT.is_file():
            self._log(f"[Error] Service script not found: {SERVICE_SCRIPT}")
            return

        self._log(f"[Service] Starting → {python} {SERVICE_SCRIPT}")
        self._set_status("Starting facial recognition service…")

        try:
            self._service_proc = subprocess.Popen(
                [python, str(SERVICE_SCRIPT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(_DIR),
                bufsize=1,
            )
            self._btn_start.configure(state="disabled")
            self._btn_stop.configure(state="normal")
            self._status_dot.configure(text_color="#22cc44")
            self._status_label.configure(text="Service Running", text_color="#22cc44")
            self._set_status("Service started.")

            # Read stdout in background thread
            threading.Thread(
                target=self._stream_output,
                args=(self._service_proc,),
                daemon=True,
            ).start()

            # Monitor process exit
            self.after(1000, self._check_service)

            # Auto-open browser
            url = self._url_entry.get().strip()
            if url:
                self.after(3000, lambda: webbrowser.open(url))
                self._log(f"[Browser] Will open {url} in 3 s…")

        except Exception as e:
            self._log(f"[Error] Failed to start service: {e}")
            self._set_status("Failed to start service.")

    def _stream_output(self, proc: subprocess.Popen):
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._log_queue.put(f"[{_ts()}] [SVC] {line}")
        except Exception:
            pass

    def _check_service(self):
        if self._service_proc is None:
            return
        ret = self._service_proc.poll()
        if ret is not None:
            self._log(f"[Service] Exited with code {ret}")
            self._set_status(f"Service exited (code {ret}).")
            self._btn_start.configure(state="normal")
            self._btn_stop.configure(state="disabled")
            self._status_dot.configure(text_color="#555577")
            self._status_label.configure(text="Service Stopped", text_color="#888")
        else:
            self.after(1000, self._check_service)

    def _stop_service(self):
        if self._service_proc and self._service_proc.poll() is None:
            self._log("[Service] Sending termination signal…")
            self._service_proc.terminate()
            try:
                self._service_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._service_proc.kill()
            self._log("[Service] Stopped.")
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._status_dot.configure(text_color="#555577")
        self._status_label.configure(text="Service Stopped", text_color="#888")
        self._set_status("Service stopped.")

    # ── Browser ─────────────────────────────────────────────────────────────

    def _open_browser(self):
        url = self._url_entry.get().strip() or "http://localhost:5000"
        self._log(f"[Browser] Opening {url}")
        webbrowser.open(url)
        self._set_status(f"Opened browser → {url}")

    # ── Camera config ────────────────────────────────────────────────────────

    def _populate_fields(self):
        pass  # Fields are populated in _build_camera_tab using _env

    def _save_camera_config(self):
        env = dict(self._env)  # preserve existing keys

        for i, fields in enumerate(self._cam_fields, 1):
            enabled = fields.get(f"_enabled_{i}")
            if isinstance(enabled, ctk.BooleanVar) and not enabled.get():
                # Clear this camera's keys
                for key in [f"CAM{i}_NAME", f"CAM{i}_IP", f"CAM{i}_USER",
                             f"CAM{i}_PASS", f"CAM{i}_PORT", f"CAM{i}_STREAM_PATH"]:
                    env.pop(key, None)
                continue
            for key, widget in fields.items():
                if key.startswith("_"):
                    continue
                val = widget.get().strip()
                if val:
                    env[key] = val
                else:
                    env.pop(key, None)

        env["API_URL"]    = self._api_url_entry.get().strip()
        env["API_KEY"]    = self._api_key_entry.get().strip()
        env["YOLO_IMGSZ"] = self._yolo_entry.get().strip() or "640"

        _save_env(env)
        self._env = env

        # Save web URL + python path to launcher config
        self._config["web_url"]     = self._url_entry.get().strip()
        self._config["python_path"] = self._py_entry.get().strip()
        self._config["preview_rtsp"] = self._preview_url.get().strip()
        _save_config(self._config)

        self._cam_status_label.configure(
            text=f"✓ Configuration saved to {ENV_FILE.name}",
            text_color="#4caf50",
        )
        self._log(f"[Config] Saved camera configuration to {ENV_FILE}")
        self._set_status("Configuration saved.")
        self.after(4000, lambda: self._cam_status_label.configure(text=""))

    def _test_camera(self, idx: int, fields: dict):
        ip = fields.get(f"CAM{idx}_IP")
        user = fields.get(f"CAM{idx}_USER")
        pwd = fields.get(f"CAM{idx}_PASS")
        port = fields.get(f"CAM{idx}_PORT")
        path = fields.get(f"CAM{idx}_STREAM_PATH")

        if not ip or not ip.get().strip():
            self._cam_status_label.configure(text=f"Camera {idx}: No IP set.", text_color="#f44336")
            return

        ip_v   = ip.get().strip()
        user_v = user.get().strip() if user else "admin"
        pwd_v  = pwd.get().strip() if pwd else ""
        port_v = port.get().strip() if port else "554"
        path_v = path.get().strip() if path else "/Streaming/Channels/101"

        from urllib.parse import quote
        url = f"rtsp://{quote(user_v, safe='')}:{quote(pwd_v, safe='')}@{ip_v}:{port_v}{path_v}"

        self._cam_status_label.configure(text=f"Testing Camera {idx}…", text_color="#ff9800")
        self._log(f"[Test] Camera {idx}: connecting to rtsp://****@{ip_v}:{port_v}{path_v}")

        def run_test():
            ok, msg = _test_rtsp(url)
            self._log_queue.put(f"[{_ts()}] [Test] Camera {idx}: {'OK' if ok else 'FAIL'} — {msg}")
            self.after(0, lambda: self._cam_status_label.configure(
                text=f"Camera {idx}: {'✓ ' if ok else '✗ '}{msg}",
                text_color="#4caf50" if ok else "#f44336",
            ))

        threading.Thread(target=run_test, daemon=True).start()

    # ── Auto-detect Python ───────────────────────────────────────────────────

    def _autodetect_python(self):
        found = _find_python()
        self._py_entry.delete(0, "end")
        self._py_entry.insert(0, found)
        self._log(f"[Config] Python auto-detected: {found}")
        self._set_status(f"Python: {found}")

    # ── Live Preview ─────────────────────────────────────────────────────────

    def _canvas_placeholder(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or PREVIEW_W
        h = self._canvas.winfo_height() or PREVIEW_H
        self._canvas.create_rectangle(0, 0, w, h, fill="#0a0a1a", outline="")
        self._canvas.create_text(
            w // 2, h // 2,
            text="📷  No camera feed\nClick ▶ Start Preview to connect",
            fill="#333355", font=("Courier New", 14), justify="center",
        )

    def _start_preview(self):
        if self._preview_active:
            return

        rtsp = self._preview_url.get().strip()
        if not rtsp:
            self._log("[Preview] No RTSP URL set. Enter a URL and try again.")
            return

        self._log(f"[Preview] Connecting to: {rtsp.split('@')[-1] if '@' in rtsp else rtsp}")
        self._preview_active = True
        self._btn_preview_start.configure(state="disabled")
        self._btn_preview_stop.configure(state="normal")

        threading.Thread(target=self._preview_thread, args=(rtsp,), daemon=True).start()

    def _preview_thread(self, url: str):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._preview_cap = cap

        frame_count = 0
        fps_time = time.time()
        fps = 0

        while self._preview_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = now

            # Resize for display
            try:
                cw = self._canvas.winfo_width()  or PREVIEW_W
                ch = self._canvas.winfo_height() or PREVIEW_H
                if cw < 10:
                    cw, ch = PREVIEW_W, PREVIEW_H
                h, w = frame.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (nw, nh))
                rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img     = Image.fromarray(rgb)
                photo   = ImageTk.PhotoImage(img)

                def update_canvas(p=photo, nw=nw, nh=nh, cw=cw, ch=ch, fps=fps):
                    self._canvas.delete("all")
                    self._canvas.create_image(
                        cw // 2, ch // 2, anchor="center", image=p,
                    )
                    self._canvas._photo = p  # prevent GC
                    self._preview_info.configure(
                        text=f"Feed: {w}×{h}  →  display {nw}×{nh}  |  {fps} FPS",
                        text_color="#445566",
                    )

                self.after(0, update_canvas)
            except Exception:
                pass

            time.sleep(0.033)  # ~30 fps

        cap.release()
        self._preview_cap = None
        self.after(0, self._canvas_placeholder)
        self.after(0, lambda: self._preview_info.configure(text="Preview stopped", text_color="#444466"))

    def _stop_preview(self):
        self._preview_active = False
        self._btn_preview_start.configure(state="normal")
        self._btn_preview_stop.configure(state="disabled")
        self._log("[Preview] Stopped.")
        self._set_status("Camera preview stopped.")

    # ── Close ────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._preview_active = False
        if self._service_proc and self._service_proc.poll() is None:
            import tkinter.messagebox as mb
            if mb.askyesno("Exit", "The recognition service is running. Stop it and exit?"):
                self._stop_service()
            else:
                return
        # Save config on exit
        self._config["web_url"]      = self._url_entry.get().strip()
        self._config["python_path"]  = self._py_entry.get().strip()
        self._config["preview_rtsp"] = self._preview_url.get().strip()
        _save_config(self._config)
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ControlPanel()
    app.mainloop()
