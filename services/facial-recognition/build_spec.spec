# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for BSU Facial Recognition Control Panel
# Run: pyinstaller build_spec.spec

import sys
from pathlib import Path

block_cipher = None
HERE = Path(SPECPATH)

a = Analysis(
    [str(HERE / 'control_panel.py')],
    pathex=[str(HERE)],
    binaries=[],
    datas=[
        # Bundle the service script so the launcher can find it
        (str(HERE / 'facial_recognition_service.py'), '.'),
        # Include customtkinter theme assets
    ],
    hiddenimports=[
        'customtkinter',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'dotenv',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'webbrowser',
        'subprocess',
        'threading',
        'queue',
        'json',
        'pathlib',
        'socket',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML packages from the launcher exe
        'torch', 'torchvision', 'ultralytics', 'facenet_pytorch',
        'deep_sort_realtime', 'scipy', 'sklearn', 'matplotlib',
        'pandas', 'numpy.testing', 'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect customtkinter assets (themes, fonts, images)
try:
    import customtkinter
    ctk_path = Path(customtkinter.__file__).parent
    a.datas += [
        (str(f), str(f.relative_to(ctk_path.parent)))
        for f in ctk_path.rglob('*')
        if f.is_file()
    ]
except Exception as e:
    print(f"[WARN] Could not collect customtkinter assets: {e}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BSU_FaceRec_Launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # No terminal window (windowed app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='bsu_icon.ico',  # Uncomment if you have an .ico file
)
