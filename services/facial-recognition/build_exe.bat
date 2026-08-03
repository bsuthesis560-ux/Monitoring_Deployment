@echo off
REM ============================================================
REM  BSU Personnel Monitoring — Build Control Panel .exe
REM  Run this script from the services\facial-recognition\ folder
REM ============================================================

echo.
echo  ============================================================
echo   BSU Facial Recognition Control Panel — EXE Builder
echo  ============================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ and ensure it is added to PATH.
    pause
    exit /b 1
)

echo [1/4] Installing launcher dependencies...
python -m pip install customtkinter Pillow opencv-python python-dotenv pyinstaller ^
    --quiet --disable-pip-version-check
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/4] Cleaning previous build artifacts...
IF EXIST build\   rmdir /s /q build
IF EXIST dist\    rmdir /s /q dist

echo.
echo [3/4] Running PyInstaller...
python -m PyInstaller build_spec.spec --noconfirm --clean
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] PyInstaller failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo [4/4] Packaging output...
IF EXIST dist\BSU_FaceRec_Launcher.exe (
    echo.
    echo  ============================================================
    echo   BUILD SUCCESSFUL
    echo   Output: dist\BSU_FaceRec_Launcher.exe
    echo  ============================================================
    echo.
    echo  IMPORTANT — copy these files alongside the .exe:
    echo    facial_recognition_service.py
    echo    .env  (camera and API credentials)
    echo    registered_personnel\  (folder with enrolled photos)
    echo    yolov8n-face.pt        (YOLO model weights)
    echo.
    echo  The Python facial recognition environment (torch, ultralytics,
    echo  facenet-pytorch, deep-sort-realtime) must be installed on the
    echo  machine that will run the launcher.
    echo.
) ELSE (
    echo [ERROR] EXE not found in dist\. Build may have failed silently.
)

pause
