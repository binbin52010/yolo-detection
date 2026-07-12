@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\pythonw.exe"
set "YOLO_START_HIDDEN=0"
set "YOLO_DISABLE_TUNNEL=0"

if not exist "%VENV_PYTHON%" (
    echo ERROR: venv\Scripts\pythonw.exe was not found.
    echo Run the dependency installer first.
    pause
    exit /b 1
)

start "" "%VENV_PYTHON%" "%SCRIPT_DIR%yolo_cam.py"
exit /b 0
