@echo off
chcp 65001 >nul
set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [??] ???????????? ????.bat
    pause
    exit /b
)

"%VENV_PYTHON%" "%SCRIPT_DIR%yolo_cam.py"
if errorlevel 1 (
    echo.
    echo ???????????
    pause
)
