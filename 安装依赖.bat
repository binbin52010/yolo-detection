@echo off
chcp 65001 >nul
echo ========================================
echo    YOLOv8 ?????? - ????
echo ========================================
echo.

set SCRIPT_DIR=%~dp0

python --version >nul 2>&1
if errorlevel 1 (
    echo [??] ???? Python????? Python 3.8+
    pause
    exit /b
)
echo Python ???:
python --version

:: ??????
echo.
echo [?? 1/4] ??????...
if exist "%SCRIPT_DIR%venv" (
    echo ??????????
) else (
    python -m venv "%SCRIPT_DIR%venv"
    if errorlevel 1 (
        echo [??] ????????
        pause
        exit /b
    )
    echo ????????
)

set VENV_PIP=%SCRIPT_DIR%venv\Scripts\pip.exe
set VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe

:: ?? PyTorch CUDA ?
echo.
echo [?? 2/4] ?? PyTorch CUDA?? 2.5GB???????????...
"%VENV_PIP%" install torch torchvision -i https://download.pytorch.org/whl/cu124 --trusted-host download.pytorch.org
if errorlevel 1 (
    echo [??] CUDA ???????? CPU ?...
    "%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org
    if errorlevel 1 (
        "%VENV_PIP%" install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    )
)

:: ??????
echo.
echo [?? 3/4] ???????...
"%VENV_PIP%" install ultralytics opencv-python pillow -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

:: ????
echo.
echo [?? 4/4] ??????...
if not exist "%SCRIPT_DIR%models\yolov8n.pt" (
    if not exist "%SCRIPT_DIR%models" mkdir "%SCRIPT_DIR%models"
    echo ?? YOLOv8 ??...
    "%VENV_PYTHON%" -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    if exist "%USERPROFILE%\.cache\ultralytics\yolov8n.pt" (
        copy "%USERPROFILE%\.cache\ultralytics\yolov8n.pt" "%SCRIPT_DIR%models\yolov8n.pt"
    )
)

echo.
echo ========================================
echo    ??????? ????.bat ??
echo    ????????? venv ????
echo ========================================
pause
