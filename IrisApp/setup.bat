@echo off
title Gesture Control — First Time Setup
echo.
echo   Gesture Control — First Time Setup
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Please install Python 3.10 or 3.11 from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo  [1/4] Python found.

echo  [2/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to create venv.
    pause
    exit /b 1
)

echo  [3/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo  [4/4] Installing dependencies (this may take several minutes)...
echo        mediapipe, opencv, faster-whisper, pyautogui, sounddevice,
echo        deep-translator, keyboard — please wait...
echo.

venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.9 pyautogui numpy sounddevice faster-whisper deep-translator keyboard

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Some packages failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo   Setup complete!
echo   You can now launch GestureControl.exe
echo.
pause
