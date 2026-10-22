@echo off
setlocal
cd /d "%~dp0"
title AI Misinformation Detector

echo ============================================
echo   AI-Generated Misinformation Detector
echo ============================================

echo.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [ERROR] Python was not found. Install Python 3.11+ and tick Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create virtual environment.
    pause
    exit /b 1
  )
)

if not exist ".venv\installed.flag" (
  echo Installing packages. First run may take a few minutes...
  echo This step does NOT upgrade pip, so it will not interrupt the setup.
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Package installation failed.
    echo Try running this file again. If it still fails, delete the .venv folder and rerun.
    pause
    exit /b 1
  )
  echo done> ".venv\installed.flag"
)

echo.
echo Starting Streamlit...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
