@echo off
echo ==========================================
echo AI Misinformation Media Detector - Training
echo ==========================================
echo.
echo 1. NEWS vs NOT-NEWS
if exist data\input_type.csv (
  .venv\Scripts\python train_input_type.py --data data\input_type.csv
) else (
  echo [SKIP] data\input_type.csv not found
)
echo.
echo 2. AI TEXT vs HUMAN
if exist data\ai_text.csv (
  .venv\Scripts\python train_ai_text.py --data data\ai_text.csv --max-per-class 10000
) else (
  echo [SKIP] data\ai_text.csv not found
)
echo.
echo 3. IMAGE REAL vs AI
if exist data\image\train\REAL (
  .venv\Scripts\python train_image.py --data data\image --epochs 8 --batch-size 32
) else (
  echo [SKIP] image dataset folders not found
)
echo.
echo Training finished. Start app with:
echo .venv\Scripts\python -m streamlit run app.py
pause
