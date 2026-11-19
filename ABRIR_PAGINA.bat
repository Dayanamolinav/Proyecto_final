@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py
) else (
  python -m streamlit run app/streamlit_app.py
)
if errorlevel 1 (
  echo.
  echo Instala las dependencias desde esta carpeta:
  echo python -m pip install -r requirements.txt
  echo.
  pause
)
