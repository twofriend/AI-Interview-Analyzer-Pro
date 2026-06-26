@echo off
title AI Interview Analyzer Pro
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run setup_windows.bat.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m streamlit run app.py
pause
