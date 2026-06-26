@echo off
title AI Interview Analyzer Pro - Setup
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Python 3.11 was not found. Install 64-bit Python 3.11 and add it to PATH.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_installation.py
pause
