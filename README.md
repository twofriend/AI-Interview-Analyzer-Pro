# AI Interview Analyzer Pro — Academic Version

## Included functions

- Streamlit video upload and dashboard
- DeepFace facial-emotion timeline
- MediaPipe camera-facing attention proxy
- Faster-Whisper local speech transcription
- Librosa voice-delivery measurements
- Sentiment, filler-word, vocabulary, and keyword analysis
- Weighted interview-practice score
- CSV, transcript, and detailed PDF downloads

The application intentionally avoids automated hire/reject decisions.

## Recommended environment

- Windows 10/11
- 64-bit Python 3.11
- At least 8 GB RAM
- Internet for the first run so pretrained model files can download
- A clear 30–90 second MP4 for the first test

## Easy setup

1. Extract the ZIP.
2. Double-click `setup_windows.bat`.
3. After installation, double-click `run_app.bat`.

## Manual commands

```cmd
cd /d "D:\DATA RECOVERY\AI_Interview_Analyzer_Pro_Academic"
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_installation.py
python -m streamlit run app.py
```

## First-run notes

- Faster-Whisper downloads the selected `tiny` or `base` speech model on first use.
- DeepFace downloads its pretrained emotion model on first use.
- Begin with the `tiny` transcription model and a short video.

## Responsible use

This is an academic interview-practice tool. It must not be the sole basis for hiring, rejection,
grading, diagnosis, or claims about honesty, personality, intelligence, or competence.
