import importlib.util

packages = {
    "Streamlit": "streamlit",
    "DeepFace": "deepface",
    "MediaPipe": "mediapipe",
    "Faster-Whisper": "faster_whisper",
    "Librosa": "librosa",
    "ReportLab": "reportlab",
}
print("\nAI Interview Analyzer Pro — package check\n")
for name, module in packages.items():
    print(f"{'[OK]' if importlib.util.find_spec(module) else '[MISSING]'} {name}")
print("\nRun with: python -m streamlit run app.py")
