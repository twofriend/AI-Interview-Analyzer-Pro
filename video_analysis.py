from __future__ import annotations

from dataclasses import dataclass

import cv2
import pandas as pd


@dataclass
class EmotionResult:
    timeline: pd.DataFrame
    composure_score: float
    dominant_emotion: str
    frames_analyzed: int
    error: str = ""


@dataclass
class GazeResult:
    forward_gaze_score: float
    face_visibility_score: float
    frames_sampled: int
    method: str
    error: str = ""


def analyze_emotions(video_path: str, sample_every_seconds: float = 1.5, max_samples: int = 80) -> EmotionResult:
    """Analyze sampled frames with DeepFace's pretrained emotion model."""
    try:
        from deepface import DeepFace
    except Exception:
        return EmotionResult(pd.DataFrame(), 50.0, "Unavailable", 0, "DeepFace could not be imported.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return EmotionResult(pd.DataFrame(), 50.0, "Unavailable", 0, "Video could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * sample_every_seconds))
    rows, frame_index, analyzed = [], 0, 0

    while analyzed < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]
                emotions = result.get("emotion", {})
                dominant = str(result.get("dominant_emotion", "unknown"))
                confidence = float(emotions.get(dominant, 0.0))
                rows.append({
                    "time_sec": round(frame_index / fps, 2),
                    "emotion": dominant,
                    "confidence": round(confidence, 2),
                })
                analyzed += 1
            except Exception:
                pass
        frame_index += 1

    cap.release()
    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return EmotionResult(timeline, 50.0, "Unavailable", 0, "No usable face frames were analyzed.")

    dominant = str(timeline["emotion"].mode().iloc[0])
    calm_ratio = float(timeline["emotion"].isin({"neutral", "happy"}).mean())
    avg_conf = float(timeline["confidence"].mean()) / 100.0
    score = round(max(0.0, min(100.0, (0.75 * calm_ratio + 0.25 * avg_conf) * 100)), 2)
    return EmotionResult(timeline, score, dominant, len(timeline))


def _ratio(value: float, start: float, end: float) -> float:
    width = max(abs(end - start), 1e-6)
    return abs(value - start) / width


def analyze_gaze(video_path: str, sample_every_seconds: float = 0.75, max_samples: int = 160) -> GazeResult:
    """Estimate camera-facing attention using MediaPipe iris and head alignment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return GazeResult(50.0, 0.0, 0, "Unavailable", "Video could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * sample_every_seconds))
    total = face_frames = forward_frames = frame_index = 0

    try:
        import mediapipe as mp
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        while total < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                total += 1
                result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_face_landmarks:
                    face_frames += 1
                    lm = result.multi_face_landmarks[0].landmark
                    right_ratio = _ratio(lm[468].x, lm[33].x, lm[133].x)
                    left_ratio = _ratio(lm[473].x, lm[362].x, lm[263].x)
                    xs = [point.x for point in lm[:468]]
                    face_center_x = (min(xs) + max(xs)) / 2
                    head_aligned = abs(lm[1].x - face_center_x) < 0.08
                    iris_centered = 0.25 <= right_ratio <= 0.75 and 0.25 <= left_ratio <= 0.75
                    if iris_centered and head_aligned:
                        forward_frames += 1
            frame_index += 1
        mesh.close()
        method = "MediaPipe iris/head-alignment proxy"
    except Exception:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        while total < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                total += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.2, 5)
                if len(faces):
                    face_frames += 1
                    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                    if abs((x + w / 2) - gray.shape[1] / 2) < gray.shape[1] * 0.2:
                        forward_frames += 1
            frame_index += 1
        method = "OpenCV frontal-face fallback"

    cap.release()
    if total == 0:
        return GazeResult(50.0, 0.0, 0, method, "No frames were sampled.")
    visibility = round(face_frames / total * 100, 2)
    forward = round(forward_frames / face_frames * 100, 2) if face_frames else 0.0
    return GazeResult(forward, visibility, total, method)
