from __future__ import annotations

import html
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Interview Analyzer Pro — Academic",
    page_icon="🎙️",
    layout="wide",
)


@dataclass
class EmotionResult:
    timeline: pd.DataFrame
    score: float
    dominant: str
    frames: int
    error: str = ""


@dataclass
class GazeResult:
    score: float
    face_visibility: float
    frames: int
    method: str
    error: str = ""


@dataclass
class TranscriptResult:
    text: str
    language: str
    probability: float
    segments: list[dict[str, Any]]
    error: str = ""


@dataclass
class VoiceResult:
    duration_sec: float
    speaking_rate_wpm: float
    mean_rms: float
    median_pitch_hz: float
    silence_ratio: float
    score: float
    interpretation: str
    error: str = ""


@dataclass
class TextResult:
    sentiment: str
    sentiment_score: float
    polarity: float
    word_count: int
    filler_count: int
    filler_rate: float
    lexical_diversity: float
    keyword_coverage: float
    matched_keywords: list[str]
    missing_keywords: list[str]


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.read())
    temp.flush()
    temp.close()
    return Path(temp.name)


def extract_audio(video_path: Path) -> Path:
    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is unavailable. Install imageio-ffmpeg.") from exc

    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed. Confirm that the video contains audio.")

    return output_path


def analyze_emotions(video_path: Path, interval_seconds: float = 1.5, max_samples: int = 40) -> EmotionResult:
    try:
        from deepface import DeepFace
    except Exception as exc:
        return EmotionResult(pd.DataFrame(), 50.0, "Unavailable", 0, f"DeepFace unavailable: {exc}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return EmotionResult(pd.DataFrame(), 50.0, "Unavailable", 0, "Video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    step = max(1, int(fps * interval_seconds))

    rows: list[dict[str, Any]] = []
    frame_number = 0

    while len(rows) < max_samples:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_number % step == 0:
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]

                emotion_values = result.get("emotion", {})
                dominant = str(result.get("dominant_emotion", "unknown"))
                confidence = float(emotion_values.get(dominant, 0.0))
                rows.append(
                    {
                        "time_sec": round(frame_number / fps, 2),
                        "emotion": dominant,
                        "confidence": round(confidence, 2),
                    }
                )
            except Exception:
                pass

        frame_number += 1

    capture.release()
    timeline = pd.DataFrame(rows)

    if timeline.empty:
        return EmotionResult(
            timeline,
            50.0,
            "Unavailable",
            0,
            "No usable face frames were analyzed. Use a clear front-facing video.",
        )

    dominant = str(timeline["emotion"].mode().iloc[0])
    calm_ratio = float(timeline["emotion"].isin({"neutral", "happy"}).mean())
    confidence_ratio = float(timeline["confidence"].mean()) / 100
    score = round(max(0.0, min(100.0, (0.75 * calm_ratio + 0.25 * confidence_ratio) * 100)), 2)

    return EmotionResult(timeline, score, dominant, len(timeline))


def analyze_gaze(video_path: Path, interval_seconds: float = 0.75, max_samples: int = 80) -> GazeResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return GazeResult(50.0, 0.0, 0, "Unavailable", "Video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    step = max(1, int(fps * interval_seconds))

    total = 0
    visible = 0
    forward = 0
    frame_number = 0

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while total < max_samples:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_number % step == 0:
            total += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces):
                visible += 1
                x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
                face_center = x + width / 2
                frame_center = gray.shape[1] / 2
                if abs(face_center - frame_center) < gray.shape[1] * 0.20:
                    forward += 1

        frame_number += 1

    capture.release()

    if total == 0:
        return GazeResult(50.0, 0.0, 0, "OpenCV face-alignment proxy", "No frames were sampled.")

    visibility = round(visible / total * 100, 2)
    score = round(forward / visible * 100, 2) if visible else 0.0
    return GazeResult(score, visibility, total, "OpenCV face-alignment proxy")


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: Path, model_size: str) -> TranscriptResult:
    try:
        model = load_whisper_model(model_size)
        generator, information = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )

        segments: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for segment in generator:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
                segments.append(
                    {
                        "start_sec": round(float(segment.start), 2),
                        "end_sec": round(float(segment.end), 2),
                        "text": text,
                    }
                )

        return TranscriptResult(
            text=" ".join(text_parts),
            language=str(getattr(information, "language", "unknown")),
            probability=round(float(getattr(information, "language_probability", 0.0)), 3),
            segments=segments,
        )
    except Exception as exc:
        return TranscriptResult("", "unknown", 0.0, [], f"Transcription unavailable: {exc}")


FILLERS = ["um", "uh", "erm", "hmm", "you know", "like", "basically", "actually", "sort of", "kind of"]


def analyze_text(text: str, keywords: list[str]) -> TextResult:
    lower_text = text.lower().strip()
    words = re.findall(r"\b[\w'-]+\b", lower_text)
    word_count = len(words)

    filler_count = sum(
        len(re.findall(rf"\b{re.escape(filler)}\b", lower_text))
        for filler in FILLERS
    )
    filler_rate = filler_count / word_count * 100 if word_count else 0.0
    lexical_diversity = len(set(words)) / word_count * 100 if word_count else 0.0

    try:
        from textblob import TextBlob
        polarity = float(TextBlob(text).sentiment.polarity) if text else 0.0
    except Exception:
        polarity = 0.0

    sentiment_score = (polarity + 1) / 2 * 100
    if sentiment_score >= 65:
        sentiment = "Positive"
    elif sentiment_score <= 40:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    normalized_keywords = [keyword.strip().lower() for keyword in keywords if keyword.strip()]
    matched = [keyword for keyword in normalized_keywords if keyword in lower_text]
    missing = [keyword for keyword in normalized_keywords if keyword not in lower_text]
    coverage = len(matched) / len(normalized_keywords) * 100 if normalized_keywords else 0.0

    return TextResult(
        sentiment=sentiment,
        sentiment_score=round(sentiment_score, 2),
        polarity=round(polarity, 3),
        word_count=word_count,
        filler_count=filler_count,
        filler_rate=round(filler_rate, 2),
        lexical_diversity=round(lexical_diversity, 2),
        keyword_coverage=round(coverage, 2),
        matched_keywords=matched,
        missing_keywords=missing,
    )


def analyze_voice(audio_path: Path, word_count: int) -> VoiceResult:
    try:
        import librosa

        signal, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        duration = float(librosa.get_duration(y=signal, sr=sample_rate))
        if duration <= 0 or signal.size == 0:
            raise ValueError("No usable audio was found.")

        mean_rms = float(np.mean(librosa.feature.rms(y=signal)[0]))

        try:
            pitch = librosa.yin(signal, fmin=65, fmax=400, sr=sample_rate)
            finite_pitch = pitch[np.isfinite(pitch)]
            median_pitch = float(np.median(finite_pitch)) if finite_pitch.size else 0.0
        except Exception:
            median_pitch = 0.0

        intervals = librosa.effects.split(signal, top_db=30)
        active_samples = sum(int(end - start) for start, end in intervals)
        silence_ratio = max(0.0, min(1.0, 1 - active_samples / max(len(signal), 1)))
        speaking_rate = word_count / duration * 60 if word_count else 0.0

        energy_component = max(0.0, min(100.0, mean_rms * 900))
        pause_component = max(0.0, 100 - abs(silence_ratio - 0.20) * 220)
        rate_component = max(0.0, 100 - abs(speaking_rate - 140) * 0.75) if speaking_rate else 50.0
        score = round(0.35 * energy_component + 0.30 * pause_component + 0.35 * rate_component, 2)

        if score >= 75:
            interpretation = "Clear and reasonably steady delivery"
        elif score >= 55:
            interpretation = "Moderate delivery; review pacing, pauses, and volume"
        else:
            interpretation = "Delivery needs practice or the recording quality is weak"

        return VoiceResult(
            duration_sec=round(duration, 2),
            speaking_rate_wpm=round(speaking_rate, 2),
            mean_rms=round(mean_rms, 5),
            median_pitch_hz=round(median_pitch, 2),
            silence_ratio=round(silence_ratio * 100, 2),
            score=max(0.0, min(100.0, score)),
            interpretation=interpretation,
        )
    except Exception as exc:
        return VoiceResult(0.0, 0.0, 0.0, 0.0, 0.0, 50.0, "Voice analysis unavailable", str(exc))


def clarity_score(text_result: TextResult, has_keywords: bool) -> float:
    filler_component = max(0.0, 100 - text_result.filler_rate * 8)
    diversity_component = max(0.0, min(100.0, text_result.lexical_diversity * 1.5))

    if has_keywords:
        score = (
            text_result.sentiment_score * 0.20
            + filler_component * 0.35
            + diversity_component * 0.25
            + text_result.keyword_coverage * 0.20
        )
    else:
        score = (
            text_result.sentiment_score * 0.25
            + filler_component * 0.45
            + diversity_component * 0.30
        )

    return round(max(0.0, min(100.0, score)), 2)


def overall_score(emotion: float, gaze: float, voice: float, clarity: float, weights: dict[str, int]) -> float:
    total = sum(weights.values()) or 1
    value = (
        emotion * weights["emotion"]
        + gaze * weights["gaze"]
        + voice * weights["voice"]
        + clarity * weights["clarity"]
    ) / total
    return round(max(0.0, min(100.0, value)), 2)


def performance_band(score: float) -> str:
    if score >= 85:
        return "Excellent practice performance"
    if score >= 70:
        return "Good practice performance"
    if score >= 55:
        return "Developing — review the feedback areas"
    return "Needs additional interview practice"


def generate_pdf(
    candidate_name: str,
    role: str,
    question: str,
    score_rows: list[list[str]],
    transcript: str,
) -> Path | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return None

    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name)
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    story = [
        Paragraph("AI Interview Analyzer Pro — Academic Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"<b>Candidate:</b> {html.escape(candidate_name)}", styles["BodyText"]),
        Paragraph(f"<b>Practice role:</b> {html.escape(role)}", styles["BodyText"]),
        Paragraph(f"<b>Question:</b> {html.escape(question)}", styles["BodyText"]),
        Spacer(1, 12),
    ]

    table = Table([["Metric", "Result"]] + score_rows, colWidths=[95 * mm, 70 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.extend(
        [
            table,
            Spacer(1, 14),
            Paragraph("Transcript", styles["Heading1"]),
            Paragraph(html.escape(transcript or "Transcript not available."), styles["BodyText"]),
            Spacer(1, 14),
            Paragraph("Responsible-Use Statement", styles["Heading2"]),
            Paragraph(
                "This academic prototype is intended for interview practice. The output must not be used as the sole basis for hiring, rejection, grading, diagnosis, or claims about honesty, personality, intelligence, or competence.",
                styles["BodyText"],
            ),
        ]
    )

    document.build(story)
    return output_path


st.title("🎙️ AI Interview Analyzer Pro")
st.markdown(
    """
### Academic Multimodal Interview-Practice Dashboard

Upload a recorded interview to analyze speech transcription, voice delivery,
facial emotion patterns, camera-facing attention, sentiment, filler words,
and expected-keyword coverage.
"""
)

st.warning(
    "Academic practice tool only. Do not use these scores as an automated hiring, rejection, grading, diagnostic, honesty, personality, intelligence, or competence decision."
)

with st.sidebar:
    st.header("Candidate and Rubric")
    candidate_name = st.text_input("Candidate name", "Syed Nasir Shah")
    role = st.text_input("Practice role", "AI / Data Science Position")
    question = st.text_area(
        "Interview question",
        "Tell me about yourself and explain why you are suitable for this role.",
        height=90,
    )
    keyword_text = st.text_input(
        "Expected keywords",
        "python, machine learning, deep learning, communication, project",
    )
    whisper_model = st.selectbox("Transcription model", ["tiny", "base"], index=0)

    st.header("Practice-Score Weights")
    weights = {
        "emotion": st.slider("Composure pattern", 0, 100, 20),
        "gaze": st.slider("Camera-facing attention", 0, 100, 20),
        "voice": st.slider("Voice delivery", 0, 100, 30),
        "clarity": st.slider("Transcript clarity", 0, 100, 30),
    }

uploaded_video = st.file_uploader(
    "Upload a recorded interview video",
    type=["mp4", "mov", "avi", "mkv"],
    help="For the first cloud test, use a clear 20–40 second MP4.",
)

if uploaded_video is not None:
    video_path = save_upload(uploaded_video)
    st.video(str(video_path))

    if st.button("🚀 Analyze Practice Interview", type="primary"):
        progress = st.progress(0, text="Preparing video...")

        try:
            progress.progress(10, text="Extracting audio...")
            audio_path = extract_audio(video_path)

            progress.progress(25, text="Transcribing speech...")
            transcript_result = transcribe_audio(audio_path, whisper_model)

            keywords = [item.strip() for item in keyword_text.split(",") if item.strip()]

            progress.progress(40, text="Analyzing language...")
            text_result = analyze_text(transcript_result.text, keywords)

            progress.progress(55, text="Analyzing voice delivery...")
            voice_result = analyze_voice(audio_path, text_result.word_count)

            progress.progress(70, text="Analyzing facial emotion patterns...")
            emotion_result = analyze_emotions(video_path)

            progress.progress(85, text="Estimating camera-facing attention...")
            gaze_result = analyze_gaze(video_path)

            clarity = clarity_score(text_result, bool(keywords))
            overall = overall_score(
                emotion_result.score,
                gaze_result.score,
                voice_result.score,
                clarity,
                weights,
            )

            st.session_state["analysis"] = {
                "candidate_name": candidate_name,
                "role": role,
                "question": question,
                "emotion": emotion_result,
                "gaze": gaze_result,
                "transcript": transcript_result,
                "voice": voice_result,
                "text": text_result,
                "clarity": clarity,
                "overall": overall,
                "band": performance_band(overall),
            }

            progress.progress(100, text="Analysis completed.")
        except Exception as exc:
            progress.empty()
            st.error(f"Analysis could not be completed: {exc}")

if "analysis" in st.session_state:
    result = st.session_state["analysis"]
    emotion_result: EmotionResult = result["emotion"]
    gaze_result: GazeResult = result["gaze"]
    transcript_result: TranscriptResult = result["transcript"]
    voice_result: VoiceResult = result["voice"]
    text_result: TextResult = result["text"]
    clarity = result["clarity"]
    overall = result["overall"]

    st.divider()
    st.header("Results")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Composure pattern", f"{emotion_result.score:.1f}%")
    col2.metric("Camera-facing attention", f"{gaze_result.score:.1f}%")
    col3.metric("Voice delivery", f"{voice_result.score:.1f}%")
    col4.metric("Transcript clarity", f"{clarity:.1f}%")
    col5.metric("Overall practice score", f"{overall:.1f}%")

    st.info(f"**Performance band:** {result['band']}")

    score_frame = pd.DataFrame(
        {
            "Metric": [
                "Composure pattern",
                "Camera-facing attention",
                "Voice delivery",
                "Transcript clarity",
                "Overall practice score",
            ],
            "Score": [
                emotion_result.score,
                gaze_result.score,
                voice_result.score,
                clarity,
                overall,
            ],
        }
    )
    st.bar_chart(score_frame.set_index("Metric"))

    emotion_tab, gaze_tab, voice_tab, transcript_tab, report_tab = st.tabs(
        ["😊 Emotion", "👀 Attention", "🎙️ Voice", "📝 Transcript", "📄 Report"]
    )

    with emotion_tab:
        if emotion_result.error:
            st.warning(emotion_result.error)
        if not emotion_result.timeline.empty:
            st.dataframe(emotion_result.timeline, use_container_width=True)
            chart_data = emotion_result.timeline.pivot_table(
                index="time_sec",
                columns="emotion",
                values="confidence",
                aggfunc="mean",
                fill_value=0,
            )
            st.line_chart(chart_data)
        st.write("Dominant sampled emotion:", emotion_result.dominant)
        st.write("Frames analyzed:", emotion_result.frames)

    with gaze_tab:
        if gaze_result.error:
            st.warning(gaze_result.error)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Camera-facing attention": f"{gaze_result.score:.2f}%",
                        "Face visibility": f"{gaze_result.face_visibility:.2f}%",
                        "Frames sampled": gaze_result.frames,
                        "Method": gaze_result.method,
                    }
                ]
            ),
            use_container_width=True,
        )

    with voice_tab:
        if voice_result.error:
            st.warning(voice_result.error)
        st.dataframe(pd.DataFrame([asdict(voice_result)]), use_container_width=True)
        st.write("Interpretation:", voice_result.interpretation)

    with transcript_tab:
        if transcript_result.error:
            st.warning(transcript_result.error)
        st.subheader("Transcript")
        st.write(transcript_result.text or "No transcript was produced.")
        st.download_button(
            "Download Transcript",
            transcript_result.text,
            file_name="interview_transcript.txt",
            mime="text/plain",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Language": transcript_result.language,
                        "Language probability": transcript_result.probability,
                        "Word count": text_result.word_count,
                        "Sentiment": text_result.sentiment,
                        "Filler words": text_result.filler_count,
                        "Fillers per 100 words": text_result.filler_rate,
                        "Lexical diversity": text_result.lexical_diversity,
                        "Keyword coverage": text_result.keyword_coverage,
                    }
                ]
            ),
            use_container_width=True,
        )
        st.write("Matched keywords:", ", ".join(text_result.matched_keywords) or "None")
        st.write("Missing keywords:", ", ".join(text_result.missing_keywords) or "None")

    with report_tab:
        csv_data = score_frame.to_csv(index=False)
        st.download_button(
            "Download Scores CSV",
            csv_data,
            file_name="interview_scores.csv",
            mime="text/csv",
        )

        score_rows = [
            ["Composure pattern", f"{emotion_result.score:.2f}%"],
            ["Camera-facing attention", f"{gaze_result.score:.2f}%"],
            ["Voice delivery", f"{voice_result.score:.2f}%"],
            ["Transcript clarity", f"{clarity:.2f}%"],
            ["Overall practice score", f"{overall:.2f}%"],
            ["Performance band", result["band"]],
        ]

        pdf_path = generate_pdf(
            result["candidate_name"],
            result["role"],
            result["question"],
            score_rows,
            transcript_result.text,
        )

        if pdf_path is not None:
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    "Download Detailed PDF Report",
                    pdf_file,
                    file_name="AI_Interview_Analyzer_Report.pdf",
                    mime="application/pdf",
                )

        st.subheader("Limitations")
        st.markdown(
            """
- Facial emotion labels do not reveal a person's internal state.
- Camera-facing attention is not the same as genuine eye contact.
- Voice and transcript scores can be affected by recording quality.
- A human evaluator must review the original recording and context.
"""
        )
else:
    st.info("Upload a short interview video and click **Analyze Practice Interview**.")

st.divider()
st.caption("AI Interview Analyzer Pro — Academic Version")
