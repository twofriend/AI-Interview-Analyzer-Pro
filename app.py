from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.audio_analysis import analyze_voice, extract_audio_to_wav, transcribe_audio
from modules.report_generator import generate_pdf
from modules.scoring import clarity_score, overall_practice_score, performance_band
from modules.text_analysis import analyze_text
from modules.video_analysis import analyze_emotions, analyze_gaze

st.set_page_config(page_title="AI Interview Analyzer Pro — Academic", page_icon="🎙️", layout="wide")

st.title("🎙️ AI Interview Analyzer Pro")
st.markdown("""
### Academic multimodal interview-practice dashboard

Analyze a recorded practice interview using pretrained deep-learning models for facial emotion
patterns, camera-facing visual attention, speech transcription, voice delivery, and text features.
""")
st.warning(
    "Academic practice tool only: do not use the scores as an automated hiring, rejection, "
    "grading, diagnostic, honesty, personality, intelligence, or competence decision."
)

with st.sidebar:
    st.header("Candidate and rubric")
    candidate_name = st.text_input("Candidate name", "Syed Nasir Shah")
    role = st.text_input("Practice role", "AI / Data Science Position")
    question = st.text_area(
        "Interview question",
        "Tell me about yourself and explain why you are suitable for this role.",
        height=90,
    )
    keyword_text = st.text_input(
        "Expected keywords (comma-separated)",
        "python, machine learning, deep learning, communication, project",
    )
    whisper_size = st.selectbox(
        "Transcription model",
        ["tiny", "base"],
        index=0,
        help="Tiny is faster. Base is usually more accurate but takes longer.",
    )

    st.header("Practice-score weights")
    weights = {
        "composure": st.slider("Composure pattern", 0, 100, 20),
        "gaze": st.slider("Camera-facing attention", 0, 100, 20),
        "delivery": st.slider("Voice delivery", 0, 100, 30),
        "clarity": st.slider("Transcript clarity", 0, 100, 30),
    }

uploaded = st.file_uploader(
    "Upload a recorded interview video",
    type=["mp4", "mov", "avi", "mkv"],
    help="For a quick test, use a clear 30–90 second video with one visible speaker.",
)

if uploaded is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    temp_video.write(uploaded.read())
    temp_video.close()
    video_path = Path(temp_video.name)
    st.video(str(video_path))

    if st.button("🚀 Analyze practice interview", type="primary"):
        progress = st.progress(0, text="Preparing the video…")
        try:
            progress.progress(10, text="Extracting the audio track…")
            audio_path = extract_audio_to_wav(video_path)

            progress.progress(25, text="Transcribing speech with Faster-Whisper…")
            transcript_result = transcribe_audio(str(audio_path), whisper_size)

            progress.progress(45, text="Analyzing transcript and expected keywords…")
            keywords = [item.strip() for item in keyword_text.split(",") if item.strip()]
            text_result = analyze_text(transcript_result.text, keywords)

            progress.progress(55, text="Analyzing voice delivery…")
            voice_result = analyze_voice(str(audio_path), text_result.word_count)

            progress.progress(70, text="Analyzing facial emotion patterns…")
            emotion_result = analyze_emotions(str(video_path))

            progress.progress(85, text="Estimating camera-facing visual attention…")
            gaze_result = analyze_gaze(str(video_path))

            clarity = clarity_score(
                text_result.sentiment_score,
                text_result.filler_rate_per_100_words,
                text_result.lexical_diversity,
                text_result.keyword_coverage,
                bool(keywords),
            )
            overall = overall_practice_score(
                emotion_result.composure_score,
                gaze_result.forward_gaze_score,
                voice_result.delivery_score,
                clarity,
                weights,
            )
            band = performance_band(overall)

            st.session_state["analysis"] = {
                "candidate_name": candidate_name,
                "role": role,
                "question": question,
                "emotion": emotion_result,
                "gaze": gaze_result,
                "voice": voice_result,
                "transcript": transcript_result,
                "text": text_result,
                "clarity": clarity,
                "overall": overall,
                "band": band,
            }
            progress.progress(100, text="Analysis complete.")
        except Exception as exc:
            progress.empty()
            st.error(f"Analysis could not be completed: {exc}")

if "analysis" in st.session_state:
    data = st.session_state["analysis"]
    emotion = data["emotion"]
    gaze = data["gaze"]
    voice = data["voice"]
    transcript = data["transcript"]
    text = data["text"]
    clarity = data["clarity"]
    overall = data["overall"]
    band = data["band"]

    st.divider()
    st.header("Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Composure pattern", f"{emotion.composure_score:.1f}%")
    c2.metric("Camera-facing attention", f"{gaze.forward_gaze_score:.1f}%")
    c3.metric("Voice delivery", f"{voice.delivery_score:.1f}%")
    c4.metric("Transcript clarity", f"{clarity:.1f}%")
    c5.metric("Overall practice score", f"{overall:.1f}%")
    st.info(f"**Performance band:** {band}")

    score_df = pd.DataFrame({
        "Metric": ["Composure pattern", "Camera-facing attention", "Voice delivery", "Transcript clarity", "Overall"],
        "Score": [emotion.composure_score, gaze.forward_gaze_score, voice.delivery_score, clarity, overall],
    })
    st.plotly_chart(px.bar(score_df, x="Metric", y="Score", range_y=[0,100],
                           title="Practice performance overview", text_auto=".1f"), use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "😊 Emotion timeline", "👀 Visual attention", "🎙️ Voice delivery",
        "📝 Transcript and text", "📄 Report and downloads"
    ])

    with tab1:
        if emotion.error:
            st.warning(emotion.error)
        if not emotion.timeline.empty:
            st.dataframe(emotion.timeline, use_container_width=True)
            st.plotly_chart(px.scatter(emotion.timeline, x="time_sec", y="confidence", color="emotion",
                                       title="Detected dominant emotion over sampled frames"),
                               use_container_width=True)
            counts = emotion.timeline["emotion"].value_counts().reset_index()
            counts.columns = ["Emotion", "Frames"]
            st.plotly_chart(px.pie(counts, names="Emotion", values="Frames", title="Emotion distribution"),
                               use_container_width=True)
        st.write("Dominant sampled emotion:", emotion.dominant_emotion)
        st.write("Frames analyzed:", emotion.frames_analyzed)

    with tab2:
        if gaze.error:
            st.warning(gaze.error)
        st.dataframe(pd.DataFrame([{
            "Forward-gaze proxy": f"{gaze.forward_gaze_score:.2f}%",
            "Face visibility": f"{gaze.face_visibility_score:.2f}%",
            "Frames sampled": gaze.frames_sampled,
            "Method": gaze.method,
        }]), use_container_width=True)
        st.caption("Approximate camera-facing attention proxy; not a clinical eye tracker.")

    with tab3:
        if voice.error:
            st.warning(voice.error)
        st.dataframe(pd.DataFrame([asdict(voice)]), use_container_width=True)
        st.write("Interpretation:", voice.interpretation)

    with tab4:
        if transcript.error:
            st.warning(f"Transcription note: {transcript.error}")
        st.subheader("Transcript")
        st.write(transcript.text or "No transcript was produced.")
        st.download_button("Download transcript", transcript.text or "", "interview_transcript.txt", "text/plain")

        st.dataframe(pd.DataFrame([{
            "Detected language": transcript.language,
            "Language probability": transcript.language_probability,
            "Word count": text.word_count,
            "Sentiment": text.sentiment_label,
            "Sentiment score": text.sentiment_score,
            "Filler words": text.filler_count,
            "Fillers per 100 words": text.filler_rate_per_100_words,
            "Lexical diversity": text.lexical_diversity,
            "Expected-keyword coverage": text.keyword_coverage,
        }]), use_container_width=True)
        st.write("Matched keywords:", ", ".join(text.matched_keywords) or "None")
        st.write("Missing keywords:", ", ".join(text.missing_keywords) or "None")
        if transcript.segments:
            st.subheader("Timestamped segments")
            st.dataframe(pd.DataFrame(transcript.segments), use_container_width=True)

    with tab5:
        summary_df = pd.DataFrame([
            {"Metric": "Composure pattern", "Score": emotion.composure_score},
            {"Metric": "Camera-facing attention", "Score": gaze.forward_gaze_score},
            {"Metric": "Voice delivery", "Score": voice.delivery_score},
            {"Metric": "Transcript clarity", "Score": clarity},
            {"Metric": "Overall practice score", "Score": overall},
        ])
        st.download_button("Download scores CSV", summary_df.to_csv(index=False),
                           "interview_practice_scores.csv", "text/csv")

        pdf_path = generate_pdf(
            data["candidate_name"], data["role"], data["question"],
            {"Composure pattern": emotion.composure_score,
             "Camera-facing attention": gaze.forward_gaze_score,
             "Voice delivery": voice.delivery_score,
             "Transcript clarity": clarity,
             "Overall practice score": overall},
            band,
            {"Dominant sampled emotion": emotion.dominant_emotion,
             "Frames analyzed": emotion.frames_analyzed,
             "Analysis note": emotion.error or "Completed"},
            {"Forward-gaze proxy": f"{gaze.forward_gaze_score:.2f}%",
             "Face visibility": f"{gaze.face_visibility_score:.2f}%",
             "Method": gaze.method,
             "Analysis note": gaze.error or "Completed"},
            {"Duration": f"{voice.duration_sec:.2f} seconds",
             "Speaking rate": f"{voice.speaking_rate_wpm:.2f} words/minute",
             "Median pitch": f"{voice.median_pitch_hz:.2f} Hz",
             "Silence ratio": f"{voice.silence_ratio:.2f}%",
             "Interpretation": voice.interpretation},
            {"Language": transcript.language,
             "Word count": text.word_count,
             "Sentiment": text.sentiment_label,
             "Filler words": text.filler_count,
             "Lexical diversity": f"{text.lexical_diversity:.2f}%",
             "Keyword coverage": f"{text.keyword_coverage:.2f}%"},
            transcript.text,
        )
        with open(pdf_path, "rb") as handle:
            st.download_button("Download detailed PDF report", handle,
                               "AI_Interview_Analyzer_Academic_Report.pdf", "application/pdf")

        st.subheader("Limitations")
        st.markdown("""
- Pretrained models can be wrong and may behave differently across lighting, accents, cultures, disabilities, and recording devices.
- Facial emotion labels do not reveal a person's internal state.
- Camera-facing attention is not the same as genuine eye contact.
- Voice and language metrics may reflect recording quality rather than the speaker.
- A human evaluator must review the original video, transcript, context, and rubric.
""")
