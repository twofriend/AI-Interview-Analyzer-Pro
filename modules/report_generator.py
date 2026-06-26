from __future__ import annotations

import html
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def safe(value: object) -> str:
    return html.escape(str(value))


def generate_pdf(candidate_name: str, role: str, question: str, scores: dict[str, float], band: str,
                 emotion_summary: dict, gaze_summary: dict, voice_summary: dict,
                 text_summary: dict, transcript: str) -> Path:
    path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    story = [
        Paragraph("AI Interview Analyzer Pro — Academic Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"<b>Candidate:</b> {safe(candidate_name)}", styles["BodyText"]),
        Paragraph(f"<b>Practice role:</b> {safe(role)}", styles["BodyText"]),
        Paragraph(f"<b>Interview question:</b> {safe(question or 'Not supplied')}", styles["BodyText"]),
        Spacer(1, 12),
    ]

    rows = [["Practice Metric", "Score"]] + [[k, f"{v:.2f}%"] for k, v in scores.items()] + [["Performance band", band]]
    table = Table(rows, colWidths=[95*mm, 70*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [table, Spacer(1, 14)]

    for heading, values in [
        ("Facial emotion summary", emotion_summary),
        ("Visual-attention summary", gaze_summary),
        ("Voice-delivery summary", voice_summary),
        ("Transcript and language summary", text_summary),
    ]:
        story.append(Paragraph(heading, styles["Heading2"]))
        t = Table([["Measure", "Result"]] + [[safe(k), safe(v)] for k, v in values.items()], colWidths=[75*mm, 90*mm])
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCEAF5")),
                               ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                               ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                               ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story += [t, Spacer(1, 12)]

    story += [PageBreak(), Paragraph("Transcript", styles["Heading1"]),
              Paragraph(safe(transcript or "Transcript not available."), styles["BodyText"]),
              Spacer(1, 14), Paragraph("Responsible-use statement", styles["Heading2"]),
              Paragraph("This academic prototype is intended for interview practice and research demonstration. "
                        "Emotion, gaze, voice, and language signals are approximate and may be affected by camera "
                        "position, culture, disability, accent, lighting, noise, and model limitations. The output "
                        "must not be used as the sole basis for hiring, rejection, grading, diagnosis, or claims "
                        "about honesty, personality, intelligence, or competence.", styles["BodyText"])]
    doc.build(story)
    return path
