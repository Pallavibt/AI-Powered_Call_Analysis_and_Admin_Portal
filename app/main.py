
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.data import get_all_calls, get_call_by_id
from app.analysis import analyze_call


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"

app = FastAPI(
    title="AI-Powered Call Analysis API",
    description="Analyze customer calls and generate PDF reports.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return {
        "message": "Call Analysis API is running",
        "documentation": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/calls")
def list_calls():
    result = []

    for call in get_all_calls():
        result.append({
            "id": call["id"],
            "title": call["title"],
            "customer": call["customer"],
            "scenario": call["scenario"],
            "transcript": call["transcript"],
            "audio": f"/static/audio/{call['id']}.wav",
        })

    return {"calls": result}


@app.get("/api/calls/{call_id}/analyze")
def analyze_one_call(call_id: str):
    call = get_call_by_id(call_id)

    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")

    audio_file = AUDIO_DIR / f"{call_id}.wav"

    if not audio_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio file missing. Run scripts/generate_audio.py",
        )

    return analyze_call(call)


@app.get("/api/calls/{call_id}/report")
def download_report(call_id: str):
    call = get_call_by_id(call_id)

    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")

    analysis = analyze_call(call)

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph("AI-Powered Call Analysis Report", styles["Title"])
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Call ID:</b> {analysis['call_id']}", styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Customer:</b> {analysis['customer']}", styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Title:</b> {analysis['title']}", styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    summary = [
        ["Metric", "Result"],
        ["Sentiment", analysis["sentiment"]],
        ["Sentiment confidence", f"{analysis['sentiment_confidence']}%"],
        ["Estimated purchase probability",
         f"{analysis['purchase_probability']}%"],
    ]

    table = Table(summary, colWidths=[230, 230])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Transcript", styles["Heading2"]))
    story.append(Paragraph(analysis["transcript"], styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Extracted Features", styles["Heading2"]))
    for name, value in analysis["features"].items():
        story.append(Paragraph(
            f"<b>{name.replace('_', ' ').title()}:</b> {value}",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Detailed Insights", styles["Heading2"]))
    for insight in analysis["insights"]:
        story.append(Paragraph(f"- {insight}", styles["BodyText"]))
        story.append(Spacer(1, 5))

    story.append(Paragraph("Recommended Actions", styles["Heading2"]))
    for recommendation in analysis["recommendations"]:
        story.append(Paragraph(f"- {recommendation}", styles["BodyText"]))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"<b>Model limitation:</b> {analysis['model_note']}",
        styles["BodyText"],
    ))

    document.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="{call_id}_report.pdf"'
        },
    )
