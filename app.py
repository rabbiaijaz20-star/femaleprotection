import base64
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from groq import Groq
from PIL import Image, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="EvidenceGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "EvidenceGuard AI"
APP_TAGLINE = "Digital Evidence Organization & Analysis"
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence_data" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CATEGORIES = [
    "Threat",
    "Blackmail / Extortion",
    "Sexual Harassment",
    "Cyberstalking",
    "Impersonation",
    "Abuse / Insults",
    "General Online Harassment",
    "Unclear",
]

SOURCES = ["WhatsApp", "Instagram", "Facebook", "SMS", "Email", "Other"]

NCCIA_COMPLAINT_URL = "https://complaint.nccia.gov.pk/"
NCCIA_INFO_URL = "https://www.nccia.gov.pk/faqs.php"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .eg-hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(127,127,127,.20);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(30,41,59,.95), rgba(15,23,42,.98));
        color: white;
        margin-bottom: 1.25rem;
    }

    .eg-hero h1 {
        margin: 0;
        font-size: 2.15rem;
        letter-spacing: -0.03em;
    }

    .eg-hero p {
        margin: .45rem 0 0;
        color: #cbd5e1;
        font-size: 1rem;
    }

    .eg-badge {
        display: inline-block;
        padding: .28rem .65rem;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.15);
        color: #e2e8f0;
        font-size: .78rem;
        margin-bottom: .7rem;
    }

    .eg-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(127,127,127,.18);
        border-radius: 14px;
        background: rgba(127,127,127,.045);
        margin-bottom: .8rem;
    }

    .eg-muted {
        color: #64748b;
        font-size: .9rem;
    }

    .eg-small {
        font-size: .78rem;
        color: #64748b;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(127,127,127,.16);
        border-radius: 14px;
        padding: .7rem;
    }

    .stButton > button {
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def now_display():
    return datetime.now().strftime("%d %b %Y, %I:%M %p")


def make_case_id():
    return f"EG-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GROQ_API_KEY", "")

    return str(key).strip()


def get_client():
    key = get_api_key()
    return Groq(api_key=key) if key else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120] or "evidence.jpg"


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json_object(raw: str):
    raw = strip_thinking(raw)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in AI response.")

    candidate = raw[start:end + 1]
    return json.loads(candidate)


def validate_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None, "No file selected."

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return None, "Only PNG, JPG, and JPEG images are supported."

    data = uploaded_file.getvalue()
    if not data:
        return None, "The uploaded file is empty."

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"File is larger than {MAX_UPLOAD_MB} MB."

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
    except (UnidentifiedImageError, OSError):
        return None, "The file could not be verified as a valid image."

    return data, None


def clean_result(result):
    categories = result.get("categories", [])
    if isinstance(categories, str):
        categories = [categories]
    if not isinstance(categories, list):
        categories = []

    categories = [
        c.strip() for c in categories
        if isinstance(c, str) and c.strip() in CATEGORIES
    ]

    risk = normalize_text(result.get("risk_level", "Unclear")).title()
    if risk not in {"High", "Medium", "Low", "Unclear"}:
        risk = "Unclear"

    return {
        "extracted_text": normalize_text(result.get("extracted_text")),
        "detected_date": normalize_text(result.get("detected_date")),
        "categories": categories or ["Unclear"],
        "risk_level": risk,
        "factual_summary": normalize_text(result.get("factual_summary")),
        "preservation_notes": normalize_text(result.get("preservation_notes")),
        "confidence": normalize_text(result.get("confidence", "Not provided")),
    }


# ============================================================
# GROQ VISION ANALYSIS
# ============================================================

def analyze_evidence(data: bytes, filename: str):
    client = get_client()
    if not client:
        raise RuntimeError("Groq API key is not configured.")

    encoded = base64.b64encode(data).decode("utf-8")

    prompt = """
You are EvidenceGuard AI, an evidence-organization assistant.

Analyze the supplied screenshot ONLY for visible or reasonably readable information.
Do not invent names, dates, phone numbers, usernames, locations, identities, or events.
Do not make legal conclusions and do not state that a person committed a crime.

Return ONLY one valid JSON object. Do not use markdown and do not include <think> tags.

Required JSON keys:
{
  "extracted_text": "faithful transcription of readable visible text",
  "detected_date": "date/time visible in the image, otherwise empty string",
  "categories": ["one or more allowed categories"],
  "risk_level": "High, Medium, Low, or Unclear",
  "factual_summary": "short neutral summary of what is visibly shown",
  "preservation_notes": "practical evidence-preservation notes based only on what is visible",
  "confidence": "High, Medium, Low, or Not provided"
}

Allowed categories:
Threat
Blackmail / Extortion
Sexual Harassment
Cyberstalking
Impersonation
Abuse / Insults
General Online Harassment
Unclear

Rules:
- If text is unreadable, say so instead of guessing.
- Preserve wording as faithfully as possible in extracted_text.
- Risk level is an organizational signal, not a legal determination.
- If there is no clear evidence of a category, use "Unclear".
"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this evidence image. Original filename: {filename}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}"
                        },
                    },
                ],
            },
        ],
        temperature=0,
        max_completion_tokens=2048,
        reasoning_effort="none",
    )

    raw = completion.choices[0].message.content or ""
    result = extract_json_object(raw)
    return clean_result(result)


# ============================================================
# LOCAL EVIDENCE STORAGE
# ============================================================

def save_evidence(data: bytes, filename: str, source: str, analysis: dict, case_id: str):
    evidence_hash = sha256_bytes(data)
    evidence_id = f"EV-{datetime.now().strftime('%Y%m%d%H%M%S')}-{evidence_hash[:8].upper()}"

    safe_name = safe_filename(filename)
    stored_name = f"{evidence_hash[:16]}_{safe_name}"
    stored_path = EVIDENCE_DIR / stored_name
    stored_path.write_bytes(data)

    record = {
        "evidence_id": evidence_id,
        "case_id": case_id,
        "original_filename": filename,
        "stored_filename": stored_name,
        "source": source,
        "sha256": evidence_hash,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        **analysis,
    }

    return record


def load_records(case_id):
    records = st.session_state.get("records", [])
    return [r for r in records if r.get("case_id") == case_id]


def delete_evidence_file(record):
    path = EVIDENCE_DIR / record.get("stored_filename", "")
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ============================================================
# GAP ANALYSIS
# ============================================================

def identify_gaps(records):
    gaps = []

    if not records:
        return ["No evidence has been added to this case."]

    dates = [r.get("detected_date", "").strip() for r in records]
    if any(not d for d in dates):
        gaps.append("One or more evidence items do not contain a clearly detected date/time.")

    if len(records) == 1:
        gaps.append("Only one evidence item is currently recorded; additional context may be useful.")

    sources = {r.get("source") for r in records if r.get("source")}
    if len(sources) == 1:
        gaps.append("Only one platform/source is represented in the current evidence set.")

    if any(r.get("confidence") in {"Low", "Not provided"} for r in records):
        gaps.append("At least one item has low or unavailable extraction confidence; manual verification is recommended.")

    if any(not r.get("extracted_text") for r in records):
        gaps.append("At least one item contains no reliable extracted text.")

    gaps.append("Independent verification of account ownership/identity may be required by the relevant authority.")
    gaps.append("Original device/account context may be requested during a formal investigation.")

    return gaps


# ============================================================
# PDF GENERATION
# ============================================================

def register_fonts():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "EGSans"),
        ("C:/Windows/Fonts/arial.ttf", "EGArial"),
    ]

    for path, name in candidates:
        try:
            if Path(path).exists():
                pdfmetrics.registerFont(TTFont(name, path))
                return name
        except Exception:
            continue

    return "Helvetica"


def pdf_paragraph(text, style):
    safe = (
        normalize_text(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(safe, style)


def build_pdf(case_id, records, gaps):
    font_name = register_fonts()
    output = io.BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title=f"EvidenceGuard AI - {case_id}",
        author="EvidenceGuard AI",
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "EGTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=21,
        leading=25,
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    subtitle = ParagraphStyle(
        "EGSubtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9.5,
        textColor=colors.HexColor("#64748B"),
        leading=13,
        spaceAfter=14,
    )

    section = ParagraphStyle(
        "EGSection",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=7,
    )

    body = ParagraphStyle(
        "EGBODY",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8.8,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )

    small = ParagraphStyle(
        "EGSmall",
        parent=body,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
    )

    label = ParagraphStyle(
        "EGLabel",
        parent=body,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
    )

    story = []

    # Header
    story.append(Paragraph(APP_NAME, title))
    story.append(Paragraph(APP_TAGLINE, subtitle))

    meta = [
        [Paragraph("<b>Case ID</b>", body), pdf_paragraph(case_id, body),
         Paragraph("<b>Generated</b>", body), pdf_paragraph(now_display(), body)],
        [Paragraph("<b>Evidence items</b>", body), str(len(records)),
         Paragraph("<b>Sources</b>", body), str(len({r.get("source") for r in records if r.get("source")}))],
    ]

    meta_table = Table(meta, colWidths=[28*mm, 52*mm, 30*mm, 52*mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), .25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)

    story.append(Paragraph("REPORT PURPOSE", section))
    story.append(Paragraph(
        "This report organizes digital evidence supplied to EvidenceGuard AI and presents "
        "AI-assisted transcription, classification, chronology, and preservation notes. "
        "It is intended as a supporting evidence index and not as a forensic authentication report.",
        body,
    ))

    story.append(Paragraph("EVIDENCE INDEX", section))

    index_data = [[
        Paragraph("<b>ID</b>", body),
        Paragraph("<b>Date</b>", body),
        Paragraph("<b>Source</b>", body),
        Paragraph("<b>Category</b>", body),
        Paragraph("<b>Risk</b>", body),
    ]]

    for r in records:
        index_data.append([
            pdf_paragraph(r.get("evidence_id", ""), small),
            pdf_paragraph(r.get("detected_date") or "Not detected", small),
            pdf_paragraph(r.get("source", ""), small),
            pdf_paragraph(", ".join(r.get("categories", [])), small),
            pdf_paragraph(r.get("risk_level", "Unclear"), small),
        ])

    index_table = Table(index_data, colWidths=[35*mm, 28*mm, 25*mm, 55*mm, 20*mm], repeatRows=1)
    index_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), .25, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(index_table)

    for i, r in enumerate(records, start=1):
        story.append(PageBreak())
        story.append(Paragraph(f"EVIDENCE ITEM {i:03d}", section))

        details = [
            [Paragraph("<b>Evidence ID</b>", body), pdf_paragraph(r.get("evidence_id"), body)],
            [Paragraph("<b>Original filename</b>", body), pdf_paragraph(r.get("original_filename"), body)],
            [Paragraph("<b>Source/platform</b>", body), pdf_paragraph(r.get("source"), body)],
            [Paragraph("<b>Detected date/time</b>", body), pdf_paragraph(r.get("detected_date") or "Not detected", body)],
            [Paragraph("<b>Risk level</b>", body), pdf_paragraph(r.get("risk_level"), body)],
            [Paragraph("<b>Category</b>", body), pdf_paragraph(", ".join(r.get("categories", [])), body)],
            [Paragraph("<b>Extraction confidence</b>", body), pdf_paragraph(r.get("confidence") or "Not provided", body)],
            [Paragraph("<b>SHA-256</b>", small), pdf_paragraph(r.get("sha256"), small)],
        ]

        details_table = Table(details, colWidths=[43*mm, 120*mm])
        details_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), .25, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(details_table)

        stored_path = EVIDENCE_DIR / r.get("stored_filename", "")
        if stored_path.exists():
            try:
                img = Image.open(stored_path)
                w, h = img.size
                max_w = 150 * mm
                max_h = 100 * mm
                scale = min(max_w / w, max_h / h, 1)
                story.append(Spacer(1, 5))
                story.append(RLImage(str(stored_path), width=w*scale, height=h*scale))
            except Exception:
                pass

        story.append(Paragraph("VISIBLE / EXTRACTED CONTENT", section))
        extracted = r.get("extracted_text") or "No reliable text was extracted."
        story.append(pdf_paragraph(extracted, body))

        story.append(Paragraph("FACTUAL AI SUMMARY", section))
        summary = r.get("factual_summary") or "No summary was generated."
        story.append(pdf_paragraph(summary, body))

        story.append(Paragraph("PRESERVATION NOTES", section))
        notes = r.get("preservation_notes") or "Preserve the original file without modification."
        story.append(pdf_paragraph(notes, body))

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "AI analysis is an organizational aid. It should be manually reviewed against the original evidence.",
            small,
        ))

    story.append(PageBreak())
    story.append(Paragraph("EVIDENCE GAPS & FOLLOW-UP", section))
    for gap in gaps:
        story.append(Paragraph(f"• {gap}", body))

    story.append(Paragraph("IMPORTANT NOTICE", section))
    story.append(Paragraph(
        "<b>This report is not legal advice, a forensic authentication report, or a determination of guilt.</b> "
        "The original evidence should be preserved in its original form and provided to the relevant authority "
        "or legal professional when requested. AI-generated classifications and summaries should be independently verified.",
        body,
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"EvidenceGuard AI • Generated {now_display()} • Case {case_id}",
        small,
    ))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 7)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(16*mm, 9*mm, "EvidenceGuard AI — Supporting evidence organization report")
        canvas.drawRightString(A4[0]-16*mm, 9*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "case_id" not in st.session_state:
    st.session_state.case_id = make_case_id()

if "records" not in st.session_state:
    st.session_state.records = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
<div class="eg-hero">
    <div class="eg-badge">AI-assisted digital evidence organizer</div>
    <h1>🛡️ {APP_NAME}</h1>
    <p>{APP_TAGLINE}</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Case")
    st.code(st.session_state.case_id)

    if st.button("➕ Start New Case", use_container_width=True):
        st.session_state.case_id = make_case_id()
        st.session_state.records = []
        st.session_state.analysis_result = None
        st.rerun()

    st.divider()
    st.markdown("### Pakistan Support")
    st.markdown("**Emergency:** 15")
    st.markdown("[Virtual Women Police Station](https://vwps.psca.gop.pk/)")
    st.markdown("[PKCERT SAFER](https://pkcert.gov.pk/safer.asp)")

    st.divider()
    if get_api_key():
        st.success("Groq AI configured")
    else:
        st.error("Groq API key not configured")

    st.caption(
        "Privacy: uploaded evidence is processed by Groq for AI analysis and may be stored "
        "locally by this application. Do not upload evidence unless you are authorized to process it."
    )


# ============================================================
# DASHBOARD METRICS
# ============================================================

records = load_records(st.session_state.case_id)
high_count = sum(r.get("risk_level") == "High" for r in records)
source_count = len({r.get("source") for r in records if r.get("source")})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Evidence Items", len(records))
c2.metric("Sources", source_count)
c3.metric("High Risk", high_count)
c4.metric("Case ID", st.session_state.case_id)

st.divider()

tabs = st.tabs([
    "📥 Evidence",
    "🕒 Timeline",
    "🔎 Gap Identifier",
    "📄 Case PDF",
    "🔐 Privacy",
])


# ============================================================
# EVIDENCE TAB
# ============================================================

with tabs[0]:
    st.subheader("Add Digital Evidence")
    st.caption(
        "Upload an original screenshot, select its source, and run AI-assisted organization."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        source = st.selectbox("Evidence source", SOURCES)
        uploaded = st.file_uploader(
            "Screenshot",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
        )

        consent = st.checkbox(
            "I am authorized to process this evidence and understand that the image will be sent to Groq for AI analysis."
        )

        analyze_button = st.button(
            "🔎 Analyze Evidence",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None or not consent,
        )

    with col2:
        if uploaded:
            data, error = validate_uploaded_image(uploaded)

            if error:
                st.error(error)
            else:
                st.image(data, caption=uploaded.name, use_container_width=True)
                st.caption(f"SHA-256: `{sha256_bytes(data)}`")

    if analyze_button and uploaded:
        data, error = validate_uploaded_image(uploaded)

        if error:
            st.error(error)
        else:
            try:
                with st.spinner("Analyzing evidence securely..."):
                    result = analyze_evidence(data, uploaded.name)
                st.session_state.analysis_result = result
                st.session_state.analysis_bytes = data
                st.session_state.analysis_filename = uploaded.name
                st.session_state.analysis_source = source
                st.success("Analysis completed. Review the result before saving.")
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}")

    result = st.session_state.analysis_result

    if result:
        st.divider()
        st.subheader("Analysis Review")

        r1, r2, r3 = st.columns(3)
        r1.metric("Risk", result.get("risk_level", "Unclear"))
        r2.metric("Confidence", result.get("confidence", "Not provided"))
        r3.metric("Category", result.get("categories", ["Unclear"])[0])

        st.markdown("### Extracted content")
        st.text_area(
            "Review the transcription before saving",
            result.get("extracted_text", ""),
            height=160,
            key="review_extracted_text",
        )

        st.markdown("### Factual summary")
        st.write(result.get("factual_summary") or "No summary available.")

        st.markdown("### Preservation notes")
        st.write(result.get("preservation_notes") or "Preserve the original file without modification.")

        if st.button("💾 Save Evidence to Case", type="primary"):
            record = save_evidence(
                st.session_state.analysis_bytes,
                st.session_state.analysis_filename,
                st.session_state.analysis_source,
                result,
                st.session_state.case_id,
            )

            # Use manually reviewed extracted text.
            record["extracted_text"] = st.session_state.get(
                "review_extracted_text",
                record.get("extracted_text", ""),
            )

            st.session_state.records.append(record)
            st.session_state.analysis_result = None
            st.success(f"Evidence saved as {record['evidence_id']}.")
            st.rerun()

    if records:
        st.divider()
        st.subheader("Current Case Evidence")

        for idx, record in enumerate(records):
            with st.expander(
                f"{record.get('evidence_id')} — {record.get('source')} — {record.get('risk_level')}",
                expanded=False,
            ):
                a, b = st.columns([1, 2])

                with a:
                    path = EVIDENCE_DIR / record.get("stored_filename", "")
                    if path.exists():
                        st.image(str(path), use_container_width=True)

                with b:
                    st.write(f"**Original filename:** {record.get('original_filename')}")
                    st.write(f"**Detected date:** {record.get('detected_date') or 'Not detected'}")
                    st.write(f"**Categories:** {', '.join(record.get('categories', []))}")
                    st.write(f"**SHA-256:** `{record.get('sha256')}`")
                    st.write(f"**Confidence:** {record.get('confidence') or 'Not provided'}")
                    st.write("**Summary:**", record.get("factual_summary") or "Not available")

                    if st.button("Delete evidence", key=f"delete_{idx}"):
                        delete_evidence_file(record)
                        st.session_state.records.pop(idx)
                        st.rerun()


# ============================================================
# TIMELINE TAB
# ============================================================

with tabs[1]:
    st.subheader("Evidence Timeline")

    if not records:
        st.info("Add evidence to build the case timeline.")
    else:
        timeline = sorted(
            records,
            key=lambda r: (r.get("detected_date") or "9999", r.get("uploaded_at") or ""),
        )

        for i, r in enumerate(timeline, start=1):
            st.markdown(
                f"""
<div class="eg-card">
    <b>{i:02d}. {r.get('evidence_id')}</b><br>
    <span class="eg-muted">
    {r.get('detected_date') or 'Date not detected'} ·
    {r.get('source')} ·
    {', '.join(r.get('categories', []))} ·
    Risk: {r.get('risk_level')}
    </span>
</div>
""",
                unsafe_allow_html=True,
            )


# ============================================================
# GAP IDENTIFIER TAB
# ============================================================

with tabs[2]:
    st.subheader("Evidence Gap Identifier")
    st.caption("These are organizational follow-up points, not legal findings.")

    gaps = identify_gaps(records)

    for gap in gaps:
        st.warning(gap)

    if records:
        st.markdown("### Recommended organization")
        st.write(
            "Keep the original screenshots unchanged, retain their SHA-256 hashes, "
            "and preserve additional surrounding conversation/context where available."
        )


# ============================================================
# PDF TAB
# ============================================================

with tabs[3]:
    st.subheader("Professional Case Report")

    if not records:
        st.info("Add at least one evidence item before generating the report.")
    else:
        gaps = identify_gaps(records)

        st.markdown(
            """
            <div class="eg-card">
            <b>Report type:</b> Supporting Digital Evidence Organization Report<br>
            <span class="eg-small">
            This report organizes the evidence recorded in this case. It does not replace
            original evidence and does not constitute forensic authentication or legal advice.
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pdf_bytes = build_pdf(st.session_state.case_id, records, gaps)

        st.download_button(
            "📄 Generate / Download Professional Case PDF",
            data=pdf_bytes,
            file_name=f"EvidenceGuard_{st.session_state.case_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "Keep the original screenshots/files together with this PDF. The PDF is an "
            "organized supporting report, not a replacement for the original evidence."
        )

        st.divider()

        # --------------------------------------------------------
        # OFFICIAL REPORTING / SUBMISSION
        # --------------------------------------------------------
        st.subheader("📨 Submit Case to the Official Authority")

        st.markdown(
            f"""
            <div class="eg-card">
            <b>Your evidence package is ready.</b><br><br>
            EvidenceGuard AI has organized the evidence, generated a case report, and
            calculated SHA-256 hashes for the saved evidence files.<br><br>
            <b>Next step:</b> submit the complaint through the official
            National Cyber Crime Investigation Agency (NCCIA) channel.
            </div>
            """,
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Case ID", st.session_state.case_id)
        s2.metric("Evidence Items", len(records))
        s3.metric("High Risk Items", high_count)

        st.warning(
            "EvidenceGuard AI does not file or register the complaint itself. "
            "The button below opens the official NCCIA complaint portal. "
            "You must complete the official form and upload/provide the requested information "
            "and evidence yourself."
        )

        st.link_button(
            "📨 Open Official NCCIA Complaint Portal",
            NCCIA_COMPLAINT_URL,
            type="primary",
            use_container_width=True,
        )

        st.markdown("### What to take/submit")
        st.markdown(
            """
            - **EvidenceGuard case PDF** generated above.
            - **Original evidence files/screenshots** — do not rely on the PDF alone.
            - Your relevant case details and chronology.
            - Any other information requested by NCCIA.
            """
        )

        st.info(
            "NCCIA's current FAQ says complaint registration may involve a written application, "
            "CNIC copy, and evidence copy. It also lists the online NCCIA complaint form and "
            "Cybercrime Reporting Centres. Check the official instructions before submitting."
        )

        st.markdown("### Official guidance")
        st.link_button(
            "🌐 View NCCIA Complaint & Reporting Guidance",
            NCCIA_INFO_URL,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="eg-card">
            <b>Important:</b> EvidenceGuard AI is an independent evidence-organization tool,
            not a government department. A generated report does not mean that a complaint has
            been registered, accepted, investigated, or proven. The relevant authority decides
            what evidence it requires and what action, if any, is taken.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PRIVACY TAB
# ============================================================

with tabs[4]:
    st.subheader("Privacy & Security")

    st.markdown(
        """
### What this application does

- Evidence files are hashed with SHA-256.
- Evidence files are stored in the application's local evidence directory.
- Images are sent to Groq only when you choose to analyze them.
- AI output is presented as an organizational aid.
- The application does not determine guilt or provide legal advice.

### Before public deployment

1. Keep `GROQ_API_KEY` in deployment secrets/environment variables.
2. Never commit your real API key to GitHub.
3. Do not expose the evidence storage directory as a public static folder.
4. Add authentication if the application is intended for private case work.
5. Consider a retention/deletion policy for uploaded evidence.
6. Use HTTPS.
7. Review the privacy obligations that apply to your deployment and users.

### Important

The original screenshot remains the primary evidence. The AI-generated transcription,
classification, and summary should always be checked against the original image.
"""
    )
