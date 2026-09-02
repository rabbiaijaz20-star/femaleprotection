# EvidenceGuard AI — Professional Deployment Version

EvidenceGuard AI is a Streamlit application for organizing digital evidence and producing an AI-assisted supporting evidence report.

## Features

- Groq vision analysis
- Screenshot upload and validation
- SHA-256 evidence hashing
- Evidence IDs and case IDs
- Source/platform classification
- Neutral AI summaries
- Evidence timeline
- Evidence gap identification
- Professional PDF case report
- Original evidence preservation
- Privacy/security guidance

## Local setup

### 1. Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure Groq

Create:

`.streamlit/secrets.toml`

with:

```toml
GROQ_API_KEY = "YOUR_REAL_GROQ_API_KEY"
```

Never commit the real key to GitHub.

### 4. Run

```powershell
streamlit run app.py
```

## Important deployment note

The generated PDF is a supporting digital evidence organization report. It is not a forensic authentication report, legal advice, or a determination of guilt.

For real cases, preserve the original evidence files and provide them to the relevant authority or legal professional when requested.

## Public deployment security

For a public deployment, use the platform's secret manager for `GROQ_API_KEY`. Do not publish `.streamlit/secrets.toml`.

If this application will process real sensitive cases, add authentication and a clear retention/deletion policy before public launch.
