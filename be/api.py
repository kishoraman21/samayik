"""
Samayik AI — FastAPI HTTP Server
=================================
Thin wrapper around SamayikPipeline. All logic lives in main.py.
Changes in main.py are automatically reflected here.

Run with:
  uvicorn api:app --reload --port 8000
"""

import io
import re
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from main import SamayikPipeline

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Samayik AI",
    description="Transform messy hospital patient records into FHIR R4 standard.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline once at startup
pipeline = SamayikPipeline()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Samayik AI",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "convert": "POST /convert — upload CSV, JSON, or SQL file",
            "docs": "GET /docs — interactive API docs (Swagger UI)",
        },
    }


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    patient_id: str = Form(default=""),
    first_name: str = Form(default=""),
    last_name: str = Form(default=""),
    dob: str = Form(default=""),
    gender: str = Form(default=""),
):
    """
    Convert a hospital data file (CSV, JSON, or SQL) to a FHIR R4 Bundle.
    Patient identity is auto-extracted from the data itself.
    Form fields are used as fallback only.
    """
    # Read uploaded file
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "data.csv"

    # Form metadata (used as fallback by PatientExtractor)
    form_meta = {
        "patient_id": patient_id,
        "first_name": first_name,
        "last_name": last_name,
        "dob": dob,
        "gender": gender,
    }

    # Auto-detect format
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    fmt_map = {"csv": "csv", "json": "json", "sql": "sql", "xml": "xml"}

    if ext in fmt_map:
        fmt = fmt_map[ext]
    else:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            fmt = "json"
        elif stripped.startswith("<"):
            fmt = "xml"
        elif "INSERT INTO" in stripped.upper():
            fmt = "sql"
        else:
            fmt = "csv"

    # Handle SQL by converting to CSV first
    if fmt == "sql":
        records = _parse_sql(text)
        if not records:
            raise HTTPException(status_code=400, detail="No INSERT INTO statements found in SQL file.")
        import csv as csv_module
        output = io.StringIO()
        writer = csv_module.DictWriter(output, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        text = output.getvalue()
        fmt = "csv"

    # Run the pipeline — ALL logic is in main.py
    try:
        result = pipeline.run(
            raw_data=text,
            fmt=fmt,
            patient_meta=form_meta,
            original_file=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "format_detected": ext if ext in fmt_map else fmt,
        "filename": filename,
        "patient": result.get("patient_extracted", {}),
        "fhir_bundle": result["fhir_bundle"],
        "dri_result": result["dri_result"],
        "alert": result["alert"],
        "mapped_fields": result["mapped_fields"],
        "validation": result.get("validation", {}),
    }


# ─────────────────────────────────────────────
# SQL Parser (lightweight, no DB needed)
# ─────────────────────────────────────────────

def _parse_sql_values(values_str: str) -> list[str]:
    values = []
    current = ""
    in_quotes = False
    quote_char = ""
    for i, ch in enumerate(values_str):
        if not in_quotes and ch in ("'", '"'):
            in_quotes = True
            quote_char = ch
        elif in_quotes and ch == quote_char:
            if i + 1 < len(values_str) and values_str[i + 1] == quote_char:
                current += ch
            else:
                in_quotes = False
        elif ch == "," and not in_quotes:
            values.append(current.strip())
            current = ""
        else:
            current += ch
    values.append(current.strip())
    return [("" if v.upper() == "NULL" else v) for v in values]


def _parse_sql(text: str) -> list[dict]:
    records = []
    insert_re = re.compile(
        r"INSERT\s+INTO\s+\w+\s*\(([^)]+)\)\s*VALUES\s*([\s\S]*?)(?:;|$)",
        re.IGNORECASE,
    )
    value_group_re = re.compile(r"\(([^)]*)\)")
    for match in insert_re.finditer(text):
        columns = [c.strip().strip("`\"'[]") for c in match.group(1).split(",")]
        for vm in value_group_re.finditer(match.group(2)):
            values = _parse_sql_values(vm.group(1))
            record = {col: (values[i] if i < len(values) else "") for i, col in enumerate(columns)}
            records.append(record)
    return records
