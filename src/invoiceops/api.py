"""FastAPI web app for InvoiceOps.

Run locally:  uv run uvicorn invoiceops.api:app --reload
Then open:    http://127.0.0.1:8000
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .budget import Budget
from .documents import IMAGE_TYPES, MAX_FILE_MB
from .graph import APPROVED_CSV, BUDGET_PATH, LEDGER_PATH, OUT
from .jobs import JobManager

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
SAMPLES = ROOT / "samples"
UPLOADS = OUT / "uploads"
ALLOWED = IMAGE_TYPES | {".pdf"}
MAX_BYTES = MAX_FILE_MB * 1024 * 1024
RUNS_PER_HOUR = 20  # per visitor IP: protects the API key on a public demo

app = FastAPI(title="InvoiceOps", description="Multi-agent invoice processing with LangGraph")
manager = JobManager()
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    now, window = time.time(), _hits[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RUNS_PER_HOUR:
        raise HTTPException(429, f"Demo limit: {RUNS_PER_HOUR} invoices per hour. Please try later.")
    window.append(now)


def _sample_names() -> list[str]:
    return sorted(p.name for p in SAMPLES.iterdir() if p.suffix.lower() in ALLOWED)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/samples")
def samples() -> list[str]:
    return _sample_names()


@app.get("/api/samples/{name}/file", include_in_schema=False)
def sample_file(name: str) -> FileResponse:
    if name not in _sample_names():
        raise HTTPException(404, "Unknown sample")
    return FileResponse(SAMPLES / name)


@app.post("/api/samples/{name}")
def run_sample(name: str, request: Request) -> dict:
    if name not in _sample_names():  # whitelist: no path tricks
        raise HTTPException(404, "Unknown sample")
    _rate_limit(request)
    return {"id": manager.submit(SAMPLES / name, name)}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Upload a PDF, PNG, JPG or WEBP file")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_FILE_MB} MB")
    _rate_limit(request)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{uuid4().hex}{suffix}"  # never trust the uploaded file name for paths
    path.write_bytes(data)
    name = Path(file.filename or "upload").name[:80]
    return {"id": manager.submit(path, name)}


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return manager.list()


class Review(BaseModel):
    approved: bool
    note: str = ""


@app.post("/api/jobs/{jid}/review")
def review(jid: str, body: Review) -> dict:
    try:
        manager.review(jid, body.approved, body.note[:200])
    except KeyError:
        raise HTTPException(404, "Unknown invoice")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@app.get("/api/stats")
def stats() -> dict:
    budget = Budget(BUDGET_PATH)
    all_jobs = manager.list()
    count = lambda *s: sum(j["status"] in s for j in all_jobs)
    return {
        "tokens_today": budget.used_today(),
        "daily_budget": budget.daily_limit,
        "exported": count("exported"),
        "needs_review": count("needs_review"),
        "rejected": count("rejected", "rejected_by_human"),
        "processing": count("queued", "running", "resuming"),
    }


@app.post("/api/reset")
def reset(request: Request) -> dict:
    """Clear the demo: finished jobs, the duplicate ledger and the booked CSV (budget is kept)."""
    _rate_limit(request)
    manager.clear()
    for p in (LEDGER_PATH, APPROVED_CSV):
        p.unlink(missing_ok=True)
    return {"ok": True}