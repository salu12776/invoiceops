"""InvoiceOps as a LangGraph supervisor graph.

                ┌──────────── supervisor ────────────┐
                │  (decides the next step, enforces  │
                │   the cost cap, never calls an LLM)│
                └──┬──────┬───────┬───────┬──────┬───┘
                intake extractor validator human exporter
                   (every worker reports back to the supervisor)

- Intake:     checks the file (type, size, pages) and reads the PDF text layer.
- Extractor:  vision model -> InvoiceData (retries, self-correction, injection-safe prompt).
- Validator:  Python rules -> approve / review / reject + flags.
- Human:      pauses the graph (interrupt) until a person approves or rejects.
- Exporter:   books approved invoices into out/approved.csv and the duplicate ledger.
- Audit:      writes one line per invoice to out/audit.jsonl (observability).
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .budget import Budget
from .documents import DocumentError, load_pages
from .guardrails import document_text
from .schemas import InvoiceData
from .validator import Ledger, validate

OUT = Path(__file__).resolve().parents[2] / "out"
LEDGER_PATH = OUT / "ledger.json"
BUDGET_PATH = OUT / "budget.json"
APPROVED_CSV = OUT / "approved.csv"
AUDIT_LOG = OUT / "audit.jsonl"


class State(TypedDict, total=False):
    path: str
    source: str
    started: float
    pages: int
    doc_text: str
    extraction: dict | None        # InvoiceData as dict
    tokens: int
    validation: dict | None        # ValidationResult.to_dict()
    review: dict | None            # {"approved": bool, "note": str}
    status: str                    # final outcome
    error: str
    steps: list[str]               # which agents ran, in order
    next: str


_extractor = None


def get_extractor():
    """Create the Extractor once (it opens an HTTP client)."""
    global _extractor
    if _extractor is None:
        from .extractor import Extractor
        _extractor = Extractor()
    return _extractor


def _step(state: State, name: str) -> list[str]:
    return [*state.get("steps", []), name]


# ---------------------------------------------------------------- supervisor
def supervisor(state: State) -> dict[str, Any]:
    """Pick the next agent from what is already known. Pure rules: cheap, predictable, testable."""
    if state.get("status"):
        nxt = "audit"
    elif "pages" not in state:
        nxt = "intake"
    elif "extraction" not in state:
        if Budget(BUDGET_PATH).can_afford(state["pages"]):
            nxt = "extractor"
        else:
            return {"next": "audit", "status": "over_budget",
                    "error": "Daily token budget reached; parked for later"}
    elif "validation" not in state:
        nxt = "validator"
    else:
        decision = state["validation"]["decision"]
        if decision == "approve":
            nxt = "exporter"
        elif decision == "reject":
            return {"next": "audit", "status": "rejected"}
        elif state.get("review") is None:
            nxt = "human_review"
        elif state["review"].get("approved"):
            nxt = "exporter"
        else:
            return {"next": "audit", "status": "rejected_by_human"}
    return {"next": nxt}


# ---------------------------------------------------------------- workers
def intake(state: State) -> dict[str, Any]:
    try:
        pages = load_pages(state["path"])
    except DocumentError as exc:
        return {"steps": _step(state, "intake"), "status": "unsupported", "error": str(exc)}
    return {"steps": _step(state, "intake"), "pages": len(pages), "doc_text": document_text(state["path"])}


def extractor(state: State) -> dict[str, Any]:
    pages = load_pages(state["path"])
    try:
        result = get_extractor().extract(pages)
    except Exception as exc:  # provider down after all retries
        return {"steps": _step(state, "extractor"), "extraction": None, "tokens": 0,
                "error": f"Extractor failed: {str(exc)[:200]}"}
    tokens = result.input_tokens + result.output_tokens
    Budget(BUDGET_PATH).charge(tokens)
    data = result.data.model_dump() if result.data else None
    return {"steps": _step(state, "extractor"), "extraction": data, "tokens": tokens}


def validator(state: State) -> dict[str, Any]:
    ledger = Ledger(LEDGER_PATH)
    ledger.path = None  # read-only here: only the Exporter books invoices
    raw = state.get("extraction")
    data = InvoiceData.model_validate(raw) if raw else None
    result = validate(data, ledger, source=state["source"], doc_text=state.get("doc_text", ""))
    return {"steps": _step(state, "validator"), "validation": result.to_dict()}


def human_review(state: State) -> dict[str, Any]:
    """Pause here. The caller resumes with Command(resume={"approved": bool, "note": str})."""
    answer = interrupt({
        "source": state["source"],
        "flags": state["validation"]["flags"],
        "data": state["validation"]["data"],
    })
    return {"steps": _step(state, "human_review"), "review": dict(answer)}


def exporter(state: State) -> dict[str, Any]:
    data = InvoiceData.model_validate(state["validation"]["data"])  # masked copy
    OUT.mkdir(exist_ok=True)
    new_file = not APPROVED_CSV.exists()
    with APPROVED_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["source", "vendor", "invoice_number", "date", "currency", "subtotal",
                        "discount", "tax", "total", "approved_by"])
        w.writerow([state["source"], data.vendor, data.invoice_number, data.date, data.currency,
                    data.subtotal, data.discount, data.tax, data.total,
                    "human" if state.get("review") else "auto"])
    Ledger(LEDGER_PATH).add(data, state["source"])
    return {"steps": _step(state, "exporter"), "status": "exported"}


def audit(state: State) -> dict[str, Any]:
    OUT.mkdir(exist_ok=True)
    validation = state.get("validation") or {}
    line = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "source": state["source"],
        "status": state.get("status"),
        "decision": validation.get("decision"),
        "flags": [f["code"] for f in validation.get("flags", [])],
        "review": state.get("review"),
        "tokens": state.get("tokens", 0),
        "seconds": round(time.time() - state.get("started", time.time()), 1),
        "steps": _step(state, "audit"),
        "error": state.get("error"),
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"steps": line["steps"]}


# ---------------------------------------------------------------- wiring
WORKERS = ["intake", "extractor", "validator", "human_review", "exporter"]


def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("intake", intake)
    g.add_node("extractor", extractor)
    g.add_node("validator", validator)
    g.add_node("human_review", human_review)
    g.add_node("exporter", exporter)
    g.add_node("audit", audit)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], {n: n for n in [*WORKERS, "audit"]})
    for n in WORKERS:
        g.add_edge(n, "supervisor")
    g.add_edge("audit", END)
    return g.compile(checkpointer=InMemorySaver())


def new_state(path: str | Path) -> State:
    path = Path(path)
    return {"path": str(path), "source": path.name, "started": time.time(), "steps": []}