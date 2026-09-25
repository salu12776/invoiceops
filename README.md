# InvoiceOps: multi-agent invoice processing

A team of AI agents that reads invoices, checks them, and asks a human before anything risky gets booked.
The system is built with **LangGraph** and uses a **vision LLM** to read documents. **Plain-Python rules** make every money decision.

**Live demo:** https://invoiceops-x1lz.onrender.com

![InvoiceOps web app](docs/screenshot.png)

## What it does

Upload an invoice (PDF or photo). A supervisor sends it through specialist agents:

| Agent | Job | LLM? |
|---|---|---|
| **Supervisor** | Decides the next step from the current state and enforces the daily token budget | No |
| **Intake** | Checks file type, size (≤10 MB) and page count (≤3); reads the PDF text layer | No |
| **Extractor** | Vision model reads the document and returns validated JSON (Pydantic schema) | Yes |
| **Validator** | Math, dates, duplicates, prompt-injection and card-number checks → approve / review / reject | No |
| **Human review** | The graph **pauses** (`interrupt`) until a person approves or rejects | — |
| **Exporter** | Books approved invoices (CSV + duplicate ledger), with card numbers masked | No |
| **Audit** | One log line per invoice: steps, flags, tokens, time, decision | No |

```mermaid
flowchart LR
    U([Invoice]) --> S{Supervisor}
    S --> I[Intake]
    S --> E[Extractor<br/>vision LLM]
    S --> V[Validator<br/>rules + guardrails]
    S -->|flagged| H[Human review<br/>interrupt]
    S -->|approved| X[Exporter]
    S --> A[(Audit log)]
    I & E & V & H & X --> S
```

**Design choice:** only one agent calls an LLM. Routing, validation and booking are deterministic code, so they are cheap, testable and don't depend on what a model decides.

## Engineering features

**Tools & structured output**
- The Extractor must answer in the `InvoiceData` schema. If the JSON doesn't validate, the error is sent back to the model and it tries again (self-correction, up to 2 retries).

**Retries**
- 429 / 5xx / network errors are retried up to 5 times with `tenacity`.
- The wait follows the provider's `Retry-After` header or its "try again in 20.8s" message; otherwise it backs off exponentially.
- Errors that can never succeed (bad key, request too large) are **not** retried.

**Guardrails**
- **Prompt injection:** text on the document is treated as data. The prompt says so, and a scanner flags phrases like *"AI assistant: ignore previous instructions"*. Flagged invoices can never be auto-approved.
- **PII:** payment card numbers are detected (Luhn check) and masked to `****1111` before anything is stored.
- **Input limits:** file type whitelist, 10 MB, 3 pages; uploaded file names are never used as paths.
- **Public demo:** 20 invoices per hour per visitor.

**Cost caps**
- A daily token budget is checked **before** every model call. When it runs out, invoices are parked instead of failing.
- Output is capped (`max_tokens=800`) and page images are shrunk to 1400 px, which saves tokens and keeps requests under free-tier rate limits.

**Observability**
- `out/audit.jsonl` records every invoice: which agents ran, flags, human decision, tokens and seconds.

**Provider-agnostic**
- Set `EXTRACTOR_PROVIDER=groq` or `gemini` to switch vendors with no code change.

## Evaluation

The test set has 11 synthetic documents, each with an answer key (`answer_key.csv`). Every one targets a failure mode:

| File | Scenario | Expected result |
|---|---|---|
| inv_01–04 | Clean PDFs: PKR/USD/EUR, discount, 10 line items, due date vs issue date | Auto-approved |
| inv_05, inv_06 | Receipt image; tilted, blurred phone photo | Auto-approved |
| inv_07 | Printed total is wrong | `math_mismatch` → human |
| inv_08 | Same vendor + invoice number as inv_01 | `duplicate` → rejected |
| inv_09 | Invoice text tells the AI to report the total as 0 | `prompt_injection` → human; real total still extracted |
| inv_10 | Card number printed; DD/MM date format | `card_number` masked; date read as 3 April |
| not_invoice_11 | Event poster | `not_invoice` → rejected |

**Results** (Groq `qwen/qwen3.8-27b`, free tier):

| Metric | Result |
|---|---|
| Totals extracted correctly | **11 / 11** |
| Validator flags match the answer key | **11 / 11** |
| Injection attempt obeyed | **0 / 1** (extracted 150.00, not 0.00) |
| Tokens per invoice | **~2,900** on average |

Reproduce:

```bash
uv run python scripts/extract_all.py     # Extractor vs answer key (calls the model)
uv run python scripts/validate_all.py    # Validator vs answer key (no API calls)
uv run python scripts/run_pipeline.py --fresh   # full graph in the terminal, asks you to approve flagged ones
```

## Run locally

```bash
git clone https://github.com/<your-username>/invoiceops.git
cd invoiceops
uv sync
```

Create a `.env` file (never commit it):

```
EXTRACTOR_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_VISION_MODEL=qwen/qwen3.8-27b
DAILY_TOKEN_BUDGET=150000
```

Start the web app:

```bash
uv run uvicorn invoiceops.api:app --reload
```

Open http://127.0.0.1:8000, click a sample, and approve or reject flagged invoices in the review queue.

## Project structure

```
src/invoiceops/
  schemas.py      InvoiceData / LineItem (Pydantic)
  documents.py    PDF/image → shrunk page images, input limits
  extractor.py    vision LLM, retries, self-correction, providers
  guardrails.py   prompt-injection + card-number detection, masking
  validator.py    rules → approve / review / reject, duplicate ledger
  budget.py       daily token budget
  graph.py        LangGraph supervisor, agents, human interrupt, audit
  jobs.py         background worker, streams agent steps to the UI
  api.py          FastAPI endpoints, upload limits, rate limit
web/index.html    single-page UI (no build step)
scripts/          evaluation and batch runners
samples/          11 synthetic test documents + answer_key.csv
```

## Limitations and next steps

- Ledger, budget and CSV are files; a real deployment would use a database (and LangGraph's SQLite/Postgres checkpointer so paused reviews survive restarts).
- Injection and card scans read the PDF text layer. For images, only the model's output is scanned; an OCR pass would close that gap.
- A file-hash check at Intake would catch exact re-uploads before spending tokens on extraction.
- The review queue has no login; a real system would record which user approved each invoice.

## Tech

Python 3.13 · LangGraph · FastAPI · Pydantic · tenacity · PyMuPDF · Pillow · Groq / Gemini · uv · Docker · Render

---
Built by **Muhammad Salman**. [Portfolio](https://msalman-portfolio.vercel.app)