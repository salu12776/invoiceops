"""Validator agent: plain Python rules, no LLM.

The model reads the invoice; code checks it. Rules are deterministic, free and testable,
so money decisions never depend on a model's mood.

Each problem becomes a Flag. The decision is:
- reject  -> not an invoice, or a duplicate (must never be booked)
- review  -> a human must look (math wrong, injection attempt, bad date, missing fields)
- approve -> clean; can be exported automatically
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from .guardrails import find_cards, find_injection, mask_cards
from .schemas import InvoiceData

MONEY_TOLERANCE = 1.0  # rupees/dollars; allows rounding on printed totals

REJECT = {"not_invoice", "duplicate"}
INFO = {"card_number"}  # handled automatically (masked), no human needed


@dataclass
class Flag:
    code: str
    message: str


@dataclass
class ValidationResult:
    decision: str                      # approve | review | reject
    flags: list[Flag] = field(default_factory=list)
    data: InvoiceData | None = None    # cleaned copy (card numbers masked)

    @property
    def codes(self) -> set[str]:
        return {f.code for f in self.flags}

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "flags": [asdict(f) for f in self.flags],
            "data": self.data.model_dump() if self.data else None,
        }


class Ledger:
    """Remembers invoices already booked (vendor + invoice number) to catch duplicates.

    Stored as a small JSON file; later this becomes a database table.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.seen: dict[str, str] = {}
        if self.path and self.path.exists():
            self.seen = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(data: InvoiceData) -> str | None:
        if not data.vendor or not data.invoice_number:
            return None
        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
        return f"{norm(data.vendor)}|{norm(data.invoice_number)}"

    def find(self, data: InvoiceData) -> str | None:
        k = self.key(data)
        return self.seen.get(k) if k else None

    def add(self, data: InvoiceData, source: str) -> None:
        k = self.key(data)
        if k:
            self.seen[k] = source
            if self.path:
                self.path.write_text(json.dumps(self.seen, indent=2), encoding="utf-8")


def _check_math(d: InvoiceData, flags: list[Flag]) -> None:
    if d.total is None:
        flags.append(Flag("missing_fields", "Total not found"))
        return
    extra = d.other_total  # Python adds the listed charges; the model never does arithmetic
    if d.subtotal is not None:
        expected = d.subtotal - (d.discount or 0) + (d.tax or 0) + extra
        if abs(expected - d.total) > MONEY_TOLERANCE:
            listed = " + ".join(f"{c.label} {c.amount:,.2f}" for c in d.other_charges) or "none"
            flags.append(Flag("math_mismatch",
                              f"subtotal - discount + tax + other charges ({listed}) = {expected:,.2f} "
                              f"but printed total is {d.total:,.2f}"))
    amounts = [li.amount for li in d.line_items if li.amount is not None]
    if d.subtotal is not None and amounts and len(amounts) == len(d.line_items):
        # Some documents list every charge as a line, so the lines may add up to the subtotal,
        # to subtotal + extra charges, or to the grand total. Any of these is consistent.
        items = sum(amounts)
        ok = [d.subtotal, d.subtotal + extra, d.total]
        if all(abs(items - x) > MONEY_TOLERANCE for x in ok):
            flags.append(Flag("math_mismatch",
                              f"line items add up to {items:,.2f} but subtotal is {d.subtotal:,.2f}"))


def _check_date(d: InvoiceData, flags: list[Flag], today: date) -> None:
    if not d.date:
        flags.append(Flag("missing_fields", "Invoice date not found"))
        return
    try:
        issued = datetime.strptime(d.date, "%Y-%m-%d").date()
    except ValueError:
        flags.append(Flag("bad_date", f"Date '{d.date}' is not YYYY-MM-DD"))
        return
    if issued > today:
        flags.append(Flag("bad_date", f"Invoice date {issued} is in the future"))


def validate(data: InvoiceData | None, ledger: Ledger, source: str = "",
             doc_text: str = "", today: date | None = None) -> ValidationResult:
    """Run every rule. doc_text is the PDF text layer (used for injection and card scans)."""
    today = today or date.today()
    flags: list[Flag] = []

    if data is None:
        return ValidationResult("review", [Flag("extraction_failed", "Extractor returned nothing")])
    if not data.is_invoice:
        return ValidationResult("reject", [Flag("not_invoice", "Document is not an invoice")], data)

    # Guardrails on the raw text + everything the model returned
    model_text = data.model_dump_json()
    hits = find_injection(doc_text) + find_injection(model_text)
    if hits:
        flags.append(Flag("prompt_injection", f"Instructions found in document: {hits[0]!r}"))
    cards = find_cards(doc_text) + find_cards(model_text)
    if cards:
        flags.append(Flag("card_number", f"Card number found and masked ({cards[0]})"))
        data = InvoiceData.model_validate_json(mask_cards(model_text))

    # Clean up the invoice number: "#BL-1087" / "No. BL-1087" -> "BL-1087"
    if data.invoice_number:
        clean = data.invoice_number.strip().removeprefix("No.").strip().lstrip("#").strip()
        data = data.model_copy(update={"invoice_number": clean})

    for name in ("vendor", "invoice_number"):
        if not getattr(data, name):
            flags.append(Flag("missing_fields", f"{name} not found"))
    _check_math(data, flags)
    _check_date(data, flags, today)

    first = ledger.find(data)
    if first and first != source:
        flags.append(Flag("duplicate", f"Same vendor + invoice number as {first}"))

    codes = {f.code for f in flags}
    if codes & REJECT:
        decision = "reject"
    elif codes - INFO:
        decision = "review"
    else:
        decision = "approve"
    if decision != "reject":
        ledger.add(data, source)
    return ValidationResult(decision, flags, data)