"""Guardrails: checks on the raw document text that don't need an LLM.

- Prompt-injection scan: finds text that tries to give the AI orders.
- Card-number scan: finds payment card numbers (Luhn-checked) so they can be masked before storing.

PDFs usually have a text layer, so these checks are exact and free. Images have no text layer;
for those we still scan whatever text the Extractor returned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) instructions",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"\b(ai|assistant|chatgpt)( assistant)?\s*:",
    r"you are now",
    r"system prompt",
    r"(mark|set|report)\b.{0,40}\b(approved|as paid|already paid)",
    r"report the total as",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def document_text(path: str | Path) -> str:
    """Text layer of a PDF ('' for images or scanned PDFs)."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return ""
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def find_injection(text: str) -> list[str]:
    """Return the suspicious phrases found in the text."""
    return sorted({m.group(0).strip() for m in _INJECTION_RE.finditer(text or "")})


def _luhn_ok(digits: str) -> bool:
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        double = not double
    return total % 10 == 0


def find_cards(text: str) -> list[str]:
    """Return card numbers found in the text, already masked (e.g. ****1111)."""
    found = []
    for m in _CARD_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("****" + digits[-4:])
    return found


def mask_cards(text: str) -> str:
    """Replace every card number in the text with ****last4."""
    def _mask(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "****" + digits[-4:] if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)
    return _CARD_RE.sub(_mask, text or "")