"""Run the Extractor on every file in samples/ and compare totals with answer_key.csv.

Usage:  uv run python scripts/extract_all.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from invoiceops.documents import DocumentError, load_pages
from invoiceops.extractor import Extractor

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT = ROOT / "out"
PAUSE_SECONDS = 2  # the extractor also waits whenever the provider says "slow down"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    key = {r["file"]: r for r in csv.DictReader((ROOT / "answer_key.csv").open(encoding="utf-8"))}
    extractor = Extractor()
    results, total_in, total_out = {}, 0, 0

    print(f"{'file':<20} {'expected':>12} {'got':>12}  ok   tokens")
    for path in sorted(SAMPLES.iterdir()):
        if path.name.startswith("."):
            continue
        try:
            pages = load_pages(path)
        except DocumentError as exc:
            print(f"{path.name:<20} skipped: {exc}")
            continue
        try:
            r = extractor.extract(pages)
        except Exception as exc:  # one failed file must not stop the batch
            print(f"{path.name:<20} FAILED: {str(exc)[:90]}")
            results[path.name] = {"data": None, "error": str(exc)[:500]}
            continue
        total_in += r.input_tokens
        total_out += r.output_tokens
        exp = key.get(path.name, {})
        got = r.data.model_dump() if r.data else None
        results[path.name] = {"data": got, "tokens": [r.input_tokens, r.output_tokens], "attempts": r.attempts, "errors": r.errors}

        if exp.get("is_invoice") == "False":
            ok = got is not None and got["is_invoice"] is False
            print(f"{path.name:<20} {'not invoice':>12} {('not invoice' if ok else 'INVOICE?'):>12}  {'✓' if ok else '✗'}   {r.input_tokens + r.output_tokens}")
        else:
            want = float(exp["total"]) if exp.get("total") else None
            have = got["total"] if got else None
            ok = want is not None and have is not None and abs(want - have) < 0.01
            print(f"{path.name:<20} {want!s:>12} {have!s:>12}  {'✓' if ok else '✗'}   {r.input_tokens + r.output_tokens}")
        time.sleep(PAUSE_SECONDS)

    (OUT / "extractions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nTokens used: {total_in} in + {total_out} out. Full results: out/extractions.json")


if __name__ == "__main__":
    main()