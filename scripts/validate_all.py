"""Run the Validator on out/extractions.json and compare flags with answer_key.csv.

No API calls: it reuses what extract_all.py already extracted.
Usage:  uv run python scripts/validate_all.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from invoiceops.guardrails import document_text
from invoiceops.schemas import InvoiceData
from invoiceops.validator import Ledger, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def main() -> None:
    key = {r["file"]: r for r in csv.DictReader((ROOT / "answer_key.csv").open(encoding="utf-8"))}
    extractions = json.loads((OUT / "extractions.json").read_text(encoding="utf-8"))
    ledger = Ledger()  # fresh each run, so inv_01 is "first" and inv_08 the duplicate
    report, passed = {}, 0

    print(f"{'file':<20} {'decision':<8} {'expected flags':<18} {'got flags':<30} ok")
    for name in sorted(extractions):
        raw = extractions[name].get("data")
        data = InvoiceData.model_validate(raw) if raw else None
        result = validate(data, ledger, source=name, doc_text=document_text(ROOT / "samples" / name))
        want = {f for f in key.get(name, {}).get("expected_flags", "").split(";") if f}
        got = result.codes
        ok = want == got
        passed += ok
        print(f"{name:<20} {result.decision:<8} {','.join(sorted(want)) or '-':<18} {','.join(sorted(got)) or '-':<30} {'✓' if ok else '✗'}")
        for f in result.flags:
            print(f"{'':<20}   · {f.code}: {f.message}")
        report[name] = result.to_dict()

    (OUT / "validations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(extractions)} match the answer key. Full results: out/validations.json")


if __name__ == "__main__":
    main()