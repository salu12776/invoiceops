"""Run the full multi-agent pipeline on invoices, with a human approving flagged ones.

Usage:
  uv run python scripts/run_pipeline.py --fresh                 # all samples, start from a clean ledger
  uv run python scripts/run_pipeline.py samples/inv_07.pdf      # one file
  uv run python scripts/run_pipeline.py --fresh --no-input      # unattended: flagged invoices are rejected
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langgraph.types import Command

from invoiceops.graph import APPROVED_CSV, AUDIT_LOG, LEDGER_PATH, build_graph, new_state

ROOT = Path(__file__).resolve().parents[1]


def ask_human(payload: dict, no_input: bool) -> dict:
    print(f"\n  ⚠  {payload['source']} needs a human decision")
    for f in payload["flags"]:
        print(f"     · {f['code']}: {f['message']}")
    d = payload.get("data") or {}
    print(f"     vendor={d.get('vendor')}  no={d.get('invoice_number')}  date={d.get('date')}  "
          f"total={d.get('total')} {d.get('currency')}")
    if no_input:
        print("     --no-input: rejected automatically")
        return {"approved": False, "note": "no reviewer available"}
    answer = input("     Approve and export? [y/N] ").strip().lower()
    return {"approved": answer == "y", "note": "approved in terminal" if answer == "y" else "rejected in terminal"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--fresh", action="store_true", help="forget previously booked invoices")
    parser.add_argument("--no-input", action="store_true", help="don't ask; reject flagged invoices")
    args = parser.parse_args()

    if args.fresh:
        for p in (LEDGER_PATH, APPROVED_CSV, AUDIT_LOG):
            p.unlink(missing_ok=True)

    files = [Path(f) for f in args.files] or sorted(p for p in (ROOT / "samples").iterdir() if not p.name.startswith("."))
    graph = build_graph()
    summary = []

    for path in files:
        config = {"configurable": {"thread_id": f"{path.name}-{time.time()}"}}
        print(f"▶ {path.name}")
        state = graph.invoke(new_state(path), config)
        while "__interrupt__" in state:  # the graph paused at human_review
            answer = ask_human(state["__interrupt__"][0].value, args.no_input)
            state = graph.invoke(Command(resume=answer), config)
        flags = [f["code"] for f in (state.get("validation") or {}).get("flags", [])]
        print(f"  {' → '.join(state['steps'])}")
        print(f"  = {state.get('status')}  flags={flags or '-'}  tokens={state.get('tokens', 0)}"
              + (f"  error={state['error']}" if state.get("error") else ""))
        summary.append((path.name, state.get("status"), flags, state.get("tokens", 0)))

    print("\n" + "-" * 60)
    for name, status, flags, tokens in summary:
        print(f"{name:<20} {status:<18} {','.join(flags) or '-':<22} {tokens}")
    print(f"\nTotal tokens: {sum(s[3] for s in summary)}")
    print(f"Booked: {APPROVED_CSV.relative_to(ROOT)}   Audit log: {AUDIT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()