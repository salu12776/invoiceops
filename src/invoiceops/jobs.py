"""Job manager for the web app: runs the LangGraph pipeline in the background.

- One worker thread processes invoices one at a time (free-tier rate limits favour this anyway).
- graph.stream(...) reports every agent step, so the web page can show agents working live.
- When the graph pauses for a human, the job waits in the review queue until someone clicks
  Approve / Reject; the same thread_id then resumes the paused graph.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from .graph import build_graph, new_state

ACTIVE = {"queued", "running", "resuming"}


class JobManager:
    def __init__(self, graph=None) -> None:
        self.graph = graph or build_graph()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------ public API
    def submit(self, path: str | Path, name: str) -> str:
        jid = uuid4().hex[:8]
        state = new_state(path)
        state["source"] = f"{name} ({jid})"  # unique, so a re-upload is caught as a duplicate
        with self.lock:
            self.jobs[jid] = {
                "id": jid, "name": name, "status": "queued", "current": None, "steps": [],
                "decision": None, "flags": [], "data": None, "tokens": 0, "review": None,
                "error": None, "created": time.time(), "finished": None,
            }
        self.queue.put((jid, state))
        return jid

    def review(self, jid: str, approved: bool, note: str = "") -> None:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None:
                raise KeyError(jid)
            if job["status"] != "needs_review":
                raise ValueError("This invoice is not waiting for review")
            job["status"] = "resuming"
        self.queue.put((jid, Command(resume={"approved": approved, "note": note or ""})))

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values()), key=lambda j: j["created"], reverse=True)

    def clear(self) -> None:
        with self.lock:
            for jid in [j for j, job in self.jobs.items() if job["status"] not in ACTIVE]:
                del self.jobs[jid]

    # ------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            jid, graph_input = self.queue.get()
            try:
                self._run(jid, graph_input)
            except Exception as exc:  # never let one bad file kill the worker
                with self.lock:
                    job = self.jobs[jid]
                    job.update(status="error", current=None, error=str(exc)[:300], finished=time.time())

    def _run(self, jid: str, graph_input: Any) -> None:
        config = {"configurable": {"thread_id": jid}}
        with self.lock:
            self.jobs[jid]["status"] = "running"
        outcome = None
        for chunk in self.graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                with self.lock:
                    job = self.jobs[jid]
                    if node == "__interrupt__":
                        job.update(status="needs_review", current="human_review")
                        return
                    update = update or {}
                    if node == "supervisor":
                        job["current"] = update.get("next")
                    outcome = update.get("status", outcome)
                    self._apply(job, update)
        with self.lock:
            self.jobs[jid].update(status=outcome or "done", current=None, finished=time.time())

    @staticmethod
    def _apply(job: dict[str, Any], update: dict[str, Any]) -> None:
        if "steps" in update:
            job["steps"] = update["steps"]
        if update.get("extraction") is not None and job["data"] is None:
            job["data"] = update["extraction"]
        if "validation" in update and update["validation"]:
            v = update["validation"]
            job.update(decision=v["decision"], flags=v["flags"], data=v["data"] or job["data"])
        for key in ("tokens", "error", "review"):
            if update.get(key) is not None:
                job[key] = update[key]