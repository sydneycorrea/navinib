"""Structured run log (Checkpoint 6.1 runtime monitoring): one JSON line per agent run.

Each line has a trace id and everything a compliance review would ask for: the
request (personal data redacted), the model, the policy and label versions, a hash
of every prompt, each check's result, the lane each output went to, and the trace of
agent steps. logs/runs.jsonl is append-only and is read by the evaluation and the
regional manager's weekly sample.
"""
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from navinib import config, guardrails
from navinib.log import get_logger

log = get_logger("navinib.audit")

RUNS_PATH = config.ROOT / "logs" / "runs.jsonl"


def prompt_versions() -> dict:
    """A short hash of each system prompt, so a change in behaviour can be traced to a prompt edit."""
    from navinib.agents import brief, compliance, coordinator   # late import: agents import this module's users
    prompts = {"brief": brief.SYSTEM, "claims": compliance.CLAIMS_SYSTEM, "summary": coordinator.SUMMARY_SYSTEM,
               "adverse_event": guardrails.AE_SYSTEM}
    return {k: hashlib.sha1(v.encode()).hexdigest()[:8] for k, v in prompts.items()}


def new_run_id() -> str:
    """A trace id for one run."""
    return uuid.uuid4().hex[:12]


def record(run_id: str, request: dict, final: dict, model: str, started: float,
           path: Path | str | None = None) -> dict:
    """Append one run to the JSON log and return the entry."""
    req = dict(request)
    if req.get("message"):
        req["message"] = guardrails.redact(req["message"])[0]
    briefs = {pid: {"status": b["status"], "checks": b.get("checks", {}), "label_versions": b.get("label_versions")}
              for pid, b in (final.get("briefs") or {}).items()}
    entry = {
        "run_id": run_id,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 3),
        "request": req,
        "model": model,
        "policy_version": config.POLICY["version"],
        "label_version": guardrails.current_label(),
        "prompts": prompt_versions(),
        "status": final.get("status"),
        "rounds": final.get("round", 0),
        "visits": [f"{v['start']} {v['provider_id']} {v['call_type']}" for v in (final.get("plan") or {}).get("visits", [])],
        "compliance": final.get("compliance"),
        "flags": final.get("flags", []),
        "lanes": final.get("lanes", []),
        "escalations": final.get("escalations", []),
        "briefs": briefs,
        "trace": final.get("trace", []),
    }
    out = Path(path or RUNS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    log.info("run %s logged: %s in %.2fs", run_id, entry["status"], entry["seconds"])
    return entry


def read(path: Path | str | None = None) -> list[dict]:
    """Every logged run, oldest first."""
    p = Path(path or RUNS_PATH)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
