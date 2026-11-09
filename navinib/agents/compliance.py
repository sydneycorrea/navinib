"""Compliance check agent (the Plan validator): rules first, then the LLM for claims.

Every problem names the agent that caused it, so the Coordinator can send the
work back there. Compliance never edits a plan or a brief itself.
"""
import json
import sqlite3
from datetime import date

from navinib.agents.llm import citations
from navinib.log import get_logger

log = get_logger("navinib.agents.compliance")

CLAIMS_SYSTEM = """You check a pre-call brief for a pharmaceutical field rep against its sources.
List each statement that is not supported by the sources, one per line.
Reply with exactly PASS if every statement is supported."""


def compliance(state: dict, config: dict) -> dict:
    """Check the plan's timing and each brief; record the round that was checked."""
    cfg = config["configurable"]
    req, plan = state["request"], state["plan"]
    issues = _plan_issues(cfg["con"], req["day"], state["schedule"], plan["visits"])
    for v in plan["visits"]:
        b = state.get("briefs", {}).get(v["provider_id"])
        if b and b["status"] == "ok":
            issues += _brief_issues(cfg["llm"], v["provider_id"], b)
    result = {"passed": not issues, "issues": issues, "round": state.get("round", 0)}
    line = f"compliance round {result['round']}: " + ("passed" if not issues else
                                                    "; ".join(f"{i['agent']} {i.get('provider_id', '')}: "
                                                              f"{i['problem']}" for i in issues))
    log.info(line)
    return {"compliance": result, "trace": [line]}


def _plan_issues(con: sqlite3.Connection, day: str, schedule: dict, visits: list) -> list:
    """Each new call must sit inside its office's hours and not overlap a booked event or another call."""
    weekday = date.fromisoformat(day).weekday()
    busy = [(e["start_min"], e["end_min"], e["title"]) for e in schedule["events"] if e["status"] != "cancelled"]
    issues = []
    for v in visits:
        s, e = _minutes(v["start"]), _minutes(v["end"])
        if v["call_type"] == "face_to_face":
            h = con.execute("SELECT open_min, close_min FROM location_hours WHERE location_id = ? AND weekday = ?",
                            (v["location_id"], weekday)).fetchone()
            if h is None or s < h["open_min"] or e > h["close_min"]:
                issues.append({"agent": "route_replan", "provider_id": v["provider_id"],
                               "problem": f"{v['start']}-{v['end']} is outside office hours"})
        for bs, be, title in busy:
            if s < be and bs < e:
                issues.append({"agent": "route_replan", "provider_id": v["provider_id"],
                               "problem": f"{v['start']}-{v['end']} overlaps {title}"})
        busy.append((s, e, v["full_name"]))
    return issues


def _brief_issues(llm, pid: str, b: dict) -> list:
    """No off-label item may appear, every citation must be a source it was given, then the LLM claims check."""
    cited = citations(b["text"])
    issues = [{"agent": "brief", "provider_id": pid, "problem": f"mentions do-not-raise item {ref}"}
              for ref in b["do_not_raise"] if ref in b["text"]]
    issues += [{"agent": "brief", "provider_id": pid, "problem": f"cites {ref}, which is not one of its sources"}
               for ref in cited if ref not in b["allowed_refs"] and ref not in b["do_not_raise"]]
    if issues:
        return issues
    answer = llm.complete("claims", CLAIMS_SYSTEM, json.dumps({"brief": b["text"], "sources": b["allowed_refs"]}),
                          data={"brief": b})
    if not answer.strip().startswith("PASS"):
        issues.append({"agent": "brief", "provider_id": pid, "problem": f"unsupported claim: {answer.strip()[:200]}"})
    return issues


def _minutes(hhmm: str) -> int:
    """'14:10' -> 850."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)
