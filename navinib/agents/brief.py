"""Brief agent: one pre-call brief per new call, written by the LLM from retrieved, cited sources.

The graph runs one copy per call in parallel. If retrieval or the model fails, the
call keeps its place in the plan with a partial brief built from structured data
only (why the provider ranked high, goal gap, lab alerts, recent calls, who is at
the office). It is marked as having no label or trial sources and makes no product
claims, so nothing in it is unsourced.
"""
import json
import threading

from navinib.agents.llm import citations
from navinib.log import get_logger
from navinib.tools.brief import build_brief
from navinib.tools.providers import location_contacts, provider_lookup, provider_scoring

log = get_logger("navinib.agents.brief")

# Parallel briefs share one SQLite connection and one Chroma client, which aren't safe to
# query at the same moment from several threads. Retrieval is fast, so it takes turns;
# the LLM call, the slow part, still runs in parallel.
_RETRIEVAL = threading.Lock()

SYSTEM = """You write a pre-call brief for a pharmaceutical field rep, at most 150 words.
Use only the facts in the input. Put the source id in square brackets after each fact, e.g. [N-0388].
Talking points may come only from the approved label (refs starting 'Label') or approved trials.
Never mention anything listed in do_not_raise; if the provider may ask about it, say to refer
them to Medical Information. The only drug name you may use is Product X.
If feedback is given, it is what Compliance rejected last time: fix exactly that."""


def brief(payload: dict, config: dict) -> dict:
    """Write the brief for one call (payload: visit, rep_id, day, round, feedback)."""
    cfg = config["configurable"]
    v = payload["visit"]
    pid = v["provider_id"]
    try:
        if cfg.get("col") is None:
            raise RuntimeError("retrieval index not available")
        with _RETRIEVAL:
            inputs = build_brief(cfg["con"], cfg["col"], pid, payload["day"], payload["rep_id"])
    except Exception as e:  # retrieval down: structured facts only
        return _partial(cfg["con"], payload, f"retrieval failed: {e}")

    allowed = sorted({n["ref"] for n in inputs["call_notes"]} | {t["ref"] for t in inputs["talking_points"]}
                     | {p["pub_id"] for p in inputs["publications"] if p["may_discuss"]})
    user = json.dumps({"visit": v, "inputs": inputs, "feedback": payload.get("feedback")}, default=str)
    try:
        text = cfg["llm"].complete("brief", SYSTEM, user,
                                   data={"visit": v, "inputs": inputs, "feedback": payload.get("feedback")})
    except Exception as e:  # model down: same structured fallback
        return _partial(cfg["con"], payload, f"model unavailable: {e}")
    log.info("brief %s round %d: %d words, cites %s", pid, payload["round"], len(text.split()), citations(text))
    return {"briefs": {pid: {"status": "ok", "text": text, "allowed_refs": allowed,
                             "do_not_raise": [d["ref"] for d in inputs["do_not_raise"]], "round": payload["round"]}},
            "trace": [f"brief {pid}: written (round {payload['round']}, cites {', '.join(citations(text))})"]}


def _partial(con, payload: dict, why: str) -> dict:
    """A brief from structured data only, for when retrieval or the model is down. It cites nothing and claims nothing."""
    v, day, rep_id = payload["visit"], payload["day"], payload["rep_id"]
    pid = v["provider_id"]
    with _RETRIEVAL:   # same shared connection as the other parallel briefs
        p = con.execute("SELECT full_name, tier FROM provider WHERE provider_id = ?", (pid,)).fetchone()
        rows = [r for r in provider_lookup(con, rep_id, day) if r["provider_id"] == pid]
        reasons = provider_scoring(con, rows, day)[0]["reasons"] if rows else []
        calls = con.execute("SELECT date, call_type, outcome FROM visit WHERE provider_id = ? AND date < ?"
                            " ORDER BY date DESC LIMIT 3", (pid, day)).fetchall()
        contacts = location_contacts(con, v["location_id"])
    how = "Visit" if v["call_type"] == "face_to_face" else "Phone call with"
    lines = [f"{how} {p['full_name']} (tier {p['tier']}) at {v['start']}.",
             "Partial brief: label and trial sources unavailable, so no product claims for this call."]
    if reasons:
        lines.append("Why this provider: " + "; ".join(reasons) + ".")
    lines.append("Recent calls: " + ("; ".join(f"{c['date']} {c['call_type'].replace('_', ' ')} ({c['outcome']})"
                                               for c in calls) or "none") + ".")
    lines += [f"Also at the office: {c['full_name']} ({c['role'].replace('_', ' ')})." for c in contacts]
    log.warning("brief %s partial: %s", pid, why)
    return {"briefs": {pid: {"status": "partial", "text": "\n".join(lines), "why": why, "allowed_refs": [],
                             "do_not_raise": [], "round": payload["round"]}},
            "trace": [f"brief {pid}: partial, structured data only ({why})"]}
