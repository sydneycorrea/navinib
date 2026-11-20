"""Brief agent: one pre-call brief per new call, written by the LLM from retrieved, cited sources.

The graph runs one copy per call in parallel. If retrieval or the model fails, the
call keeps its place in the plan with a partial brief built from structured data
only (why the provider ranked high, goal gap, lab alerts, recent calls, who is at
the office). It is marked as having no label or trial sources and makes no product
claims, so nothing in it is unsourced.

Guardrails (Checkpoint 6.1) run around the model call:
  before  every call note and retrieved passage is screened: personal data redacted,
          a note with an injected instruction quarantined, a note reporting a possible
          adverse event escalated to safety reporting and left out; label passages
          from a superseded label version are dropped
  after   the claim-by-claim check removes any dose, indication or efficacy sentence
          that its cited approved source does not support, and says so in the brief
"""
import json
import threading

from navinib import guardrails
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
Use only the facts in the input. Put the source id in square brackets after each fact, copied exactly, e.g. [N-0388] or [Label §5.1].
Talking points may come only from the approved label (refs starting 'Label') or approved trials.
Never mention anything listed in do_not_raise; if the provider may ask about it, say to refer
them to Medical Information. The only drug name you may use is Product X.
Facts about the call itself (who, tier, time, face to face or phone, office) come from the
schedule: cite them as [Schedule].
If feedback is given, it is what Compliance rejected last time: fix exactly that."""


def schedule_source(con, v: dict, inputs: dict) -> str:
    """The call's own facts from the calendar, as a source the brief can cite as [Schedule]."""
    how = "phone call" if v["call_type"] == "phone" else "face-to-face call"
    row = con.execute("SELECT affiliation, address FROM location WHERE location_id = ?",
                      (v.get("location_id"),)).fetchone() if v.get("location_id") else None
    where = f" at {row['affiliation']}, {row['address']}" if row else ""
    return (f"{inputs.get('full_name', v.get('full_name', ''))}, tier {inputs.get('tier', '?')}: "
            f"{how} {v['start']}-{v['end']}{where}.")


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

    with _RETRIEVAL:   # escalations share the connection too
        inputs, checks = _screen_inputs(cfg["con"], cfg["llm"], inputs, payload)
    with _RETRIEVAL:
        sources = {"Schedule": schedule_source(cfg["con"], v, inputs)}
    sources |= {n["ref"]: n["text"] for n in inputs["call_notes"]}
    sources |= {t["ref"]: t["text"] for t in inputs["talking_points"]}
    sources |= {t["ref"]: t["text"] for t in inputs.get("trials", []) if t["status"] == "approved"}
    sources |= {p["pub_id"]: f"{p['title']}. {p.get('abstract') or ''}" for p in inputs["publications"]
                if p["may_discuss"]}
    allowed = sorted(sources)
    model_inputs = {k: v for k, v in inputs.items() if k != "trials"}
    user = json.dumps({"visit": v, "schedule": sources["Schedule"], "inputs": model_inputs, "feedback": payload.get("feedback")}, default=str)
    try:
        text = cfg["llm"].complete("brief", SYSTEM, user,
                                   data={"visit": v, "inputs": inputs, "feedback": payload.get("feedback")})
    except Exception as e:  # model down: same structured fallback
        return _partial(cfg["con"], payload, f"model unavailable: {e}", checks)
    text, checks["citations_fixed"] = guardrails.normalize_citations(text, sources)
    text, removed = guardrails.claim_check(text, sources)
    checks["claims_removed"] = removed
    log.info("brief %s round %d: %d words, cites %s, %d claim(s) removed", pid, payload["round"],
             len(text.split()), citations(text), len(removed))
    extra = _check_line(checks)
    return {"briefs": {pid: {"status": "ok", "text": text, "allowed_refs": allowed, "sources": sources,
                             "label_versions": sorted({t["version"] for t in inputs["talking_points"]
                                                       if t.get("version")}),
                             "do_not_raise": [d["ref"] for d in inputs["do_not_raise"]], "round": payload["round"],
                             "checks": checks}},
            "trace": [f"brief {pid}: written (round {payload['round']}, cites {', '.join(dict.fromkeys(citations(text))) or 'nothing'})"
                      + extra]}


def _screen_inputs(con, llm, inputs: dict, payload: dict) -> tuple[dict, dict]:
    """Input guardrails on everything the model will read. Returns the cleaned inputs and what was found."""
    pid, rep_id = payload["visit"]["provider_id"], payload["rep_id"]
    checks = {"pii_redacted": [], "quarantined": [], "escalations": [], "superseded_label": []}
    held = guardrails.quarantined(con)
    notes = []
    for n in inputs["call_notes"]:
        if n["ref"] in held:
            checks["quarantined"].append(n["ref"])
            continue
        found = guardrails.screen(n["text"], llm)
        if found["injection"]:
            guardrails.escalate(con, "injection", n["ref"], n["text"], rep_id)
            checks["quarantined"].append(n["ref"])
            continue
        if found["adverse_event"]:
            e = guardrails.escalate(con, "adverse_event", n["ref"], n["text"], rep_id)
            checks["escalations"].append({**e, "provider_id": pid, "why": found["adverse_event"]})
            continue
        if found["pii"]:
            checks["pii_redacted"].append(n["ref"])
        notes.append({**n, "text": found["text"]})
    points = []
    for t in inputs["talking_points"]:
        if guardrails.superseded([t]):
            checks["superseded_label"].append(t["ref"])
        elif guardrails.injection(t["text"]):
            guardrails.escalate(con, "injection", t["ref"], t["text"], rep_id)
            checks["quarantined"].append(t["ref"])
        else:
            points.append(t)
    return {**inputs, "call_notes": notes, "talking_points": points}, checks


def _check_line(checks: dict) -> str:
    """What the guardrails did, for the trace."""
    parts = []
    if checks.get("escalations"):
        parts.append("escalated " + ", ".join(f"{e['source_ref']} ({e['kind'].replace('_', ' ')})"
                                             for e in checks["escalations"]))
    if checks.get("quarantined"):
        parts.append("quarantined " + ", ".join(checks["quarantined"]))
    if checks.get("pii_redacted"):
        parts.append("redacted personal data in " + ", ".join(checks["pii_redacted"]))
    if checks.get("superseded_label"):
        parts.append("dropped superseded label passages " + ", ".join(checks["superseded_label"]))
    if checks.get("claims_removed"):
        parts.append(f"removed {len(checks['claims_removed'])} unsupported claim(s)")
    return f"; guardrails: {'; '.join(parts)}" if parts else ""


def partial_cause(why: str) -> str:
    """Why a brief is partial, in the rep's words."""
    if why.startswith("model"):
        return ("the daily free-model limit is reached" if "daily" in why
                else "the model is switched off" if "kill switch" in why else "the model did not answer")
    return "label and trial sources unavailable"


def _partial(con, payload: dict, why: str, checks: dict | None = None) -> dict:
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
             f"Partial brief: {partial_cause(why)}, so no product claims for this call."]
    if reasons:
        lines.append("Why this provider: " + "; ".join(reasons) + ".")
    lines.append("Recent calls: " + ("; ".join(f"{c['date']} {c['call_type'].replace('_', ' ')} ({c['outcome']})"
                                               for c in calls) or "none") + ".")
    lines += [f"Also at the office: {c['full_name']} ({c['role'].replace('_', ' ')})." for c in contacts]
    log.warning("brief %s partial: %s", pid, why)
    checks = checks or {}
    return {"briefs": {pid: {"status": "partial", "text": "\n".join(lines), "why": why, "allowed_refs": [],
                             "sources": {}, "label_versions": [], "do_not_raise": [], "round": payload["round"],
                             "checks": checks}},
            "trace": [f"brief {pid}: partial, structured data only ({why})" + _check_line(checks)]}
