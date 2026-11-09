"""Brief agent: one pre-call brief per new call, written by the LLM from retrieved, cited sources.

The graph runs one copy per call in parallel. If retrieval fails, the call keeps
its place in the plan with 'brief unavailable' instead of failing the run.
"""
import json
import threading

from navinib.agents.llm import citations
from navinib.log import get_logger
from navinib.tools.brief import build_brief

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
If feedback is given, it is what a compliance check rejected last time: fix exactly that."""


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
    except Exception as e:  # retrieval down: keep the call, say the brief is missing
        log.warning("brief %s unavailable: %s", pid, e)
        return {"briefs": {pid: {"status": "unavailable", "text": "Brief unavailable.", "error": str(e),
                                 "round": payload["round"]}},
                "trace": [f"brief {pid}: unavailable ({e})"]}

    allowed = sorted({n["ref"] for n in inputs["call_notes"]} | {t["ref"] for t in inputs["talking_points"]}
                     | {p["pub_id"] for p in inputs["publications"] if p["may_discuss"]})
    user = json.dumps({"visit": v, "inputs": inputs, "feedback": payload.get("feedback")}, default=str)
    text = cfg["llm"].complete("brief", SYSTEM, user,
                               data={"visit": v, "inputs": inputs, "feedback": payload.get("feedback")})
    log.info("brief %s round %d: %d words, cites %s", pid, payload["round"], len(text.split()), citations(text))
    return {"briefs": {pid: {"status": "ok", "text": text, "allowed_refs": allowed,
                             "do_not_raise": [d["ref"] for d in inputs["do_not_raise"]], "round": payload["round"]}},
            "trace": [f"brief {pid}: written (round {payload['round']}, cites {', '.join(citations(text))})"]}
