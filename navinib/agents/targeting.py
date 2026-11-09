"""Targeting agent: who is worth seeing today. Plain tools, no LLM."""
from navinib.log import get_logger
from navinib.tools.providers import provider_lookup, provider_scoring

log = get_logger("navinib.agents.targeting")


def targeting(state: dict, config: dict) -> dict:
    """Rank every provider location open today, leaving out providers already booked.

    On a refill round (Route left the day underfilled), it also leaves out the
    providers Route had to drop, so the next candidates move up.
    """
    con = config["configurable"]["con"]
    req = state["request"]
    refill = state.get("refill", {})
    booked = {e["provider_id"] for e in state["schedule"]["events"] if e["provider_id"]}
    open_today = provider_lookup(con, req["rep_id"], req["day"], exclude=booked | set(refill.get("exclude", [])))
    ranked = provider_scoring(con, open_today, req["day"])
    top = ", ".join(f"{t['full_name']} {t['score']}" for t in ranked[:3])
    line = (f"targeting: ranked {len(ranked)} open locations (top: {top})"
            + (f", leaving out {', '.join(refill['exclude'])} that didn't fit" if refill else ""))
    log.info(line)
    return {"targets": {"ranked": ranked, "refill": refill.get("round", 0)}, "trace": [line]}
