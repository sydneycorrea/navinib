"""Targeting agent: who is worth seeing today. Plain tools, no LLM."""
from navinib.log import get_logger
from navinib.tools.providers import provider_lookup, provider_scoring

log = get_logger("navinib.agents.targeting")


def targeting(state: dict, config: dict) -> dict:
    """Rank every provider location open today, leaving out providers already booked."""
    con = config["configurable"]["con"]
    req = state["request"]
    booked = {e["provider_id"] for e in state["schedule"]["events"] if e["provider_id"]}
    open_today = provider_lookup(con, req["rep_id"], req["day"], exclude=booked)
    ranked = provider_scoring(con, open_today, req["day"])
    top = ", ".join(f"{t['full_name']} {t['score']}" for t in ranked[:3])
    log.info("targeting: %d locations ranked; top %s", len(ranked), top)
    return {"targets": ranked, "trace": [f"targeting: ranked {len(ranked)} open locations (top: {top})"]}
