"""Targeting agent: who is worth seeing. Plain tools, no LLM.

Two modes. Territory (morning plan): rank every provider location open today.
Nearby (cancellation): rank only providers within VISIT_RADIUS_MIN of where the
rep will be when the freed window opens, plus phone candidates for top-scored
providers no visit can reach. Routing's Tree-of-Thought replan searches that shortlist.
"""
from navinib.log import get_logger
from navinib.planner.tot import candidates, replan_setup
from navinib.tools.drive import DriveTimes
from navinib.tools.providers import provider_lookup, provider_scoring

log = get_logger("navinib.agents.targeting")


def targeting(state: dict, config: dict) -> dict:
    """Rank providers for the request's flow: nearby for a cancellation, the territory otherwise."""
    if state["request"]["flow"] == "cancellation":
        return _nearby(state, config)
    return _territory(state, config)


def _territory(state: dict, config: dict) -> dict:
    """Rank every provider location open today, leaving out providers already booked.

    On a refill round (Routing left the day underfilled), it also leaves out the
    providers Routing had to drop, so the next candidates move up.
    """
    con = config["configurable"]["con"]
    req = state["request"]
    refill = state.get("refill", {})
    booked = {e["provider_id"] for e in state["schedule"]["events"] if e["provider_id"]}
    open_today = provider_lookup(con, req["rep_id"], req["day"], exclude=booked | set(refill.get("exclude", [])))
    ranked = provider_scoring(con, open_today, req["day"])
    top = ", ".join(f"{t['full_name']} {t['score']}" for t in ranked[:3])
    line = (f"targeting (territory): ranked {len(ranked)} open locations (top: {top})"
            + (f", leaving out {', '.join(refill['exclude'])} that didn't fit" if refill else ""))
    log.info(line)
    return {"targets": {"mode": "territory", "ranked": ranked, "refill": refill.get("round", 0)}, "trace": [line]}


def _nearby(state: dict, config: dict) -> dict:
    """Shortlist for the freed window: visits near the rep's start point, phone calls for those out of reach.

    Leaves out providers already booked today, including the one who cancelled.
    """
    cfg = config["configurable"]
    con, drive, req = cfg["con"], cfg.get("drive") or DriveTimes(), state["request"]
    setup = replan_setup(con, req["rep_id"], req["day"], req["event_id"], req["now_min"], req.get("at_location_id"))
    if setup["window"] is None:
        pool, line = {"face_to_face": [], "phone": []}, "targeting (nearby): no open window, nothing to rank"
    else:
        pool = candidates(con, req["rep_id"], req["day"], setup["start"], setup["booked"], setup["window"], drive)
        names = lambda cs: ", ".join(c["full_name"] for c in cs) or "none"   # noqa: E731
        line = (f"targeting (nearby): window {setup['window']['start']}-{setup['window']['end']} from "
                f"{setup['start']['id']}; visit: {names(pool['face_to_face'])}; phone: {names(pool['phone'])}")
    log.info(line)
    return {"targets": {"mode": "nearby", "pool": pool, "ranked": pool["face_to_face"] + pool["phone"],
                        "refill": 0}, "trace": [line]}
