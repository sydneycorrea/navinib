"""Route and Replan agent: the morning plan (OR-Tools) and the cancellation replan (Tree-of-Thought).

Merged into one agent (Checkpoint 5.1) because both need the same state: the
day's schedule, where the rep is, office hours and drive times. No LLM.
"""
import sqlite3

from navinib.config import VISIT_MIN
from navinib.log import get_logger
from navinib.planner.tot import tree_of_thought
from navinib.tools.drive import DriveTimes
from navinib.tools.providers import drive_minutes
from navinib.tools.route import free_windows, plan_route

log = get_logger("navinib.agents.route_replan")

NEAR_MIN = 20        # morning candidates must be this close to where the window starts
N_CANDIDATES = 8     # candidates per window handed to OR-Tools
UNDERFILL_MIN = VISIT_MIN + 15   # idle minutes in a window that could still hold a visit and its drive


def route_replan(state: dict, config: dict) -> dict:
    """Plan new calls for the request's flow and record the round the plan was made in."""
    cfg = config["configurable"]
    con, drive = cfg["con"], cfg.get("drive") or DriveTimes()
    req = state["request"]
    exclude = set(state.get("feedback", {}).get("exclude", []))
    if req["flow"] == "cancellation":
        plan = _replan(con, req, drive, exclude)
    else:
        plan = _morning(con, req, state["schedule"], state["targets"]["ranked"], drive, exclude)
    plan["round"] = state.get("round", 0)
    plan["refill"] = state.get("refill", {}).get("round", 0)
    line = (f"route_replan ({req['flow']}): {len(plan['visits'])} new call(s): "
            + ("; ".join(f"{v['start']} {v['full_name']}" for v in plan["visits"]) or "none"))
    log.info(line)
    return {"plan": plan, "trace": [line]}


def _morning(con: sqlite3.Connection, req: dict, schedule: dict, targets: list, drive: DriveTimes,
             exclude: set[str]) -> dict:
    """Fill each free window of the day with OR-Tools, from where the rep is when it opens.

    Also reports which providers didn't fit (office hours or drive time) and whether
    a window was left with room for another visit, so the Coordinator can ask
    Targeting for the next candidates once.
    """
    visits, dropped, planned, underfilled = [], [], set(exclude), False
    for w in free_windows(schedule, now_min=req.get("now_min")):
        start = _point(con, req["rep_id"], w["last_location_id"])
        cands = [t for t in targets if t["provider_id"] not in planned
                 and drive_minutes(start["lat"], start["lon"], t["lat"], t["lon"]) <= NEAR_MIN][:N_CANDIDATES]
        r = plan_route(start, w["start_min"], w["end_min"], cands, drive) if cands else {"visits": [], "dropped": []}
        for v in r["visits"]:
            planned.add(v["provider_id"])
            visits.append({**v, "call_type": "face_to_face"})
        dropped += r["dropped"]
        used = sum(v["drive_min"] + VISIT_MIN for v in r["visits"])
        if w["end_min"] - w["start_min"] - used >= UNDERFILL_MIN:
            underfilled = True
    didnt_fit = sorted({d["provider_id"] for d in dropped if not d["reason"].startswith("fits on its own")})
    return {"flow": "morning", "visits": visits, "dropped": dropped, "didnt_fit": didnt_fit,
            "underfilled": underfilled, "alternatives": [], "reason": None, "drive_source": drive.source}


def _replan(con: sqlite3.Connection, req: dict, drive: DriveTimes, exclude: set[str]) -> dict:
    """Tree-of-Thought replan of the window the cancelled call freed."""
    r = tree_of_thought(con, req["rep_id"], req["day"], req["event_id"], req["now_min"], drive,
                        at_location_id=req.get("at_location_id"), exclude=exclude)
    return {"flow": "cancellation", "window": r["window"], "start_location_id": r["start_location_id"],
            "start_reason": r["start_reason"], "visits": r["chosen"]["steps"], "alternatives": r["alternatives"],
            "reason": r["reason"], "drive_source": r["drive_source"]}


def _point(con: sqlite3.Connection, rep_id: str, location_id: str | None) -> dict:
    """A location as a route point, or the rep's home when there is no earlier call."""
    if location_id:
        loc = con.execute("SELECT * FROM location WHERE location_id = ?", (location_id,)).fetchone()
        return {"id": location_id, "lat": loc["lat"], "lon": loc["lon"]}
    rep = con.execute("SELECT * FROM rep WHERE rep_id = ?", (rep_id,)).fetchone()
    return {"id": f"home:{rep_id}", "lat": rep["home_lat"], "lon": rep["home_lon"]}
