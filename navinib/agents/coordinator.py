"""Coordinator agent: the supervisor at the hub. Every agent reports back here, and it decides what runs next.

Morning plan:   Targeting -> Route and Replan -> Brief (one per call, in parallel) -> Compliance
Cancellation:   cancel the call -> Route and Replan (ToT, Targeting skipped) -> Brief (new calls only) -> Compliance
Two links go both ways. If the morning route leaves a window with room for another
visit after dropping providers that didn't fit (office hours or drive time), the
Coordinator asks Targeting once for the next candidates, without the dropped ones, and
reruns Route (at most MAX_REFILLS). A Compliance fail goes back to the agent that caused
it, at most MAX_RETRIES times; anything still wrong is flagged to the rep, never
silently fixed. Every other handoff is one-way.

The routing is a fixed policy, so the flows are predictable and testable. The LLM
writes the rep-facing summary; with a real model it will also read the rep's
free-text request and choose among the allowed next steps.
"""
from navinib.log import get_logger
from navinib.tools.schedule import cancel_event, schedule_lookup

log = get_logger("navinib.agents.coordinator")

MAX_RETRIES = 2
MAX_REFILLS = 1

SUMMARY_SYSTEM = """You are Navinib, a co-pilot for a pharmaceutical field rep. In at most 80 words, tell
the rep what is proposed for their day and why, mention any brief that is unavailable and any compliance
issue that needs their review, and ask them to approve. Use only the data given."""


def coordinator(state: dict, config: dict) -> dict:
    """Decide the next step from the state; on the last step, write the summary for the rep."""
    cfg = config["configurable"]
    req = state["request"]
    if "schedule" not in state:
        return _start(cfg["con"], req)
    refill = state.get("refill", {}).get("round", 0)
    if req["flow"] == "morning" and ("targets" not in state or state["targets"]["refill"] < refill):
        return _go("targeting", "rank providers" if not refill else "next candidates for the underfilled day")
    rnd = state.get("round", 0)
    plan = state.get("plan")
    if plan is None or plan["round"] < state.get("route_round", -1) or plan["refill"] < refill:
        why = ("plan the new calls" if plan is None else f"redo the route (retry {rnd})" if
               plan["round"] < state.get("route_round", -1) else "refill the day with the next candidates")
        return _go("route_replan", why)
    if req["flow"] == "morning" and refill < MAX_REFILLS and plan["underfilled"] and plan["didnt_fit"]:
        return _refill(plan, refill + 1)
    if _briefs_needed(state):
        return _go("brief", f"write {len(_briefs_needed(state))} brief(s)")
    c = state.get("compliance")
    if c is None or c["round"] < rnd:
        return _go("compliance", "check the plan and briefs")
    if not c["passed"] and rnd < MAX_RETRIES:
        return _retry(state, c["issues"], rnd + 1)
    return _finish(state, cfg["llm"], [] if c["passed"] else c["issues"])


def next_step(state: dict) -> str:
    """Edge function: the node named by the Coordinator's decision ('end' when done)."""
    return state["next"]


def briefs_to_write(state: dict) -> list[dict]:
    """Payloads for the calls that still need a brief, one per parallel Brief run."""
    req, rnd = state["request"], state.get("round", 0)
    feedback = state.get("feedback", {}).get("brief", {})
    return [{"visit": v, "rep_id": req["rep_id"], "day": req["day"], "round": rnd,
             "feedback": feedback.get(v["provider_id"])} for v in _briefs_needed(state)]


def _briefs_needed(state: dict) -> list[dict]:
    """Planned calls with no brief yet, or whose brief Compliance sent back this round."""
    briefs, redo = state.get("briefs", {}), state.get("brief_rounds", {})
    return [v for v in state["plan"]["visits"]
            if v["provider_id"] not in briefs or briefs[v["provider_id"]]["round"] < redo.get(v["provider_id"], 0)]


def _start(con, req: dict) -> dict:
    """First step: record a cancellation if there is one, then read the day's schedule."""
    if req["flow"] == "cancellation":
        cancel_event(con, req["event_id"])
    schedule = schedule_lookup(con, req["rep_id"], req["day"])
    nxt = "targeting" if req["flow"] == "morning" else "route_replan"
    line = (f"coordinator: {req['flow']} request" + (f", {req['event_id']} cancelled" if req.get("event_id") else "")
            + f"; {len(schedule['events'])} events today -> {nxt}")
    log.info(line)
    return {"schedule": schedule, "round": 0, "status": "working", "next": nxt, "trace": [line]}


def _go(agent: str, why: str) -> dict:
    """Hand the work to one agent."""
    log.info("coordinator -> %s (%s)", agent, why)
    return {"next": agent, "trace": [f"coordinator -> {agent}: {why}"]}


def _refill(plan: dict, rnd: int) -> dict:
    """Route left room in the day after dropping providers that didn't fit: ask Targeting for the next ones."""
    update = {"refill": {"round": rnd, "exclude": plan["didnt_fit"]}}
    update.update(_go("targeting", f"day underfilled; {', '.join(plan['didnt_fit'])} didn't fit "
                                   f"(refill {rnd} of {MAX_REFILLS})"))
    return update


def _retry(state: dict, issues: list, rnd: int) -> dict:
    """Send each problem back to the agent that caused it, with what Compliance found."""
    brief_fb = {i["provider_id"]: i["problem"] for i in issues if i["agent"] == "brief"}
    route_ids = sorted({i["provider_id"] for i in issues if i["agent"] == "route_replan"})
    fb = dict(state.get("feedback", {}))
    fb["brief"] = {**fb.get("brief", {}), **brief_fb}
    fb["exclude"] = sorted(set(fb.get("exclude", [])) | set(route_ids))
    update = {"round": rnd, "feedback": fb,
              "brief_rounds": {**state.get("brief_rounds", {}), **{pid: rnd for pid in brief_fb}}}
    if route_ids:
        update["route_round"] = rnd
    update.update(_go("route_replan" if route_ids else "brief",
                      f"compliance failed; retry {rnd} of {MAX_RETRIES}"))
    return update


def _finish(state: dict, llm, flags: list) -> dict:
    """Merge the results into one summary for the rep and wait for approval."""
    plan = state["plan"]
    unavailable = [v["full_name"] for v in plan["visits"]
                   if state.get("briefs", {}).get(v["provider_id"], {}).get("status") == "unavailable"]
    data = {"visits": plan["visits"], "reason": plan.get("reason"), "unavailable": unavailable, "flags": flags}
    message = llm.complete("summary", SUMMARY_SYSTEM, str(data), data=data)
    line = f"coordinator: done, {len(plan['visits'])} call(s) proposed, {len(flags)} flag(s) for the rep"
    log.info(line)
    return {"next": "end", "status": "awaiting_approval", "flags": flags, "message": message, "trace": [line]}
