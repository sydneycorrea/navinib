"""Route optimizer (Action 3 in Checkpoint 2.1).

Two steps. free_windows reads the day's schedule and returns the gaps the rep
can fill, keeping a prep/drive buffer before each booked call. plan_route then
orders candidate visits inside one gap with OR-Tools (routing with time
windows): each visit must start after the office opens and finish before it
closes, and everything must end before the window does. It either returns a
feasible plan or says why each candidate was left out; it never guesses.
"""
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from navinib import config
from navinib.log import get_logger
from navinib.tools.drive import DriveTimes
from navinib.tools.schedule import hhmm

DAY_START, DAY_END = 8 * 60, 17 * 60 + 30

log = get_logger("navinib.tools.route")


def free_windows(day: dict, now_min: int | None = None, buffer_min: int = config.PREP_BUFFER_MIN,
                 day_start: int = DAY_START, day_end: int = DAY_END) -> list[dict]:
    """Open gaps in a schedule_lookup() result. Cancelled events don't block time."""
    events = [e for e in day["events"] if e["status"] != "cancelled"]
    windows, cursor = [], day_start if now_min is None else max(day_start, now_min)
    last_location = None
    for e in events + [None]:
        if e is None:
            end, before = day_end, None
        else:
            end = e["start_min"] - (buffer_min if e["kind"] == "call" else 0)
            before = e["event_id"]
        if end - cursor >= config.VISIT_MIN:
            windows.append({"start_min": cursor, "end_min": end, "start": hhmm(cursor), "end": hhmm(end),
                            "before_event": before, "last_location_id": last_location})
        if e is not None:
            cursor = max(cursor, e["end_min"])
            if e["kind"] == "call" and e.get("location_id"):
                last_location = e["location_id"]
    log.info("free_windows now=%s buffer=%d -> %s", hhmm(now_min) if now_min is not None else "-", buffer_min,
             ", ".join(f"{w['start']}-{w['end']}" for w in windows) or "none")
    return windows


def plan_route(start: dict, start_min: int, end_min: int, candidates: list[dict],
               drive: DriveTimes | None = None, visit_min: int = config.VISIT_MIN,
               max_visits: int | None = None) -> dict:
    """Pick and order visits from `candidates` (provider_lookup/provider_scoring rows) for one window.

    start: {'id', 'lat', 'lon'}, where the rep is at start_min.
    Higher-scoring candidates are preferred; a candidate can be left out, but a
    visit that is planned always fits its office hours.
    """
    drive = drive or DriveTimes()
    log.info("plan_route from %s, window %s-%s, %d candidates, max_visits=%s", start["id"], hhmm(start_min),
             hhmm(end_min), len(candidates), max_visits)
    pts = [start] + [{"id": c["location_id"], "lat": c["lat"], "lon": c["lon"]} for c in candidates]

    # Rule out what can't work on its own first, so every drop has a plain reason.
    dropped, usable = [], []
    for i, c in enumerate(candidates, 1):
        arrive = start_min + drive.between(start, pts[i])
        finish = max(arrive, c["open_min"]) + visit_min
        if finish > c["close_min"]:
            dropped.append({**_brief(c), "reason": f"closes at {hhmm(c['close_min'])}; earliest finish "
                                                   f"{hhmm(finish)}"})
        elif finish > end_min:
            dropped.append({**_brief(c), "reason": f"earliest finish {hhmm(finish)} is after the window "
                                                   f"ends at {hhmm(end_min)}"})
        else:
            usable.append(i)

    visits = []
    if usable:
        visits, solver_dropped = _solve(pts, candidates, usable, start_min, end_min, drive, visit_min,
                                        max_visits)
        dropped += [{**_brief(candidates[i - 1]), "reason": "fits on its own, but higher-priority visits "
                                                            "used the time"} for i in solver_dropped]
    for v in visits:
        log.info("  planned %s at %s %s-%s (%d min drive)", v["full_name"], v["location_id"], v["start"],
                 v["end"], v["drive_min"])
    for d in dropped:
        log.info("  dropped %s at %s: %s", d["full_name"], d["location_id"], d["reason"])
    if not visits:
        log.warning("plan_route: no feasible visit in %s-%s", hhmm(start_min), hhmm(end_min))
    return {"feasible": bool(visits), "window": {"start": hhmm(start_min), "end": hhmm(end_min)},
            "visits": visits, "dropped": dropped, "drive_source": drive.source}


def _brief(c: dict) -> dict:
    return {"provider_id": c["provider_id"], "location_id": c["location_id"], "full_name": c["full_name"]}


def _solve(pts, candidates, usable, start_min, end_min, drive, visit_min, max_visits):
    nodes = [0] + usable                      # node 0 = start; last node = open end (free to reach)
    n = len(nodes) + 1
    end_node = n - 1
    mgr = pywrapcp.RoutingIndexManager(n, 1, [0], [end_node])
    routing = pywrapcp.RoutingModel(mgr)

    def travel(a, b):
        if b == end_node or a == end_node:
            return 0
        return drive.between(pts[nodes[a]], pts[nodes[b]])

    def transit(i, j):
        a, b = mgr.IndexToNode(i), mgr.IndexToNode(j)
        return travel(a, b) + (visit_min if 0 < a < end_node else 0)

    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 24 * 60, 24 * 60, False, "Time")
    time = routing.GetDimensionOrDie("Time")
    time.CumulVar(routing.Start(0)).SetRange(start_min, start_min)
    time.CumulVar(routing.End(0)).SetMax(end_min)
    for k, i in enumerate(usable, 1):
        c = candidates[i - 1]
        idx = mgr.NodeToIndex(k)
        time.CumulVar(idx).SetRange(max(c["open_min"], start_min), min(c["close_min"], end_min) - visit_min)
        # Leaving a visit out costs its score, so the solver keeps the most valuable ones.
        routing.AddDisjunction([idx], 1000 + int(1000 * c.get("score", 0)))

    if max_visits is not None:
        count = routing.RegisterUnaryTransitCallback(lambda i: 1 if 0 < mgr.IndexToNode(i) < end_node else 0)
        routing.AddDimension(count, 0, max_visits, True, "Visits")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return [], usable

    visits, seen, idx, prev = [], set(), routing.Start(0), 0
    while not routing.IsEnd(idx):
        idx = sol.Value(routing.NextVar(idx))
        node = mgr.IndexToNode(idx)
        if node == end_node:
            break
        c = candidates[nodes[node] - 1]
        begin = sol.Min(time.CumulVar(idx))
        visits.append({**_brief(c), "affiliation": c["affiliation"],
                       "drive_min": travel(prev, node), "start": hhmm(begin), "end": hhmm(begin + visit_min),
                       "start_min": begin, "end_min": begin + visit_min,
                       "closes": hhmm(c["close_min"]), "score": c.get("score")})
        seen.add(nodes[node])
        prev = node
    return visits, [i for i in usable if i not in seen]


if __name__ == "__main__":
    # The Checkpoint 2.1 replan: Dr. Smith's 2:00 call cancels and the rep is at her office.
    from navinib.db import connect
    from navinib.tools.providers import provider_lookup, provider_scoring
    from navinib.tools.schedule import cancel_event, schedule_lookup

    from navinib.log import setup_logging

    setup_logging()
    con = connect()
    cancel_event(con, "ev-04")
    window = free_windows(schedule_lookup(con, "rep-01", config.DEMO_DATE), now_min=14 * 60)[0]
    loc = con.execute("SELECT * FROM location WHERE location_id = 'L-004'").fetchone()
    start = {"id": "L-004", "lat": loc["lat"], "lon": loc["lon"]}
    near = provider_lookup(con, "rep-01", config.DEMO_DATE, near=(loc["lat"], loc["lon"]),
                           max_drive_min=20, exclude={"p-003"})
    top3 = provider_scoring(con, near, config.DEMO_DATE)[:3]
    plan = plan_route(start, window["start_min"], window["end_min"], top3, max_visits=1)
    print(f"Open window {plan['window']['start']} to {plan['window']['end']} "
          f"(drive times: {plan['drive_source']})")
    for v in plan["visits"]:
        print(f"  Visit {v['full_name']} at {v['affiliation']}: {v['start']} to {v['end']} "
              f"({v['drive_min']} min drive, closes {v['closes']})")
    for d in plan["dropped"]:
        print(f"  Not planned: {d['full_name']} ({d['reason']})")
    con.close()
