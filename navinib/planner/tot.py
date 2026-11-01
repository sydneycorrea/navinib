"""Tree-of-Thought replan (Checkpoint 4.1).

When a call cancels, there is more than one reasonable way to use the freed
time. Instead of taking the first feasible fill, the replan grows a small tree
of alternatives and compares them:

  thought    one next step for the open window: visit a nearby provider in
             person, or phone a provider no visit can reach today (phone
             counts half toward goals but needs no drive)
  generate   from each partial plan, every next step that could follow it
  check      hard rules prune a branch outright: office hours, the window end,
             drive time, no provider twice
  evaluate   a fixed value function scores what a plan has achieved; a
             partial plan is ranked by its promise: that value plus what the
             time it has left could still earn at the best visit's rate
  search     beam search, up to TOT_MAX_STEPS levels: at each level keep the
             TOT_BEAM_WIDTH most promising plans of each shape (1 visit,
             1 phone + 1 visit, ...), so one strategy can't crowd out the others
  choose     the highest-value complete plan, the best plan of each other
             shape (two visits, phone + visit, ...) as alternatives, and a
             stated reason

The evaluator is plain code, not an LLM, so every choice can be explained and
tested. In phase 2 the LLM planner calls this as a tool and writes the
explanation for the rep.

Run the 2.1 scenario:  python -m navinib.planner.tot
"""
import sqlite3
from dataclasses import dataclass, field

from navinib import config
from navinib.log import get_logger
from navinib.tools.drive import DriveTimes
from navinib.tools.providers import provider_lookup, provider_scoring
from navinib.tools.route import free_windows
from navinib.tools.schedule import hhmm, schedule_lookup

log = get_logger("navinib.planner.tot")

VISIT_RADIUS_MIN = 20   # in-person candidates must be this close to where the rep is
N_VISIT_CANDIDATES = 6
N_PHONE_CANDIDATES = 4


@dataclass(frozen=True)
class Step:
    """One call in a plan: who, how (face_to_face or phone), and when."""
    provider_id: str
    full_name: str
    call_type: str
    location_id: str
    start_min: int
    end_min: int
    drive_min: int
    score: float

    def label(self) -> str:
        """Short text for logs and printouts, e.g. 'visit Dr. Alicia Lee 14:10-14:40'."""
        how = "visit" if self.call_type == "face_to_face" else "phone"
        return f"{how} {self.full_name} {hhmm(self.start_min)}-{hhmm(self.end_min)}"


@dataclass(frozen=True)
class Plan:
    """A partial or complete plan for the window: the steps so far and where the rep ends up."""
    steps: tuple[Step, ...] = ()
    at: dict = field(default_factory=dict, hash=False, compare=False)
    now_min: int = 0

    def value(self) -> float:
        """The evaluator: goal credit weighted by priority, minus a small cost per minute driven."""
        credit = sum(s.score * config.CALL_CREDIT[s.call_type] for s in self.steps)
        drive = sum(s.drive_min for s in self.steps)
        return round(credit - config.TOT_DRIVE_COST * drive, 3)

    def promise(self, end_min: int, rate: float) -> float:
        """Value so far plus the most the remaining minutes could earn at `rate` value per minute."""
        return round(self.value() + rate * max(end_min - self.now_min, 0), 3)

    def shape(self) -> str:
        """The kind of plan, e.g. '2 visits' or '1 phone + 1 visit', used to pick varied alternatives."""
        v = sum(s.call_type == "face_to_face" for s in self.steps)
        ph = len(self.steps) - v
        parts = ([f"{ph} phone"] if ph else []) + ([f"{v} visit{'s' if v > 1 else ''}"] if v else [])
        return " + ".join(parts) or "no calls"

    def providers(self) -> set[str]:
        """Providers already in this plan, so no one is called twice."""
        return {s.provider_id for s in self.steps}

    def label(self) -> str:
        """The plan's steps in order, or 'keep the time free' when empty."""
        return "; ".join(s.label() for s in self.steps) or "keep the time free (prep, notes)"


def candidates(con: sqlite3.Connection, rep_id: str, day: str, start: dict, booked: set[str],
               window: dict, drive: DriveTimes) -> dict:
    """Who each kind of thought may use.

    Visits: the top-scored providers within VISIT_RADIUS_MIN of the rep. Phone
    calls are a fallback, not a substitute: they are offered only for top-scored
    providers in the territory that no in-person visit in this window can reach
    (too far, or closing too soon). `booked` holds providers already on today's
    schedule, including the one who cancelled.
    """
    near = provider_lookup(con, rep_id, day, near=(start["lat"], start["lon"]),
                           max_drive_min=VISIT_RADIUS_MIN, exclude=booked)
    visits = provider_scoring(con, near, day)[:N_VISIT_CANDIDATES]
    root = Plan((), start, window["start_min"])
    visitable = {c["provider_id"] for c in visits
                 if _fit(root, c, "face_to_face", window["end_min"], drive)[0] is not None}
    anywhere = provider_lookup(con, rep_id, day, exclude=booked | visitable)
    phone = _one_row_per_provider(provider_scoring(con, anywhere, day))[:N_PHONE_CANDIDATES]
    return {"face_to_face": visits, "phone": phone}


def _one_row_per_provider(ranked: list[dict]) -> list[dict]:
    """A phone call reaches the provider, not a location, so keep each provider's first row."""
    seen, out = set(), []
    for c in ranked:
        if c["provider_id"] not in seen:
            seen.add(c["provider_id"])
            out.append(c)
    return out


def _fit(plan: Plan, c: dict, call_type: str, end_min: int, drive: DriveTimes) -> tuple[Plan | None, str]:
    """Add one call to `plan` if the hard rules allow it; otherwise return None and the reason."""
    if call_type == "face_to_face":
        point = {"id": c["location_id"], "lat": c["lat"], "lon": c["lon"]}
        drive_min, length = drive.between(plan.at, point), config.VISIT_MIN
    else:
        point, drive_min, length = plan.at, 0, config.PHONE_MIN
    start = max(plan.now_min + drive_min, c["open_min"])
    end = start + length
    if end > c["close_min"]:
        return None, f"closes {hhmm(c['close_min'])}, earliest finish {hhmm(end)}"
    if end > end_min:
        return None, f"earliest finish {hhmm(end)} is past {hhmm(end_min)}"
    step = Step(c["provider_id"], c["full_name"], call_type, c["location_id"], start, end, drive_min, c["score"])
    return Plan(plan.steps + (step,), point, end), ""


def next_steps(plan: Plan, pool: dict, end_min: int, drive: DriveTimes) -> tuple[list[Plan], dict]:
    """Generate every plan that adds one call to `plan`, and say why the rest were pruned.

    Pruned steps come back as {'phone Hillcrest Cancer Center': 'closes 14:30, ...'}.
    At most TOT_MAX_PHONE phone calls go in one window.
    """
    children, pruned = [], {}
    phones = sum(s.call_type == "phone" for s in plan.steps)
    for call_type, cands in pool.items():
        if call_type == "phone" and phones >= config.TOT_MAX_PHONE:
            continue
        for c in cands:
            if c["provider_id"] in plan.providers():
                continue
            child, why = _fit(plan, c, call_type, end_min, drive)
            if child is None:
                how = "visit" if call_type == "face_to_face" else "phone"
                pruned[f"{how} {c['full_name']}"] = why
            else:
                children.append(child)
    return children, pruned


def tree_of_thought(con: sqlite3.Connection, rep_id: str, day: str, cancelled_event_id: str,
                    now_min: int, drive: DriveTimes | None = None) -> dict:
    """Replan the window freed by a cancelled call by comparing alternative plans.

    Returns the chosen plan, the runners-up, the reason for the choice, and a
    per-level trace of what was kept and pruned (for review and the report).
    """
    drive = drive or DriveTimes()
    sched = schedule_lookup(con, rep_id, day)
    event = next(e for e in sched["events"] if e["event_id"] == cancelled_event_id)
    window = next((w for w in free_windows(sched, now_min=now_min) if w["start_min"] <= now_min < w["end_min"]),
                  None)
    if window is None:
        log.warning("tot: no open window at %s after %s cancelled", hhmm(now_min), cancelled_event_id)
        return {"window": None, "start_location_id": event["location_id"], "chosen": _out(Plan()),
                "alternatives": [], "reason": f"no open window of {config.VISIT_MIN}+ minutes at {hhmm(now_min)}",
                "trace": [], "drive_source": drive.source}
    loc = con.execute("SELECT * FROM location WHERE location_id = ?", (event["location_id"],)).fetchone()
    start = {"id": loc["location_id"], "lat": loc["lat"], "lon": loc["lon"]}   # the rep is at the cancelled call
    booked = {e["provider_id"] for e in sched["events"] if e["provider_id"]}
    pool = candidates(con, rep_id, day, start, booked, window, drive)
    log.info("tot: %s cancelled, window %s-%s from %s; %d visit and %d phone candidates", cancelled_event_id,
             window["start"], window["end"], start["id"], len(pool["face_to_face"]), len(pool["phone"]))

    # Remaining time is valued at the best rate a single visit achieves from here, drive included.
    end_min = window["end_min"]
    root = Plan((), start, window["start_min"])
    firsts = [p for c in pool["face_to_face"] if (p := _fit(root, c, "face_to_face", end_min, drive)[0])]
    rate = max((p.value() / (p.now_min - root.now_min) for p in firsts), default=0.0)

    beam, complete, trace = [root], [root], []
    for level in range(1, config.TOT_MAX_STEPS + 1):
        children, pruned = [], {}
        for plan in beam:
            kids, why = next_steps(plan, pool, end_min, drive)
            children += kids
            for step, reason in why.items():   # keep the first (earliest) reason per step
                pruned.setdefault(step, reason)
        children = _dedupe(children)
        children.sort(key=lambda p: p.promise(end_min, rate), reverse=True)
        beam, per_shape = [], {}
        for p in children:
            if per_shape.get(p.shape(), 0) < config.TOT_BEAM_WIDTH:
                per_shape[p.shape()] = per_shape.get(p.shape(), 0) + 1
                beam.append(p)
        complete += children
        trace.append({"level": level, "generated": len(children), "pruned": pruned,
                      "kept": [(p.label(), p.value(), p.promise(end_min, rate)) for p in beam],
                      "dropped_by_beam": [(p.label(), p.value(), p.promise(end_min, rate))
                                          for p in children if p not in beam]})
        log.info("tot level %d: %d plans, %d pruned by rules, kept: %s", level, len(children), len(pruned),
                 " | ".join(f"{p.value():.3f}/{p.promise(end_min, rate):.3f} {p.label()}" for p in beam)
                 or "none")
        if not beam:
            break

    ranked = sorted(_dedupe(complete), key=lambda p: p.value(), reverse=True)
    best = ranked[0]
    alternatives, shapes = [], {best.shape()}
    for p in ranked[1:]:                   # best plan of each other shape, for a real comparison
        if p.shape() not in shapes and len(alternatives) < 2:
            shapes.add(p.shape())
            alternatives.append(p)
    reason = _reason(best, alternatives)
    log.info("tot chose %.3f %s (%s)", best.value(), best.label(), reason)
    return {"window": {"start": window["start"], "end": window["end"]}, "start_location_id": start["id"],
            "chosen": _out(best), "alternatives": [_out(p) for p in alternatives], "reason": reason,
            "trace": trace, "drive_source": drive.source}


def _dedupe(plans: list[Plan]) -> list[Plan]:
    """Plans with the same calls in a different order are one option; keep the best-valued order."""
    best = {}
    for p in plans:
        key = frozenset((s.provider_id, s.call_type) for s in p.steps)
        if key not in best or p.value() > best[key].value():
            best[key] = p
    return list(best.values())


def _reason(best: Plan, alternatives: list[Plan]) -> str:
    """Why the chosen plan beat the best alternative, in one sentence."""
    if not best.steps:
        return "no call fits the window, so keep it free for prep and notes"
    if not alternatives:
        return f"the only plan that fits (value {best.value():.2f})"
    nxt = alternatives[0]
    return (f"{best.shape()} earns {best.value():.2f}, more than the best {nxt.shape()} plan "
            f"({nxt.value():.2f}: {nxt.label()})")


def _out(p: Plan) -> dict:
    """A plan as plain data for the caller (and later the LLM planner)."""
    return {"value": p.value(), "summary": p.label(), "shape": p.shape(),
            "steps": [{"provider_id": s.provider_id, "full_name": s.full_name, "call_type": s.call_type,
                       "location_id": s.location_id, "start": hhmm(s.start_min), "end": hhmm(s.end_min),
                       "drive_min": s.drive_min, "score": s.score} for s in p.steps]}


if __name__ == "__main__":
    # The Checkpoint 2.1 scenario: Dr. Smith's 2:00 call cancels and the rep is at her office.
    from navinib.db import connect
    from navinib.log import setup_logging
    from navinib.tools.schedule import cancel_event

    setup_logging()
    con = connect()
    cancel_event(con, "ev-04")
    r = tree_of_thought(con, "rep-01", config.DEMO_DATE, "ev-04", now_min=14 * 60)
    print(f"Open window {r['window']['start']} to {r['window']['end']} from {r['start_location_id']} "
          f"(drive times: {r['drive_source']})\n")
    for t in r["trace"]:
        print(f"Level {t['level']}: {t['generated']} plans generated, {len(t['kept'])} kept")
        if t["kept"]:
            print("   value  promise  plan")
        for label, value, promise in t["kept"]:
            print(f"   {value:5.2f}  {promise:6.2f}   {label}")
        for label, value, promise in t["dropped_by_beam"][:5]:
            print(f"   {value:5.2f}  {promise:6.2f}   (not kept) {label}")
        if len(t["dropped_by_beam"]) > 5:
            print(f"   ... and {len(t['dropped_by_beam']) - 5} more not kept")
        for step, why in t["pruned"].items():
            print(f"   ruled out: {step} ({why})")
    print(f"\nChosen, {r['chosen']['shape']} ({r['chosen']['value']:.2f}): {r['chosen']['summary']}")
    for a in r["alternatives"]:
        print(f"  Alternative, {a['shape']} ({a['value']:.2f}): {a['summary']}")
    print(f"Why: {r['reason']}")
    con.close()
