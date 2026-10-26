"""Schedule lookup (Action 1 in Checkpoint 2.1): today's calls, fixed tasks and where the rep is."""
import sqlite3

from navinib.log import get_logger

log = get_logger("navinib.tools.schedule")


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def schedule_lookup(con: sqlite3.Connection, rep_id: str, day: str) -> dict:
    rep = con.execute("SELECT * FROM rep WHERE rep_id = ?", (rep_id,)).fetchone()
    if rep is None:
        raise ValueError(f"unknown rep {rep_id!r}")
    rows = con.execute(
        "SELECT * FROM schedule_event WHERE rep_id = ? AND date = ? ORDER BY start_min",
        (rep_id, day)).fetchall()
    events = [{
        "event_id": r["event_id"], "kind": r["kind"], "title": r["title"],
        "provider_id": r["provider_id"], "location_id": r["location_id"], "status": r["status"],
        "start_min": r["start_min"], "end_min": r["end_min"],
        "start": hhmm(r["start_min"]), "end": hhmm(r["end_min"]),
    } for r in rows]
    cancelled = sum(e["status"] == "cancelled" for e in events)
    log.info("schedule_lookup rep=%s day=%s -> %d events (%d cancelled)", rep_id, day, len(events), cancelled)
    for e in events:
        log.debug("  %s %s-%s %s [%s]", e["event_id"], e["start"], e["end"], e["title"], e["status"])
    # Until live location exists, the rep starts the day at home.
    return {"rep_id": rep_id, "date": day, "events": events,
            "location": {"lat": rep["home_lat"], "lon": rep["home_lon"]}}


def cancel_event(con: sqlite3.Connection, event_id: str) -> None:
    """Mark a call cancelled (the trigger for a replan)."""
    cur = con.execute("UPDATE schedule_event SET status = 'cancelled' WHERE event_id = ?", (event_id,))
    if cur.rowcount == 0:
        raise ValueError(f"unknown event {event_id!r}")
    con.commit()
    log.info("cancel_event %s -> cancelled", event_id)
