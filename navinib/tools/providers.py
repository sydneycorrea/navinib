"""Provider lookup and provider scoring (Action 2 in Checkpoint 2.1).

Lookup answers exact questions from SQLite: who is in the territory, how far,
open when. Scoring is a fixed formula, so every rank can be explained.
"""
import math
import sqlite3
from datetime import date

from navinib import config


def drive_minutes(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Placeholder estimate (haversine x road factor) until the OSRM matrix is wired in."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    km = 2 * r * math.asin(math.sqrt(a)) * config.ROAD_FACTOR
    return max(1, round(km / config.AVG_SPEED_KMH * 60))


def alert_display(count: int) -> str:
    """Provider-level count, suppressed below the threshold so it never identifies a patient."""
    if count == 0:
        return "0"
    return f"<{config.ALERT_SUPPRESS_BELOW}" if count < config.ALERT_SUPPRESS_BELOW else str(count)


def provider_lookup(con: sqlite3.Connection, rep_id: str, day: str,
                    near: tuple[float, float] | None = None, max_drive_min: int | None = None,
                    open_from: int | None = None, open_until: int | None = None,
                    exclude: set[str] | frozenset = frozenset()) -> list[dict]:
    """Provider locations open on `day`, optionally within a drive time and open across a window.

    Returns one row per location, so a provider who practices at two places that day
    appears twice, each with that location's hours.
    """
    weekday = date.fromisoformat(day).weekday()
    rows = con.execute(
        "SELECT p.*, l.location_id, l.affiliation, l.address, l.lat, l.lon, h.open_min, h.close_min"
        " FROM provider p JOIN location l USING (provider_id)"
        " JOIN location_hours h ON h.location_id = l.location_id AND h.weekday = ?"
        " WHERE p.rep_id = ? ORDER BY l.location_id", (weekday, rep_id)).fetchall()
    out = []
    for r in rows:
        if r["provider_id"] in exclude:
            continue
        if open_from is not None and r["open_min"] > open_from:
            continue
        if open_until is not None and r["close_min"] < open_until:
            continue
        p = dict(r)
        if near is not None:
            p["drive_min"] = drive_minutes(near[0], near[1], r["lat"], r["lon"])
            if max_drive_min is not None and p["drive_min"] > max_drive_min:
                continue
        out.append(p)
    return out


def provider_scoring(con: sqlite3.Connection, providers: list[dict], today: str) -> list[dict]:
    """Score and rank providers; each result carries its breakdown and plain-language reasons.

    Goals and recency count completed face-to-face calls only; phone calls are logged
    in call history but don't replace an in-person visit.
    """
    if not providers:
        return []
    t = date.fromisoformat(today)
    month_start = t.replace(day=1).isoformat()
    vol = {r["provider_id"]: r["product_x"] + r["competitor_a"] + r["competitor_b"]
           for r in con.execute("SELECT * FROM prescribing")}
    max_vol = max(vol.values()) or 1
    alerts = {r["provider_id"]: r["open_count"] for r in con.execute("SELECT * FROM lab_alert")}
    w = config.SCORE_WEIGHTS

    ranked = []
    for p in providers:
        pid = p["provider_id"]
        last = con.execute("SELECT MAX(date) FROM visit WHERE provider_id = ? AND outcome = 'completed'"
                           " AND call_type = 'face_to_face'"
                           " AND date < ?", (pid, today)).fetchone()[0]
        done = con.execute("SELECT COUNT(*) FROM visit WHERE provider_id = ? AND outcome = 'completed'"
                           " AND call_type = 'face_to_face'"
                           " AND date >= ? AND date < ?", (pid, month_start, today)).fetchone()[0]
        goal = config.CALL_GOALS[p["tier"]]
        days = (t - date.fromisoformat(last)).days if last else None

        parts = {
            "product_use": vol.get(pid, 0) / max_vol,
            "goal_gap": max(goal - done, 0) / goal,
            "lab_alerts": min(alerts.get(pid, 0) / 10, 1.0),
            "recency": 1.0 if days is None else min(days / config.RECENCY_CAP_DAYS, 1.0),
        }
        breakdown = {k: round(w[k] * v, 3) for k, v in parts.items()}
        reasons = [
            f"oncology volume {vol.get(pid, 0)} claims/quarter",
            f"{done} of {goal} tier-{p['tier']} calls done this month",
            f"open lab alerts {alert_display(alerts.get(pid, 0))}",
            "never visited" if days is None else f"last in-person visit {days} days ago",
        ]
        ranked.append({**p, "score": round(sum(breakdown.values()), 3),
                       "breakdown": breakdown, "reasons": reasons})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def location_contacts(con: sqlite3.Connection, location_id: str) -> list[dict]:
    """Non-prescribing staff at a location, for the pre-call brief. Never scored or routed."""
    rows = con.execute("SELECT * FROM contact WHERE location_id = ? ORDER BY contact_id", (location_id,))
    return [dict(r) for r in rows]
