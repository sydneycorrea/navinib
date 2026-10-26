"""Pre-call brief inputs (Action 4 in Checkpoint 2.1).

Gathers everything the rep should see before a visit: structured facts from
SQLite and retrieved text from Chroma, each with its source id. The LLM that
writes the final one-page brief (phase 2) may only use what is collected here.
"""
import sqlite3

from navinib.rag import retrieval
from navinib.tools.providers import alert_display, location_contacts, provider_lookup

def build_brief(con: sqlite3.Connection, col, provider_id: str, day: str, rep_id: str = "rep-01",
                topic: str | None = None) -> dict:
    provider = con.execute("SELECT * FROM provider WHERE provider_id = ?", (provider_id,)).fetchone()
    if provider is None:
        raise ValueError(f"unknown provider {provider_id!r}")
    locations = [p for p in provider_lookup(con, rep_id, day) if p["provider_id"] == provider_id]
    calls = con.execute("SELECT date, call_type, outcome FROM visit WHERE provider_id = ? AND date < ?"
                        " ORDER BY date DESC LIMIT 3", (provider_id, day)).fetchall()
    alerts = con.execute("SELECT open_count FROM lab_alert WHERE provider_id = ?", (provider_id,)).fetchone()

    # With a topic, search the notes for it; without one, show the latest notes.
    notes = (retrieval.call_note_recall(col, provider_id, topic) if topic
             else retrieval.recent_call_notes(col, provider_id))
    # Talking points answer what the provider raised last time, so search the label with the notes.
    label_query = topic or " ".join(n["text"] for n in notes) or "Product X efficacy and safety"
    talking_points = retrieval.label_lookup(col, label_query)

    pubs = con.execute(
        "SELECT pb.* FROM publication pb JOIN publication_author pa USING (pub_id)"
        " WHERE pa.provider_id = ? ORDER BY pb.year DESC, pb.pub_id", (provider_id,)).fetchall()
    publications = [{**dict(p), "may_discuss": p["use_status"] != "investigational_use"} for p in pubs]
    trials = retrieval.trial_search(col, label_query)

    do_not_raise = [{"ref": t["id"], "why": "investigational trial"}
                    for t in trials if t["meta"]["status"] != "approved"]
    do_not_raise += [{"ref": p["pub_id"], "why": "their publication covers an unapproved use"}
                     for p in publications if not p["may_discuss"]]

    return {
        "provider_id": provider_id,
        "full_name": provider["full_name"],
        "tier": provider["tier"],
        "locations_today": [{"location_id": l["location_id"], "affiliation": l["affiliation"],
                             "open_min": l["open_min"], "close_min": l["close_min"]} for l in locations],
        "recent_calls": [dict(c) for c in calls],
        "lab_alerts": alert_display(alerts["open_count"] if alerts else 0),
        "call_notes": [{"ref": n["id"], "date": n["meta"]["date"], "text": n["text"]} for n in notes],
        "publications": publications,
        "talking_points": [{"ref": t["meta"]["ref"], "text": t["text"]} for t in talking_points],
        "do_not_raise": do_not_raise,
        "contacts": [c for l in locations for c in location_contacts(con, l["location_id"])],
    }


def render_brief(b: dict) -> str:
    """Plain-text view of the brief inputs, for the demo and for review."""
    hhmm = lambda m: f"{m // 60}:{m % 60:02d}"
    lines = [f"Pre-call brief: {b['full_name']} ({b['provider_id']}, tier {b['tier']})"]
    for l in b["locations_today"]:
        lines.append(f"  Today at {l['affiliation']} ({l['location_id']}), {hhmm(l['open_min'])} to "
                     f"{hhmm(l['close_min'])}")
    if not b["locations_today"]:
        lines.append("  Not at any location today")
    lines.append(f"  Open lab alerts: {b['lab_alerts']}")
    lines.append("  Recent calls: " + ("; ".join(f"{c['date']} {c['call_type'].replace('_', ' ')} "
                                                 f"({c['outcome']})" for c in b["recent_calls"]) or "none"))
    lines.append("  Last call notes:")
    lines += [f"    [{n['ref']}] {n['text'][:140]}..." for n in b["call_notes"]] or ["    none found"]
    lines.append("  Their publications:")
    lines += [f"    [{p['pub_id']}] {p['title']} ({p['journal']}, {p['year']})"
              + ("" if p["may_discuss"] else "  -> background only, do not discuss")
              for p in b["publications"]] or ["    none found"]
    lines.append("  Talking points (approved label only):")
    lines += [f"    [{t['ref']}] {t['text'][:140]}..." for t in b["talking_points"]] or ["    not found"]
    if b["do_not_raise"]:
        lines.append("  Do not raise (refer questions to Medical Information):")
        lines += [f"    [{d['ref']}] {d['why']}" for d in b["do_not_raise"]]
    if b["contacts"]:
        lines.append("  Others you may meet:")
        lines += [f"    {c['full_name']}, {c['role'].replace('_', ' ')}: {c['notes']}" for c in b["contacts"]]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from navinib import config
    from navinib.db import connect

    pid = sys.argv[1] if len(sys.argv) > 1 else "p-001"
    print(render_brief(build_brief(connect(), retrieval.build_index(), pid, config.DEMO_DATE)))
