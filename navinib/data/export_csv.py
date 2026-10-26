"""Export the synthetic world to CSV files for review in a spreadsheet.

Run:  python -m navinib.data.export_csv [out_dir]     (default: data/csv)
"""
import csv
import sys
from pathlib import Path

from navinib import config
from navinib.db import connect
from navinib.rag.corpus import CALL_NOTES

HHMM = "printf('%02d:%02d', {0} / 60, {0} % 60)"

QUERIES = {
    "providers.csv": f"""
        SELECT p.provider_id, p.full_name, p.kind, p.specialty, p.tier, l.location_id, l.affiliation,
               l.address, group_concat(substr('MonTueWedThuFriSatSun', h.weekday * 3 + 1, 3) || ' '
                   || {HHMM.format('h.open_min')} || '-' || {HHMM.format('h.close_min')}, '; ') AS hours,
               r.product_x, r.competitor_a, r.competitor_b, a.open_count AS lab_alerts
        FROM provider p JOIN location l USING (provider_id) JOIN location_hours h USING (location_id)
        JOIN prescribing r USING (provider_id) JOIN lab_alert a USING (provider_id)
        GROUP BY l.location_id ORDER BY p.provider_id, l.location_id""",
    "contacts.csv": """
        SELECT c.contact_id, c.full_name, c.role, c.location_id, l.affiliation, c.notes
        FROM contact c JOIN location l USING (location_id) ORDER BY c.contact_id""",
    "publications.csv": """
        SELECT pb.pub_id, pb.year, pb.title, pb.journal, pb.use_status,
               group_concat(p.full_name, '; ') AS authors
        FROM publication pb JOIN publication_author pa USING (pub_id) JOIN provider p USING (provider_id)
        GROUP BY pb.pub_id ORDER BY pb.year DESC, pb.pub_id""",
    "call_history.csv": """
        SELECT v.visit_id, v.date, v.call_type, v.outcome, v.provider_id, p.full_name, p.tier
        FROM visit v JOIN provider p USING (provider_id) ORDER BY v.date DESC, v.visit_id""",
    "demo_day_schedule.csv": f"""
        SELECT e.event_id, e.date, {HHMM.format('e.start_min')} AS start, {HHMM.format('e.end_min')} AS "end",
               e.kind, e.title, e.provider_id, e.location_id, l.affiliation, e.status
        FROM schedule_event e LEFT JOIN location l USING (location_id) ORDER BY e.date, e.start_min""",
}


def export(out_dir: Path | str = config.ROOT / "data" / "csv", db_path: Path | str = config.DB_PATH) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    for name, sql in QUERIES.items():
        cur = con.execute(sql)
        with open(out / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([d[0] for d in cur.description])
            w.writerows(cur.fetchall())
    with open(out / "call_notes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["note_id", "provider_id", "provider", "date", "note"])
        w.writerows(CALL_NOTES)
    con.close()
    return out


if __name__ == "__main__":
    print(f"Wrote CSVs to {export(*sys.argv[1:2])}")
