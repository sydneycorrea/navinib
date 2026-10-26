"""Build the synthetic Navinib world into SQLite.

Run:  python -m navinib.data.generate

Everything is invented and seeded, so every run produces the same world.
Provider names are fictional. Locations are random points in the Pittsburgh
area for now; public CMS practice addresses replace them in a later step.
The providers named in the call notes (rag/corpus.py) are fixed so the
retrieval layer and the structured data describe the same people.
"""
import random
from datetime import date, timedelta
from pathlib import Path

from navinib import config
from navinib.db import connect, create_schema
from navinib.log import get_logger
from navinib.rag.corpus import PUBLICATIONS

log = get_logger("navinib.data.generate")

LAT_RANGE, LON_RANGE = (40.38, 40.52), (-80.10, -79.88)

FIRST = ["Maya", "Ethan", "Grace", "Omar", "Lucia", "Victor", "Hannah", "Rahul", "Nora", "Felix",
         "Irene", "Marcus", "Sofia", "Kenji", "Leah", "Andre", "Clara", "Tomas", "Wen", "Zara"]
LAST = ["Abernathy", "Brennan", "Castellano", "Dhillon", "Everly", "Fairbanks", "Galloway",
        "Hartwell", "Ishikawa", "Jorgensen", "Kowalczyk", "Lindqvist", "Marchetti", "Nakamura",
        "Ostrowski", "Pemberton", "Quintero", "Rasmussen", "Sorensen", "Thackeray"]
STREETS = ["Maple Ave", "Ridge Rd", "Liberty St", "Summit Dr", "Penn Ave", "Cedar Ln",
           "Grove St", "Hilltop Rd", "Bridge St", "Orchard Way"]
PRACTICE_WORDS = ["Brightwater", "Oak Hollow", "Stonebridge", "Clearview", "Fox Run",
                  "Willow Creek", "Highpoint", "Harborview", "Meadowbrook", "Silver Lake"]
KINDS = [  # (kind, specialty, affiliation suffix, weight)
    ("oncologist", "Medical Oncology", "Oncology Associates", 5),
    ("oncologist", "Hematology/Oncology", "Hematology Oncology", 4),
    ("community_practice", "Medical Oncology", "Community Cancer Care", 2),
    ("infusion_center", "Infusion Services", "Infusion Center", 1),
    ("cancer_center", "Comprehensive Oncology", "Cancer Center", 1),
]

WEEKDAYS = range(5)  # Mon-Fri


def every_weekday(open_min: int, close_min: int) -> dict[int, tuple[int, int]]:
    return {d: (open_min, close_min) for d in WEEKDAYS}


# Fixed providers from the 2.1 and 3.1 stories. Ids are sequence numbers; they match
# CALL_NOTES in rag/corpus.py, and generated providers continue the sequence.
# Each provider has one or more locations: (affiliation, lat, lon, {weekday: (open, close)}).
NAMED = [
    # provider_id, full_name, kind, specialty, tier, (product_x, competitor_a, competitor_b), alerts, locations
    ("p-001", "Dr. Alicia Lee", "oncologist", "Hematology/Oncology", "A", (110, 60, 40), 7,
     [("Riverside Oncology Associates", 40.462, -79.975, every_weekday(8 * 60, 17 * 60))]),
    # Two practice locations with different days and hours. On a Tuesday he is only at
    # the Southside clinic, and only until 1:00, so he can't fill a 2pm gap.
    ("p-002", "Dr. Samuel Leigh", "oncologist", "Medical Oncology", "B", (12, 30, 18), 0,
     [("Northgate Cancer Center", 40.505, -80.020,
       {d: (8 * 60, 17 * 60) for d in (0, 2, 4)}),             # Mon, Wed, Fri
      ("Southside Hematology Clinic", 40.425, -79.965,
       {d: (9 * 60, 13 * 60) for d in (1, 3)})]),              # Tue, Thu mornings
    ("p-003", "Dr. Helen Smith", "oncologist", "Medical Oncology", "A", (41, 35, 20), 6,
     [("Eastside Oncology Group", 40.455, -79.930, every_weekday(9 * 60, 17 * 60))]),
    ("p-004", "Dr. Daniel Okafor", "oncologist", "Hematology/Oncology", "B", (0, 25, 15), 0,
     [("Okafor Hematology Clinic", 40.440, -80.005, every_weekday(7 * 60 + 30, 16 * 60))]),
    # The 2.1 example: a high-priority center with an open alert that closes at 2:30.
    ("p-005", "Hillcrest Cancer Center", "cancer_center", "Comprehensive Oncology", "A", (120, 70, 60), 9,
     [("Hillcrest Cancer Center", 40.448, -79.950, every_weekday(8 * 60, 14 * 60 + 30))]),
    ("p-006", "Dr. Priya Raman", "oncologist", "Hematology/Oncology", "B", (22, 20, 10), 0,
     [("Brightwater Hematology Group", 40.470, -79.900, every_weekday(9 * 60, 17 * 60 + 30))]),
]

# Staff who don't prescribe and are never call targets, tied to a location.
# Kevin Tran runs the lab at Dr. Lee's practice (L-001) and supplies the test data
# behind her lab alerts. Maria Gomez is the nurse navigator from call note N-0412.
CONTACTS = [
    # contact_id, location_id, full_name, role, notes
    ("c-001", "L-001", "Kevin Tran", "lab_technician",
     "Runs CBC and liver panels in the practice lab; sends results that feed lab alerts."),
    ("c-002", "L-001", "Maria Gomez", "nurse_navigator",
     "Coordinates patient labs; prefers rep visits after 2pm."),
]
LAB_ALERT_SOURCES = {"p-001": "c-001"}

# Phone calls for the story providers. Each counts as half a call toward the monthly
# goal but doesn't reset visit recency.
NAMED_PHONE_CALLS = {"p-004": ["2026-10-16"], "p-001": ["2026-10-14"]}

# Last visits for the named providers, taken from the call notes.
NAMED_VISITS = {"p-001": ["2026-08-12", "2026-09-23"], "p-002": ["2026-09-30"],
                "p-003": ["2026-10-02"], "p-004": ["2026-10-07"]}

REP = ("rep-01", "Jordan Rivera", "Pittsburgh North", "mgr-01", 40.452, -79.985)
MANAGER = ("mgr-01", "Morgan Ellison", "Mid-Atlantic")


def _demo_schedule(rep_id: str) -> list[tuple]:
    """The scripted demo day: the 2pm call with Dr. Smith is the one that cancels."""
    d = config.DEMO_DATE
    return [
        ("ev-01", rep_id, d, 8 * 60, 8 * 60 + 30, "task", None, None, "Admin and expense report"),
        ("ev-02", rep_id, d, 8 * 60 + 45, 9 * 60 + 30, "call", "p-004", "L-005", "Call: Dr. Daniel Okafor"),
        ("ev-03", rep_id, d, 12 * 60, 13 * 60, "task", None, None, "Lunch"),
        ("ev-04", rep_id, d, 14 * 60, 15 * 60 + 30, "call", "p-003", "L-004", "Call: Dr. Helen Smith"),
        ("ev-05", rep_id, d, 16 * 60 + 30, 17 * 60, "call", "p-006", "L-007", "Call: Dr. Priya Raman"),
    ]


def generate(path: Path | str = config.DB_PATH, seed: int = config.SEED) -> Path:
    rng = random.Random(seed)
    path = Path(path)
    if path.exists():
        path.unlink()
    con = connect(path)
    create_schema(con)

    con.execute("INSERT INTO manager VALUES (?, ?, ?)", MANAGER)
    con.execute("INSERT INTO rep VALUES (?, ?, ?, ?, ?, ?)", REP)
    rep_id = REP[0]

    providers, locations, hours, prescribing, alerts = [], [], [], [], []

    def add_location(pid, affiliation, lat, lon, week):
        lid = f"L-{len(locations) + 1:03d}"
        addr = f"{rng.randint(100, 2999)} {rng.choice(STREETS)}, Pittsburgh, PA"
        locations.append((lid, pid, affiliation, addr, lat, lon))
        hours.extend((lid, d, o, c) for d, (o, c) in sorted(week.items()))

    for pid, full_name, kind, spec, tier, volume, al, locs in NAMED:
        providers.append((pid, full_name, kind, spec, tier, rep_id))
        for affiliation, lat, lon, week in locs:
            add_location(pid, affiliation, lat, lon, week)
        prescribing.append((pid, *volume))
        alerts.append((pid, al))

    used_names = {p[1] for p in providers}
    weights = [k[3] for k in KINDS]
    while len(providers) < config.N_PROVIDERS:
        kind, spec, suffix, _ = rng.choices(KINDS, weights=weights)[0]
        place = rng.choice(PRACTICE_WORDS)
        if kind == "oncologist":
            full_name = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
            if full_name in used_names:
                continue
        else:
            full_name = f"{place} {suffix}"
            if full_name in used_names:
                continue
        used_names.add(full_name)
        pid = f"p-{len(providers) + 1:03d}"
        affiliation = full_name if kind != "oncologist" else f"{place} {suffix}"
        tier = rng.choices("ABC", weights=[2, 3, 3])[0]
        providers.append((pid, full_name, kind, spec, tier, rep_id))
        open_min = rng.choice([7 * 60 + 30, 8 * 60, 9 * 60])
        close_min = rng.choice([14 * 60 + 30, 16 * 60, 17 * 60, 17 * 60, 18 * 60])
        add_location(pid, affiliation, round(rng.uniform(*LAT_RANGE), 4),
                     round(rng.uniform(*LON_RANGE), 4), every_weekday(open_min, close_min))
        scale = {"A": 60, "B": 30, "C": 12}[tier]
        prescribing.append((pid, rng.randint(0, scale), rng.randint(0, scale), rng.randint(0, scale)))
        alerts.append((pid, rng.choice([0, 0, 0, 2, 4, 6, 8, 11])))

    con.executemany("INSERT INTO provider VALUES (?, ?, ?, ?, ?, ?)", providers)
    con.executemany("INSERT INTO location VALUES (?, ?, ?, ?, ?, ?)", locations)
    con.executemany("INSERT INTO location_hours VALUES (?, ?, ?, ?)", hours)
    con.executemany("INSERT INTO contact VALUES (?, ?, ?, ?, ?)", CONTACTS)
    con.executemany("INSERT INTO publication VALUES (?, ?, ?, ?, ?)",
                    [(pub, title, journal, year, use) for pub, _, title, journal, year, use, _ in PUBLICATIONS])
    con.executemany("INSERT INTO publication_author VALUES (?, ?)",
                    [(pub, pid) for pub, authors, *_ in PUBLICATIONS for pid in authors])
    con.executemany("INSERT INTO prescribing VALUES (?, ?, ?, ?)", prescribing)
    con.executemany("INSERT INTO lab_alert VALUES (?, ?, ?)",
                    [(pid, n, LAB_ALERT_SOURCES.get(pid)) for pid, n in alerts])

    # Visit history over the 90 days before the demo day.
    demo = date.fromisoformat(config.DEMO_DATE)
    visits = []
    for pid, dates in NAMED_VISITS.items():
        visits += [(pid, d, "completed", "face_to_face") for d in dates]
    for pid, dates in NAMED_PHONE_CALLS.items():
        visits += [(pid, d, "completed", "phone") for d in dates]
    for p in providers:
        if p[0] in NAMED_VISITS or p[0] in ("p-005", "p-006"):
            continue
        for _ in range(rng.randint(0, 4)):
            d = demo - timedelta(days=rng.randint(3, 90))
            if d.weekday() >= 5:          # reps don't call on weekends
                d -= timedelta(days=d.weekday() - 4)
            outcome = rng.choices(["completed", "cancelled", "rescheduled"], weights=[8, 1, 1])[0]
            call_type = rng.choices(["face_to_face", "phone"], weights=[3, 1])[0]
            visits.append((p[0], d.isoformat(), outcome, call_type))
    visits += [("p-005", (demo - timedelta(days=41)).isoformat(), "completed", "face_to_face"),
               ("p-006", (demo - timedelta(days=12)).isoformat(), "completed", "face_to_face")]
    visits.sort(key=lambda v: (v[1], v[0]))
    con.executemany("INSERT INTO visit VALUES (?, ?, ?, ?, ?, ?)",
                    [(f"v-{n:04d}", pid, rep_id, d, o, t) for n, (pid, d, o, t) in enumerate(visits, 1)])

    con.executemany(
        "INSERT INTO schedule_event (event_id, rep_id, date, start_min, end_min, kind, provider_id,"
        " location_id, title) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", _demo_schedule(rep_id))
    con.commit()
    con.close()
    log.info("generate seed=%d -> %d providers, %d locations, %d contacts, %d calls, %d schedule events: %s",
             seed, len(providers), len(locations), len(CONTACTS), len(visits), len(_demo_schedule(rep_id)), path)
    return path


if __name__ == "__main__":
    from navinib.log import setup_logging

    setup_logging()
    out = generate()
    print(f"Wrote synthetic world to {out}")
