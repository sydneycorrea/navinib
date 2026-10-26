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
KINDS = [  # (kind, specialty, practice suffix, weight)
    ("oncologist", "Medical Oncology", "Oncology Associates", 5),
    ("oncologist", "Hematology/Oncology", "Hematology Oncology", 4),
    ("community_practice", "Medical Oncology", "Community Cancer Care", 2),
    ("infusion_center", "Infusion Services", "Infusion Center", 1),
    ("cancer_center", "Comprehensive Oncology", "Cancer Center", 1),
]

# Fixed providers. ids match CALL_NOTES in rag/corpus.py where they appear there.
NAMED = [
    # id, name, kind, specialty, practice, tier, open, close, lat, lon, product_x, alerts
    ("lee", "Dr. Alicia Lee", "oncologist", "Hematology/Oncology", "Riverside Oncology Associates",
     "A", 8 * 60, 17 * 60, 40.462, -79.975, 95, 7),
    ("leigh", "Dr. Samuel Leigh", "oncologist", "Medical Oncology", "Northgate Cancer Center",
     "B", 8 * 60, 17 * 60, 40.505, -80.020, 12, 0),
    ("smith", "Dr. Helen Smith", "oncologist", "Medical Oncology", "Eastside Oncology Group",
     "A", 9 * 60, 17 * 60, 40.455, -79.930, 41, 6),
    ("okafor", "Dr. Daniel Okafor", "oncologist", "Hematology/Oncology", "Okafor Hematology Clinic",
     "B", 7 * 60 + 30, 16 * 60, 40.440, -80.005, 0, 0),
    # The 2.1 example: a high-priority center with an open alert that closes at 2:30.
    ("hillcrest", "Hillcrest Cancer Center", "cancer_center", "Comprehensive Oncology",
     "Hillcrest Cancer Center", "A", 8 * 60, 14 * 60 + 30, 40.448, -79.950, 58, 9),
    ("raman", "Dr. Priya Raman", "oncologist", "Hematology/Oncology", "Brightwater Hematology Group",
     "B", 9 * 60, 17 * 60 + 30, 40.470, -79.900, 22, 0),
]

# Last visits for the named providers, taken from the call notes.
NAMED_VISITS = {"lee": ["2026-08-12", "2026-09-23"], "leigh": ["2026-09-30"],
                "smith": ["2026-10-02"], "okafor": ["2026-10-07"]}

REP = ("rep-01", "Jordan Rivera", "Pittsburgh North", "mgr-01", 40.452, -79.985)
MANAGER = ("mgr-01", "Morgan Ellison", "Mid-Atlantic")


def _demo_schedule(rep_id: str) -> list[tuple]:
    """The scripted demo day: the 2pm call with Dr. Smith is the one that cancels."""
    d = config.DEMO_DATE
    return [
        ("ev-01", rep_id, d, 8 * 60, 8 * 60 + 30, "task", None, "Admin and expense report"),
        ("ev-02", rep_id, d, 8 * 60 + 45, 9 * 60 + 30, "call", "okafor", "Call: Dr. Daniel Okafor"),
        ("ev-03", rep_id, d, 12 * 60, 13 * 60, "task", None, "Lunch"),
        ("ev-04", rep_id, d, 14 * 60, 15 * 60 + 30, "call", "smith", "Call: Dr. Helen Smith"),
        ("ev-05", rep_id, d, 16 * 60 + 30, 17 * 60, "call", "raman", "Call: Dr. Priya Raman"),
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

    providers, prescribing, alerts = [], [], []
    for pid, name, kind, spec, practice, tier, o, c, lat, lon, px, al in NAMED:
        addr = f"{rng.randint(100, 2999)} {rng.choice(STREETS)}, Pittsburgh, PA"
        providers.append((pid, name, kind, spec, practice, addr, lat, lon, tier, rep_id, o, c))
        prescribing.append((pid, px, rng.randint(5, 60), rng.randint(0, 40)))
        alerts.append((pid, al))

    used_names = {p[1] for p in providers}
    weights = [k[3] for k in KINDS]
    i = 0
    while len(providers) < config.N_PROVIDERS:
        i += 1
        kind, spec, suffix, _ = rng.choices(KINDS, weights=weights)[0]
        place = rng.choice(PRACTICE_WORDS)
        if kind == "oncologist":
            name = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"
            if name in used_names:
                continue
        else:
            name = f"{place} {suffix}"
            if name in used_names:
                continue
        used_names.add(name)
        pid = f"p-{i:03d}"
        practice = name if kind != "oncologist" else f"{place} {suffix}"
        addr = f"{rng.randint(100, 2999)} {rng.choice(STREETS)}, Pittsburgh, PA"
        lat = round(rng.uniform(*LAT_RANGE), 4)
        lon = round(rng.uniform(*LON_RANGE), 4)
        tier = rng.choices("ABC", weights=[2, 3, 3])[0]
        open_min = rng.choice([7 * 60 + 30, 8 * 60, 9 * 60])
        close_min = rng.choice([14 * 60 + 30, 16 * 60, 17 * 60, 17 * 60, 18 * 60])
        providers.append((pid, name, kind, spec, practice, addr, lat, lon, tier, rep_id,
                          open_min, close_min))
        scale = {"A": 60, "B": 30, "C": 12}[tier]
        prescribing.append((pid, rng.randint(0, scale), rng.randint(0, scale), rng.randint(0, scale)))
        alerts.append((pid, rng.choice([0, 0, 0, 2, 4, 6, 8, 11])))

    con.executemany("INSERT INTO provider VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", providers)
    con.executemany("INSERT INTO prescribing VALUES (?, ?, ?, ?)", prescribing)
    con.executemany("INSERT INTO lab_alert VALUES (?, ?)", alerts)

    # Visit history over the 90 days before the demo day.
    demo = date.fromisoformat(config.DEMO_DATE)
    visits = []
    for pid, dates in NAMED_VISITS.items():
        visits += [(pid, d, "completed") for d in dates]
    for p in providers:
        if p[0] in NAMED_VISITS or p[0] in ("hillcrest", "raman"):
            continue
        for _ in range(rng.randint(0, 4)):
            d = demo - timedelta(days=rng.randint(3, 90))
            outcome = rng.choices(["completed", "cancelled", "rescheduled"], weights=[8, 1, 1])[0]
            visits.append((p[0], d.isoformat(), outcome))
    visits += [("hillcrest", (demo - timedelta(days=41)).isoformat(), "completed"),
               ("raman", (demo - timedelta(days=12)).isoformat(), "completed")]
    visits.sort(key=lambda v: (v[1], v[0]))
    con.executemany("INSERT INTO visit VALUES (?, ?, ?, ?, ?)",
                    [(f"v-{n:04d}", pid, rep_id, d, o) for n, (pid, d, o) in enumerate(visits, 1)])

    con.executemany(
        "INSERT INTO schedule_event (event_id, rep_id, date, start_min, end_min, kind, provider_id, title)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _demo_schedule(rep_id))
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    out = generate()
    print(f"Wrote synthetic world to {out}")
