"""SQLite schema and connection. Structured data only; free text for retrieval lives in Chroma."""
import sqlite3
from pathlib import Path

from navinib.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS manager (
    manager_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    region     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rep (
    rep_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    territory  TEXT NOT NULL,
    manager_id TEXT NOT NULL REFERENCES manager(manager_id),
    home_lat   REAL NOT NULL,
    home_lon   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS provider (
    provider_id   TEXT PRIMARY KEY,       -- sequence number, e.g. p-001
    full_name     TEXT NOT NULL,          -- fictional
    kind          TEXT NOT NULL,          -- oncologist | cancer_center | infusion_center | community_practice
    specialty     TEXT NOT NULL,
    affiliation   TEXT NOT NULL,          -- practice, hospital or cancer center
    address       TEXT NOT NULL,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    tier          TEXT NOT NULL CHECK (tier IN ('A', 'B', 'C')),
    rep_id        TEXT NOT NULL REFERENCES rep(rep_id),
    open_min      INTEGER NOT NULL,       -- office hours, minutes after midnight, Mon-Fri
    close_min     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prescribing (
    provider_id      TEXT PRIMARY KEY REFERENCES provider(provider_id),
    product_x        INTEGER NOT NULL,    -- synthetic claims per quarter
    competitor_a     INTEGER NOT NULL,
    competitor_b     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS visit (
    visit_id    TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES provider(provider_id),
    rep_id      TEXT NOT NULL REFERENCES rep(rep_id),
    date        TEXT NOT NULL,            -- ISO date
    outcome     TEXT NOT NULL CHECK (outcome IN ('completed', 'cancelled', 'rescheduled'))
);
CREATE TABLE IF NOT EXISTS lab_alert (
    provider_id TEXT PRIMARY KEY REFERENCES provider(provider_id),
    open_count  INTEGER NOT NULL          -- provider level only, never patient level
);
CREATE TABLE IF NOT EXISTS schedule_event (
    event_id    TEXT PRIMARY KEY,
    rep_id      TEXT NOT NULL REFERENCES rep(rep_id),
    date        TEXT NOT NULL,
    start_min   INTEGER NOT NULL,
    end_min     INTEGER NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('call', 'task')),
    provider_id TEXT REFERENCES provider(provider_id),
    title       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'planned'  -- planned | cancelled | completed
);
CREATE TABLE IF NOT EXISTS plan_decision (       -- long-term memory of accept/edit/reject
    decision_id TEXT PRIMARY KEY,
    rep_id      TEXT NOT NULL REFERENCES rep(rep_id),
    date        TEXT NOT NULL,
    decision    TEXT NOT NULL CHECK (decision IN ('accepted', 'edited', 'rejected')),
    detail      TEXT
);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
