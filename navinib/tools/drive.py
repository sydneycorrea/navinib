"""Drive times between the rep's home and provider locations.

Real road times come from OSRM's free table API and are cached to
data/drive_times.json, so planning and the demo work offline. When the cache
is missing a pair (or OSRM can't be reached), the straight-line estimate in
providers.drive_minutes is used and the result says so.

Build or refresh the cache:  python -m navinib.tools.drive
"""
import json
import sqlite3
import urllib.request
from pathlib import Path

from navinib import config
from navinib.tools.providers import drive_minutes

CACHE_PATH = config.ROOT / "data" / "drive_times.json"
OSRM_URL = "https://router.project-osrm.org/table/v1/driving/{coords}?annotations=duration"


def _points(con: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    pts = {f"home:{r['rep_id']}": (r["home_lat"], r["home_lon"]) for r in con.execute("SELECT * FROM rep")}
    pts.update({r["location_id"]: (r["lat"], r["lon"]) for r in con.execute("SELECT * FROM location")})
    return pts


def _osrm_minutes(points: dict[str, tuple[float, float]]) -> dict[str, dict[str, int]]:
    ids = list(points)
    coords = ";".join(f"{lon},{lat}" for lat, lon in points.values())   # OSRM wants lon,lat
    with urllib.request.urlopen(OSRM_URL.format(coords=coords), timeout=30) as resp:
        seconds = json.load(resp)["durations"]
    return {a: {b: max(1, round(seconds[i][j] / 60)) if i != j else 0 for j, b in enumerate(ids)}
            for i, a in enumerate(ids)}


def build_cache(con: sqlite3.Connection, path: Path | str = CACHE_PATH, use_osrm: bool = True) -> dict:
    points = _points(con)
    source = "estimate"
    minutes = None
    if use_osrm:
        try:
            minutes, source = _osrm_minutes(points), "osrm"
        except (OSError, KeyError, ValueError):   # offline, blocked or bad response
            pass
    if minutes is None:
        minutes = {a: {b: 0 if a == b else drive_minutes(*pa, *pb) for b, pb in points.items()}
                   for a, pa in points.items()}
    cache = {"source": source, "minutes": minutes}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache))
    return cache


class DriveTimes:
    """Looks up cached road minutes by point id, falling back to the estimate."""

    def __init__(self, path: Path | str = CACHE_PATH):
        p = Path(path)
        cache = json.loads(p.read_text()) if p.exists() else {"source": "estimate", "minutes": {}}
        self.source = cache["source"]
        self._minutes = cache["minutes"]

    def between(self, a: dict, b: dict) -> int:
        """a and b are dicts with 'id', 'lat' and 'lon'."""
        if a["id"] == b["id"]:
            return 0
        cached = self._minutes.get(a["id"], {}).get(b["id"])
        return cached if cached is not None else drive_minutes(a["lat"], a["lon"], b["lat"], b["lon"])


if __name__ == "__main__":
    from navinib.db import connect

    c = build_cache(connect())
    print(f"Wrote {len(c['minutes'])} x {len(c['minutes'])} drive times ({c['source']}) to {CACHE_PATH}")
