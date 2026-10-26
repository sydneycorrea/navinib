import pytest

from navinib import config
from navinib.tools import cancel_event, provider_lookup, provider_scoring, schedule_lookup
from navinib.tools.drive import DriveTimes, build_cache
from navinib.tools.route import free_windows, plan_route

LEE, SMITH, HILLCREST = "p-001", "p-003", "p-005"


@pytest.fixture()
def drive(con, tmp_path):
    path = tmp_path / "drive.json"
    build_cache(con, path, use_osrm=False)   # tests never call the network
    return DriveTimes(path)


def _start_at(con, location_id):
    loc = con.execute("SELECT * FROM location WHERE location_id = ?", (location_id,)).fetchone()
    return {"id": location_id, "lat": loc["lat"], "lon": loc["lon"]}


def _candidates_near(con, start, n=3):
    near = provider_lookup(con, "rep-01", config.DEMO_DATE, near=(start["lat"], start["lon"]),
                           max_drive_min=20, exclude={SMITH})
    return provider_scoring(con, near, config.DEMO_DATE)[:n]


def test_free_windows_keep_the_prep_buffer(con):
    cancel_event(con, "ev-04")                           # Dr. Smith's 2:00 call
    day = schedule_lookup(con, "rep-01", config.DEMO_DATE)
    w = free_windows(day, now_min=14 * 60)[0]
    # Next booked call is 4:30, minus the one-hour buffer: open 2:00 to 3:30 (Checkpoint 2.1).
    assert (w["start"], w["end"], w["before_event"]) == ("14:00", "15:30", "ev-05")


def test_free_windows_track_where_the_rep_last_was(con):
    day = schedule_lookup(con, "rep-01", config.DEMO_DATE)
    morning = free_windows(day)[0]
    assert morning["start"] == "09:30" and morning["last_location_id"] == "L-005"   # after Dr. Okafor


def test_checkpoint_2_1_replan(con, drive):
    start = _start_at(con, "L-004")
    plan = plan_route(start, 14 * 60, 15 * 60 + 30, _candidates_near(con, start), drive, max_visits=1)
    assert plan["feasible"]
    assert [v["provider_id"] for v in plan["visits"]] == [LEE]
    hillcrest = next(d for d in plan["dropped"] if d["provider_id"] == HILLCREST)
    assert hillcrest["reason"].startswith("closes at 14:30")


def test_planned_visits_respect_office_hours_and_window(con, drive):
    start = _start_at(con, "L-004")
    cands = _candidates_near(con, start, n=8)
    hours = {c["location_id"]: (c["open_min"], c["close_min"]) for c in cands}
    plan = plan_route(start, 14 * 60, 15 * 60 + 30, cands, drive)
    assert len(plan["visits"]) >= 2
    prev_end = 14 * 60
    for v in plan["visits"]:
        open_min, close_min = hours[v["location_id"]]
        assert v["start_min"] >= max(open_min, prev_end + v["drive_min"])
        assert v["end_min"] <= min(close_min, 15 * 60 + 30)
        prev_end = v["end_min"]
    assert len(plan["visits"]) + len(plan["dropped"]) == len(cands)


def test_no_feasible_plan_is_reported_not_guessed(con, drive):
    start = _start_at(con, "L-004")
    plan = plan_route(start, 14 * 60, 14 * 60 + 20, _candidates_near(con, start), drive)
    assert not plan["feasible"] and plan["visits"] == []
    assert all("reason" in d for d in plan["dropped"])


def test_drive_cache_falls_back_to_estimate(con, drive):
    assert drive.source == "estimate"
    a, b = _start_at(con, "L-001"), _start_at(con, "L-004")
    assert drive.between(a, a) == 0 and drive.between(a, b) > 0
