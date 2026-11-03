import pytest

from navinib import config
from navinib.planner.tot import tree_of_thought
from navinib.tools import cancel_event
from navinib.tools.drive import DriveTimes, build_cache

LEE, LEIGH, MARCHETTI, HILLCREST, SMITH = "p-001", "p-002", "p-012", "p-005", "p-003"


@pytest.fixture()
def estimate(con, tmp_path):
    """Straight-line drive times, so the rule checks never depend on the network."""
    path = tmp_path / "drive.json"
    build_cache(con, path, use_osrm=False)
    return DriveTimes(path)


def _replan(con, drive, now_min=14 * 60):
    """Cancel Dr. Smith's 2pm call (the 2.1 trigger) and run the Tree-of-Thought replan."""
    cancel_event(con, "ev-04")
    return tree_of_thought(con, "rep-01", config.DEMO_DATE, "ev-04", now_min, drive)


def _all_plans(r):
    """The chosen plan followed by its alternatives."""
    return [r["chosen"]] + r["alternatives"]


def test_checkpoint_4_1_story_with_real_drive_times(con):
    """With the committed OSRM times: two visits (Lee, then Marchetti) beat phoning Hillcrest + Lee."""
    drive = DriveTimes()
    if drive.source != "osrm":
        pytest.skip("needs the committed OSRM drive-time cache")
    r = _replan(con, drive)
    assert r["window"] == {"start": "14:00", "end": "15:30"}
    assert [(s["provider_id"], s["start"], s["end"]) for s in r["chosen"]["steps"]] == \
        [(LEE, "14:10", "14:40"), (MARCHETTI, "14:57", "15:27")]
    assert [a["shape"] for a in r["alternatives"]] == ["1 phone + 1 visit", "1 visit"]
    assert r["alternatives"][0]["steps"][0]["provider_id"] == HILLCREST
    assert "more than the best 1 phone + 1 visit plan" in r["reason"]


def test_hillcrest_can_only_be_reached_by_phone_before_it_closes(con, estimate):
    """Hillcrest closes at 2:30: a visit is ruled out, and any phone call to it starts at 2:00."""
    r = _replan(con, estimate)
    assert "closes 14:30" in r["trace"][0]["pruned"]["visit Hillcrest Cancer Center"]
    hillcrest = [s for p in _all_plans(r) for s in p["steps"] if s["provider_id"] == HILLCREST]
    assert hillcrest and all(s["call_type"] == "phone" and s["end"] <= "14:30" for s in hillcrest)


def test_every_plan_follows_the_hard_rules(con, estimate):
    """No plan overlaps itself, runs past the window, calls someone twice or books the cancelled provider."""
    r = _replan(con, estimate)
    for plan in _all_plans(r):
        ids = [s["provider_id"] for s in plan["steps"]]
        assert len(ids) == len(set(ids)) and SMITH not in ids
        assert sum(s["call_type"] == "phone" for s in plan["steps"]) <= config.TOT_MAX_PHONE
        prev_end = "14:00"
        for s in plan["steps"]:
            assert prev_end <= s["start"] < s["end"] <= "15:30"
            prev_end = s["end"]


def test_alternatives_are_different_strategies_and_rank_lower(con, estimate):
    """Alternatives each have a different shape from the chosen plan and a lower value."""
    r = _replan(con, estimate)
    shapes = [p["shape"] for p in _all_plans(r)]
    assert len(shapes) == len(set(shapes)) == 3
    assert all(a["value"] <= r["chosen"]["value"] for a in r["alternatives"])


def test_phone_is_only_for_providers_a_visit_cant_reach(con, estimate):
    """A provider who could be visited in person is never offered as a phone call."""
    r = _replan(con, estimate)
    visited = {s["provider_id"] for p in _all_plans(r) for s in p["steps"] if s["call_type"] == "face_to_face"}
    phoned = {s["provider_id"] for p in _all_plans(r) for s in p["steps"] if s["call_type"] == "phone"}
    assert not visited & phoned


def test_trace_records_each_level(con, estimate):
    """The trace lists, per level, the plans kept and the steps ruled out, for review and the report."""
    r = _replan(con, estimate)
    assert [t["level"] for t in r["trace"]] == list(range(1, len(r["trace"]) + 1))
    assert r["trace"][0]["kept"] and all(len(k) == 3 for k in r["trace"][0]["kept"])


def test_no_window_is_reported_not_guessed(con, estimate):
    """Too late in the gap for any call: no plan, and the reason says so."""
    r = _replan(con, estimate, now_min=15 * 60 + 10)
    assert r["window"] is None and r["chosen"]["steps"] == []
    assert "no open window" in r["reason"]


def test_start_point_follows_when_the_rep_hears(con, estimate):
    """On arrival the rep starts at the cancelled office; told at noon, from the last booked call (Okafor, L-005)."""
    assert _replan(con, estimate)["start_location_id"] == "L-004"
    noon = _replan(con, estimate, now_min=12 * 60)
    assert noon["start_location_id"] == "L-005" and "last booked call" in noon["start_reason"]


def test_live_location_overrides_the_guess(con, estimate):
    """When the rep says where they are, the replan starts there."""
    cancel_event(con, "ev-04")
    r = tree_of_thought(con, "rep-01", config.DEMO_DATE, "ev-04", 12 * 60, estimate, at_location_id="L-001")
    assert r["start_location_id"] == "L-001" and r["start_reason"] == "where the rep says they are"


def test_early_notice_opens_a_longer_window_and_reaches_hillcrest(con, estimate):
    """Told at noon, the window runs from the end of lunch (13:00), so Hillcrest can be visited before 2:30."""
    r = _replan(con, estimate, now_min=12 * 60)
    assert r["window"] == {"start": "13:00", "end": "15:30"}
    hillcrest = [s for s in r["chosen"]["steps"] if s["provider_id"] == HILLCREST]
    assert hillcrest and hillcrest[0]["call_type"] == "face_to_face" and hillcrest[0]["end"] <= "14:30"


def test_cancel_seen_at_lunch_starts_after_the_next_call(con, estimate):
    """Sydney's case: 11:30 call, lunch, 1:00 call, 2:00 call cancels (seen at 12:20), 4:30 call.

    The replan starts at the 1:00 call's office when it ends, and fills 1:30-3:30.
    """
    d = config.DEMO_DATE
    con.execute("DELETE FROM schedule_event")
    con.executemany(
        "INSERT INTO schedule_event (event_id, rep_id, date, start_min, end_min, kind, provider_id, location_id,"
        " title) VALUES (?, 'rep-01', ?, ?, ?, 'call', ?, ?, ?)",
        [("a", d, 11 * 60 + 30, 12 * 60, "p-004", "L-005", "11:30 call"),
         ("b", d, 13 * 60, 13 * 60 + 30, "p-006", "L-007", "1:00 call"),
         ("c", d, 14 * 60, 14 * 60 + 30, SMITH, "L-004", "2:00 call"),
         ("d", d, 16 * 60 + 30, 17 * 60, LEIGH, "L-003", "4:30 call")])
    cancel_event(con, "c")
    r = tree_of_thought(con, "rep-01", d, "c", 12 * 60 + 20, estimate)
    assert r["window"] == {"start": "13:30", "end": "15:30"}
    assert r["start_location_id"] == "L-007" and "last booked call" in r["start_reason"]
    assert r["chosen"]["steps"] and r["chosen"]["steps"][0]["start"] >= "13:30"
