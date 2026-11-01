from navinib import config
from navinib.tools import (cancel_event, location_contacts, provider_lookup, provider_scoring,
                           schedule_lookup)
from navinib.tools.providers import alert_display

LEE, LEIGH, SMITH, HILLCREST = "p-001", "p-002", "p-003", "p-005"   # fixed providers in data/generate.py


def test_demo_day_has_the_2pm_call(con):
    """The demo day has Dr. Smith's 2pm call."""
    day = schedule_lookup(con, "rep-01", config.DEMO_DATE)
    two_pm = [e for e in day["events"] if e["start"] == "14:00"]
    assert two_pm and two_pm[0]["provider_id"] == SMITH


def test_cancel_marks_event(con):
    """Cancelling an event marks it cancelled in the schedule."""
    cancel_event(con, "ev-04")
    day = schedule_lookup(con, "rep-01", config.DEMO_DATE)
    assert next(e for e in day["events"] if e["event_id"] == "ev-04")["status"] == "cancelled"


def test_lookup_filters_window_and_exclusions(con):
    # 2:00 to 3:30 window from the 2.1 example: Hillcrest closes at 2:30, so it must drop out.
    """Provider lookup honors the open-hours window, the drive limit and exclusions."""
    found = provider_lookup(con, "rep-01", config.DEMO_DATE, near=(40.455, -79.930), max_drive_min=20,
                            open_from=14 * 60, open_until=15 * 60 + 30, exclude={SMITH})
    ids = {p["provider_id"] for p in found}
    assert SMITH not in ids and HILLCREST not in ids
    assert all(p["drive_min"] <= 20 for p in found)


def test_scoring_is_ranked_and_explained(con):
    """Scores are sorted high to low, add up from their parts, and each has four reasons."""
    ranked = provider_scoring(con, provider_lookup(con, "rep-01", config.DEMO_DATE), config.DEMO_DATE)
    scores = [p["score"] for p in ranked]
    assert scores == sorted(scores, reverse=True)
    for p in ranked:
        assert abs(sum(p["breakdown"].values()) - p["score"]) < 0.01
        assert len(p["reasons"]) == 4


def test_alert_counts_are_suppressed():
    """Lab alert counts below the threshold are shown as '<5'."""
    assert alert_display(0) == "0"
    assert alert_display(3) == f"<{config.ALERT_SUPPRESS_BELOW}"
    assert alert_display(9) == "9"


def test_checkpoint_2_1_story_ranking(con):
    # Near Dr. Smith's office after her 2pm cancels: Hillcrest ranks first (it closes at 2:30,
    # which the route optimizer will catch), and Dr. Lee is the feasible runner-up.
    """Near Dr. Smith's office, Hillcrest ranks first and Dr. Lee second."""
    near = provider_lookup(con, "rep-01", config.DEMO_DATE, near=(40.455, -79.930), max_drive_min=20, exclude={SMITH})
    top = [p["provider_id"] for p in provider_scoring(con, near, config.DEMO_DATE)[:2]]
    assert top == [HILLCREST, LEE]


def test_provider_with_two_locations_follows_the_weekday(con):
    # Dr. Leigh: Northgate Mon/Wed/Fri 8-5, Southside clinic Tue/Thu 9-1.
    """Dr. Leigh appears at the right practice for the weekday, and nowhere on weekends."""
    def leigh_on(day):
        """Dr. Leigh's (affiliation, closing time) on a given day."""
        return [(p["affiliation"], p["close_min"]) for p in provider_lookup(con, "rep-01", day)
                if p["provider_id"] == LEIGH]
    assert leigh_on("2026-10-27") == [("Southside Hematology Clinic", 13 * 60)]   # Tuesday
    assert leigh_on("2026-10-28") == [("Northgate Cancer Center", 17 * 60)]       # Wednesday
    assert leigh_on("2026-10-31") == []                                           # Saturday


def test_two_location_provider_cant_fill_the_2pm_gap(con):
    """Dr. Leigh's Tuesday clinic closes at 1pm, so she can't fill the 2pm gap."""
    window = provider_lookup(con, "rep-01", config.DEMO_DATE, open_from=14 * 60, open_until=15 * 60 + 30)
    assert LEIGH not in {p["provider_id"] for p in window}


def test_lab_tech_is_a_contact_not_a_call_target(con):
    """Lab tech Kevin Tran is a contact at Dr. Lee's practice, never a provider to visit."""
    staff = {c["full_name"]: c for c in location_contacts(con, "L-001")}   # Dr. Lee's practice
    assert staff["Kevin Tran"]["role"] == "lab_technician"
    names = {p["full_name"] for p in provider_lookup(con, "rep-01", config.DEMO_DATE)}
    assert "Kevin Tran" not in names
    source = con.execute("SELECT source_contact_id FROM lab_alert WHERE provider_id = ?", (LEE,)).fetchone()[0]
    assert source == staff["Kevin Tran"]["contact_id"]


def test_phone_call_is_half_a_call_and_not_a_visit(con):
    """A phone call counts half toward goals and doesn't reset the last in-person visit."""
    types = {r[0] for r in con.execute("SELECT DISTINCT call_type FROM visit")}
    assert types == {"face_to_face", "phone"}
    # Dr. Lee's only October call is the Oct 14 phone call: half a call toward her goal of 4,
    # and her last in-person visit is still Sep 23 (34 days).
    lee = provider_scoring(con, [p for p in provider_lookup(con, "rep-01", config.DEMO_DATE)
                                 if p["provider_id"] == LEE], config.DEMO_DATE)[0]
    assert "0.5 of 4 tier-A calls done this month" in lee["reasons"]
    assert "last in-person visit 34 days ago" in lee["reasons"]


def test_tool_calls_are_logged(con, caplog):
    """Tool calls write what they were asked to the log."""
    import logging
    with caplog.at_level(logging.INFO, logger="navinib"):
        provider_lookup(con, "rep-01", config.DEMO_DATE, exclude={SMITH})
        cancel_event(con, "ev-04")
    text = caplog.text
    assert "provider_lookup day=2026-10-27" in text and "exclude=['p-003']" in text
    assert "cancel_event ev-04 -> cancelled" in text
