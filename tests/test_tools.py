from navinib import config
from navinib.tools import cancel_event, provider_lookup, provider_scoring, schedule_lookup
from navinib.tools.providers import alert_display

LEE, SMITH, HILLCREST = "p-001", "p-003", "p-005"   # fixed providers in data/generate.py


def test_demo_day_has_the_2pm_call(con):
    day = schedule_lookup(con, "rep-01", config.DEMO_DATE)
    two_pm = [e for e in day["events"] if e["start"] == "14:00"]
    assert two_pm and two_pm[0]["provider_id"] == SMITH


def test_cancel_marks_event(con):
    cancel_event(con, "ev-04")
    day = schedule_lookup(con, "rep-01", config.DEMO_DATE)
    assert next(e for e in day["events"] if e["event_id"] == "ev-04")["status"] == "cancelled"


def test_lookup_filters_window_and_exclusions(con):
    # 2:00 to 3:30 window from the 2.1 example: Hillcrest closes at 2:30, so it must drop out.
    found = provider_lookup(con, "rep-01", near=(40.455, -79.930), max_drive_min=20,
                            open_from=14 * 60, open_until=15 * 60 + 30, exclude={SMITH})
    ids = {p["provider_id"] for p in found}
    assert SMITH not in ids and HILLCREST not in ids
    assert all(p["drive_min"] <= 20 for p in found)


def test_scoring_is_ranked_and_explained(con):
    ranked = provider_scoring(con, provider_lookup(con, "rep-01"), config.DEMO_DATE)
    scores = [p["score"] for p in ranked]
    assert scores == sorted(scores, reverse=True)
    for p in ranked:
        assert abs(sum(p["breakdown"].values()) - p["score"]) < 0.01
        assert len(p["reasons"]) == 4


def test_alert_counts_are_suppressed():
    assert alert_display(0) == "0"
    assert alert_display(3) == f"<{config.ALERT_SUPPRESS_BELOW}"
    assert alert_display(9) == "9"


def test_checkpoint_2_1_story_ranking(con):
    # Near Dr. Smith's office after her 2pm cancels: Hillcrest ranks first (it closes at 2:30,
    # which the route optimizer will catch), and Dr. Lee is the feasible runner-up.
    near = provider_lookup(con, "rep-01", near=(40.455, -79.930), max_drive_min=20, exclude={SMITH})
    top = [p["provider_id"] for p in provider_scoring(con, near, config.DEMO_DATE)[:2]]
    assert top == [HILLCREST, LEE]
