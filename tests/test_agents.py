"""Tests for the multi-agent graph (Checkpoint 5.1). The stub LLM keeps every run deterministic."""
import pytest

from navinib import config
from navinib.agents.coordinator import MAX_RETRIES, _retry, approve_plan
from navinib.agents.graph import NODES, _owned, brief_next_call, run
from navinib.agents.llm import StubLLM, citations
from navinib.agents.specs import AGENTS
from navinib.tools.drive import DriveTimes, build_cache
from navinib.tools.schedule import to_min

retrieval = pytest.importorskip("navinib.rag.retrieval")

LEE, HILLCREST = "p-001", "p-005"

MORNING = {"flow": "morning", "rep_id": "rep-01", "day": config.DEMO_DATE}
CANCEL = {"flow": "cancellation", "rep_id": "rep-01", "day": config.DEMO_DATE, "event_id": "ev-04",
          "now_min": 14 * 60}


@pytest.fixture(scope="module")
def col(tmp_path_factory):
    """One retrieval index for the file (skipped if the embedding model can't download)."""
    try:
        return retrieval.build_index(tmp_path_factory.mktemp("chroma"))
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")


@pytest.fixture()
def drive(con, tmp_path):
    """Straight-line drive times, so the tests never call the network."""
    path = tmp_path / "drive.json"
    build_cache(con, path, use_osrm=False)
    return DriveTimes(path)


class LeakyLLM(StubLLM):
    """Mentions an off-label item in each provider's first brief, then follows the feedback."""

    def __init__(self):
        """Remember which providers already got a brief."""
        self.seen = set()

    def _brief(self, d):
        """First brief per provider leaks its first do-not-raise item; the rewrite is clean."""
        text = super()._brief(d)
        pid, dnr = d["visit"]["provider_id"], d["inputs"]["do_not_raise"]
        if pid not in self.seen and dnr:
            self.seen.add(pid)
            text += f"\nAsk about the new first-line data [{dnr[0]['ref']}]."
        return text


class StubbornLLM(StubLLM):
    """Its claims check always finds an unsupported statement, whatever the brief says."""

    def _claims(self, d):
        """Never passes."""
        return "Says response rates are 'best in class', which no source supports."


def _agents_in(trace):
    """The agent that wrote each trace line, in order."""
    return [line.split(" ")[0].rstrip(":") for line in trace]


def _briefed(con):
    """Event ids that have a saved brief."""
    return {r[0] for r in con.execute("SELECT event_id FROM call_brief")}


def test_morning_plan_has_no_briefs_until_approved(con, col, drive):
    """Morning: Coordinator -> Targeting -> Route -> Compliance (providers and times) -> rep. No brief written."""
    final = run(con, MORNING, col=col, drive=drive)
    assert _agents_in(final["trace"]) == ["coordinator", "targeting", "coordinator", "route_replan", "coordinator",
                                          "compliance", "coordinator"]
    assert final["plan"]["visits"] and final["status"] == "awaiting_approval" and final["compliance"]["passed"]
    assert not final.get("briefs") and _briefed(con) == set()


def test_cancellation_skips_targeting_and_uses_tot(con, col, drive):
    """Cancellation: the call is cancelled, Targeting is skipped, ToT replans, and no brief is written yet."""
    final = run(con, CANCEL, col=col, drive=drive)
    assert "targeting" not in _agents_in(final["trace"]) and "targets" not in final
    assert final["plan"]["flow"] == "cancellation" and final["plan"]["reason"]
    assert con.execute("SELECT status FROM schedule_event WHERE event_id = 'ev-04'").fetchone()[0] == "cancelled"
    assert not final.get("briefs")


def test_brief_next_call_writes_one_brief_when_its_time_comes(con, col):
    """Dr. Okafor is at 8:45, 6 min from home: due 8:24 (start - drive - 15 min reading), written from 8:19.
    Nothing at 8:10, his brief at 8:20, nothing after."""
    early = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("08:10"), col=col)
    assert early["status"] == "nothing_due" and "written at 08:19 for the 08:45 call" in early["message"]
    final = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("08:20"), col=col)
    assert final["status"] == "brief_ready" and set(final["briefs"]) == {"p-004"}
    assert final["due"][0]["drive_min"] == 6 and final["due"][0]["from"] == "home"
    assert _briefed(con) == {"ev-02"}
    again = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("08:25"), col=col)
    assert again["status"] == "nothing_due" and not again.get("briefs")   # never written twice


def test_a_cancelled_call_never_gets_a_brief_and_the_replan_is_briefed_right_away(con, col):
    """Dr. Smith's 2pm cancels at 13:05: no brief for it. The first replacement starts at 13:14, 9 min away,
    so its brief time has already passed and it is written as soon as the rep approves; the next one waits."""
    drive = DriveTimes()
    if drive.source != "osrm":
        pytest.skip("needs the committed OSRM drive-time cache")
    final = run(con, {**CANCEL, "now_min": to_min("13:05")}, col=col, drive=drive)
    assert brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("13:05"), col=col)["status"] == "nothing_due"
    booked = approve_plan(con, final)
    first = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("13:05"), col=col)
    assert first["status"] == "brief_ready" and _briefed(con) == {booked[0]}
    assert first["due"][0]["write_min"] < to_min("13:05")              # immediate, not an hour ahead
    assert brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("13:06"), col=col)["status"] == "nothing_due"
    assert "ev-04" not in _briefed(con)


class FixedDrive(DriveTimes):
    """Every drive takes the same number of minutes."""

    def __init__(self, minutes):
        """Remember the drive time; there is no cache."""
        self.minutes, self.source, self.estimated = minutes, "test", 0

    def between(self, a, b):
        """The fixed drive time (0 to the same place)."""
        return 0 if a["id"] == b["id"] else self.minutes


def test_a_long_drive_gets_its_brief_earlier_than_an_hour_ahead(con, col):
    """Dr. Okafor's 8:45 call 50 min away: due 7:40 (8:45 - 50 - 15), written from 7:35, which is
    more than an hour ahead. The same call 5 min away waits until 8:20."""
    far = FixedDrive(50)
    assert brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("07:30"), col=col, drive=far)["status"] == \
        "nothing_due"
    final = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("07:35"), col=col, drive=far)
    assert final["status"] == "brief_ready" and final["due"][0]["due_min"] == to_min("07:40")
    con.execute("DELETE FROM call_brief")
    near = FixedDrive(5)
    assert brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("08:15"), col=col, drive=near)["status"] == \
        "nothing_due"
    assert brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("08:20"), col=col, drive=near)["status"] == \
        "brief_ready"


def _approve_2pm_replan(con, col):
    """Cancel the 2pm on arrival with the committed road times and book the replan (Dr. Lee 14:10, Marchetti)."""
    drive = DriveTimes()
    if drive.source != "osrm":
        pytest.skip("needs the committed OSRM drive-time cache")
    final = run(con, CANCEL, col=col, drive=drive)
    approve_plan(con, final)
    return final


def test_retrieval_down_gives_a_partial_brief_from_structured_data(con):
    """No retrieval index: the due call gets a partial brief that says why the provider matters but
    makes no product claims, and the rep is told it is partial."""
    _approve_2pm_replan(con, None)
    final = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("14:00"), col=None)
    lee = final["briefs"][LEE]
    assert lee["status"] == "partial" and "retrieval failed" in lee["why"]
    assert "label and trial sources unavailable" in lee["text"]
    assert "0.5 of 4 tier-A calls done this month" in lee["text"]        # why she ranked high
    assert "2026-10-14 phone (completed)" in lee["text"]                 # recent calls
    assert "Kevin Tran (lab technician)" in lee["text"]
    assert citations(lee["text"]) == [] and "Product X" not in lee["text"]
    assert final["compliance"]["passed"] and "It is partial" in final["message"]


class DownLLM(StubLLM):
    """A model that is down: every call fails."""

    def complete(self, task, system, user, data=None):
        """Always fail, like a free model that is rate-limited or offline."""
        raise ConnectionError("model offline")


def test_model_down_still_gives_the_rep_the_plan_and_a_partial_brief(con, col, drive):
    """Model down: the plan summary is built in code, and the due call gets a partial brief."""
    final = run(con, CANCEL, col=col, llm=DownLLM(), drive=drive)
    assert final["plan"]["visits"] and final["status"] == "awaiting_approval"
    assert final["message"].startswith("Written summary unavailable (model down). Proposed:")
    approve_plan(con, final)
    b = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("14:00"), col=col, llm=DownLLM())
    assert all(x["status"] == "partial" and "model unavailable" in x["why"] for x in b["briefs"].values())


def test_estimated_drive_times_are_labeled(con, col, tmp_path):
    """No live road times (no cache file): the plan and the rep's summary say the times are estimates."""
    final = run(con, CANCEL, col=col, drive=DriveTimes(tmp_path / "missing.json"))
    assert final["plan"]["drive_estimated"]
    assert "Drive times are estimates" in final["message"]


def test_compliance_fail_goes_back_to_brief_and_is_fixed(con, col):
    """Dr. Raman's brief (16:30 call, written from 16:03) mentions an off-label item; it is sent back once and the rewrite passes."""
    final = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("16:05"), col=col, llm=LeakyLLM())
    assert final["round"] == 1 and final["compliance"]["passed"] and final["flags"] == []
    assert any("compliance round 0: brief" in line and "do-not-raise" in line for line in final["trace"])
    assert final["briefs"]["p-006"]["round"] == 1 and _briefed(con) == {"ev-05"}


def test_problem_left_after_two_retries_is_flagged_not_hidden(con, col):
    """When the problem won't go away, it is flagged to the rep after MAX_RETRIES, never silently fixed."""
    final = brief_next_call(con, "rep-01", config.DEMO_DATE, to_min("16:05"), col=col, llm=StubbornLLM())
    assert final["round"] == MAX_RETRIES and not final["compliance"]["passed"]
    assert final["flags"] and all(f["agent"] == "brief" for f in final["flags"])
    assert "Needs your review" in final["message"]
    assert con.execute("SELECT status FROM call_brief WHERE event_id = 'ev-05'").fetchone()[0] == "flagged"


def test_route_problem_goes_back_to_route_with_the_provider_excluded():
    """A timing problem sends the work to Route and Replan and leaves that provider out next time."""
    state = {"request": CANCEL, "round": 0, "plan": {"round": 0, "visits": []}}
    update = _retry(state, [{"agent": "route_replan", "provider_id": "p-001", "problem": "outside office hours"}], 1)
    assert update["next"] == "route_replan" and update["route_round"] == 1
    assert update["feedback"]["exclude"] == ["p-001"]


def test_an_agent_cannot_write_another_agents_state(con):
    """Each agent may only write its own part of the shared state."""
    rogue = _owned("targeting", lambda state, config: {"plan": {}, "trace": []})
    with pytest.raises(PermissionError, match="targeting tried to write"):
        rogue({}, {})


def test_every_node_has_an_agent_spec():
    """Five agents, defined by role, goal and tools; only Coordinator, Brief and Compliance call the LLM."""
    assert set(NODES) == set(AGENTS)
    assert {a.name for a in AGENTS.values() if a.uses_llm} == {"coordinator", "brief", "compliance"}
    assert all(a.role and a.goal and a.tools for a in AGENTS.values())


def test_underfilled_day_asks_targeting_once_for_the_next_candidates(con, col, drive, monkeypatch):
    """Route drops a provider that doesn't fit and leaves the window empty: Targeting is asked once more.

    Day: office work until 13:00, a call at Dr. Smith's office until 14:00, then free. With one candidate per window,
    Route only gets Hillcrest, which closes at 14:30 and can't fit. The Coordinator asks Targeting
    for the next candidates without Hillcrest, Route runs again, and the gap gets a visit.
    """
    from navinib.agents import route_replan as rr
    monkeypatch.setattr(rr, "N_CANDIDATES", 1)
    d = config.DEMO_DATE
    con.execute("DELETE FROM schedule_event")
    con.executemany("INSERT INTO schedule_event (event_id, rep_id, date, start_min, end_min, kind, provider_id,"
                    " location_id, title) VALUES (?, 'rep-01', ?, ?, ?, ?, ?, ?, ?)",
                    [("s0", d, 480, 780, "task", None, None, "Office work"),
                     ("s1", d, 780, 840, "call", "p-003", "L-004", "Smith")])
    final = run(con, MORNING, col=col, drive=drive)
    agents = _agents_in(final["trace"])
    assert agents.count("targeting") == 2 and agents.count("route_replan") == 2   # one refill round, no more
    assert final["refill"] == {"round": 1, "exclude": [HILLCREST]}
    assert HILLCREST not in {t["provider_id"] for t in final["targets"]["ranked"]}
    after_smith = [v for v in final["plan"]["visits"] if v["start"] >= "14:00"]
    assert after_smith and final["status"] == "awaiting_approval"
