"""Tests for the multi-agent graph (Checkpoint 5.1). The stub LLM keeps every run deterministic."""
import pytest

from navinib import config
from navinib.agents.coordinator import MAX_RETRIES, _retry
from navinib.agents.graph import NODES, _owned, run
from navinib.agents.llm import StubLLM
from navinib.agents.specs import AGENTS
from navinib.tools.drive import DriveTimes, build_cache

retrieval = pytest.importorskip("navinib.rag.retrieval")

HILLCREST = "p-005"

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


def test_morning_plan_runs_the_agents_in_order(con, col, drive):
    """Morning: Coordinator -> Targeting -> Route -> one Brief per call -> Compliance -> Coordinator."""
    final = run(con, MORNING, col=col, drive=drive)
    agents = _agents_in(final["trace"])
    calls = len(final["plan"]["visits"])
    assert calls >= 1
    assert agents == (["coordinator", "targeting", "coordinator", "route_replan", "coordinator"] + ["brief"] * calls
                      + ["coordinator", "compliance", "coordinator"])
    assert final["status"] == "awaiting_approval" and final["compliance"]["passed"]
    assert all(final["briefs"][v["provider_id"]]["status"] == "ok" for v in final["plan"]["visits"]), final["briefs"]


def test_cancellation_skips_targeting_and_uses_tot(con, col, drive):
    """Cancellation: the call is cancelled, Targeting is skipped, ToT replans, briefs only for the new calls."""
    final = run(con, CANCEL, col=col, drive=drive)
    assert "targeting" not in _agents_in(final["trace"]) and "targets" not in final
    assert final["plan"]["flow"] == "cancellation" and final["plan"]["reason"]
    assert con.execute("SELECT status FROM schedule_event WHERE event_id = 'ev-04'").fetchone()[0] == "cancelled"
    assert set(final["briefs"]) == {v["provider_id"] for v in final["plan"]["visits"]}


def test_retrieval_down_keeps_the_plan_with_brief_unavailable(con, drive):
    """No retrieval index: every call stays in the plan, its brief says unavailable, and the rep is told."""
    final = run(con, CANCEL, col=None, drive=drive)
    assert final["plan"]["visits"]
    assert all(b["status"] == "unavailable" for b in final["briefs"].values())
    assert "Brief unavailable" in final["message"] and final["status"] == "awaiting_approval"


def test_compliance_fail_goes_back_to_brief_and_is_fixed(con, col, drive):
    """A brief that mentions an off-label item is sent back with the problem; the rewrite passes."""
    final = run(con, MORNING, col=col, llm=LeakyLLM(), drive=drive)
    assert final["round"] == 1 and final["compliance"]["passed"] and final["flags"] == []
    assert any("compliance round 0: brief" in line and "do-not-raise" in line for line in final["trace"])
    resent = [p for p, r in final["brief_rounds"].items() if r == 1]
    assert resent and all(final["briefs"][p]["round"] == 1 for p in resent)


def test_problem_left_after_two_retries_is_flagged_not_hidden(con, col, drive):
    """When the problem won't go away, it is flagged to the rep after MAX_RETRIES, never silently fixed."""
    final = run(con, CANCEL, col=col, llm=StubbornLLM(), drive=drive)
    assert final["round"] == MAX_RETRIES and not final["compliance"]["passed"]
    assert final["flags"] and all(f["agent"] == "brief" for f in final["flags"])
    assert "Needs your review" in final["message"]


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
