"""The multi-agent graph (Checkpoint 5.1): a LangGraph supervisor with the Coordinator at the hub.

Every agent is a node that returns to the Coordinator; no agent calls another.
Plans are approved by the rep first; briefs are written just in time, one call ahead
(brief_next_call), so no tokens go to a call that cancels.

Run:
    python -m navinib.agents.graph morning               # plan the demo day
    python -m navinib.agents.graph cancel ev-04 14:00    # Dr. Smith's 2pm cancels on arrival
    python -m navinib.agents.graph brief 08:20           # brief the next call whose brief time has come
    python -m navinib.agents.graph demo                  # the whole story (regenerates the database)
"""
import sqlite3
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from navinib import audit
from navinib.agents.brief import brief
from navinib.agents.compliance import compliance
from navinib.agents.coordinator import approve_plan, briefs_to_write, coordinator, next_step
from navinib.agents.llm import LLM, get_llm
from navinib.agents.routing import routing
from navinib.agents.specs import AGENTS
from navinib.agents.state import NavinibState
from navinib.agents.targeting import targeting
from navinib.config import GRAPH_STEPS
from navinib.log import get_logger
from navinib.tools.drive import DriveTimes

log = get_logger("navinib.agents.graph")

NODES = {"coordinator": coordinator, "targeting": targeting, "routing": routing, "brief": brief,
         "compliance": compliance}


def _owned(name: str, fn):
    """Wrap an agent so it can only write its own part of the shared state (plus the trace)."""
    allowed = AGENTS[name].writes | {"trace"}

    def node(state, config):
        """Run the agent and refuse any write outside what it owns."""
        out = fn(state, config)
        extra = set(out) - allowed
        if extra:
            raise PermissionError(f"{name} tried to write {sorted(extra)}")
        return out
    return node


def _route(state: dict):
    """After the Coordinator: a Brief run for the due call, or the single next agent, or the end."""
    nxt = next_step(state)
    if nxt == "brief":
        return [Send("brief", payload) for payload in briefs_to_write(state)]
    return END if nxt == "end" else nxt


def build_graph():
    """Compile the hub-and-spoke graph."""
    g = StateGraph(NavinibState)
    for name, fn in NODES.items():
        g.add_node(name, _owned(name, fn))
    g.add_edge(START, "coordinator")
    g.add_conditional_edges("coordinator", _route, ["targeting", "routing", "brief", "compliance", END])
    for name in NODES:
        if name != "coordinator":
            g.add_edge(name, "coordinator")
    return g.compile()


def run(con: sqlite3.Connection, request: dict, col=None, llm: LLM | None = None,
        drive: DriveTimes | None = None, log_path=None) -> dict:
    """Run one request through the agents and return the final shared state.

    request: {'flow': 'morning' | 'cancellation' | 'brief_next', 'rep_id', 'day', and for a
    cancellation 'event_id' and 'now_min' (plus optional 'at_location_id'), for brief_next
    'now_min'}. col is the retrieval index; without it a brief is a partial one from
    structured data. An optional 'message' is the rep's own words, screened before any agent runs.
    Every run is appended to logs/runs.jsonl with a trace id (final['run_id']).
    """
    llm = llm or get_llm()
    run_id, started = audit.new_run_id(), time.time()
    config = {"configurable": {"con": con, "col": col, "llm": llm, "drive": drive or DriveTimes()},
              "recursion_limit": GRAPH_STEPS}
    log.info("run %s %s (model %s)", run_id, request, llm.name)
    final = build_graph().invoke({"request": request}, config=config)
    audit.record(run_id, request, final, llm.name, started, log_path)
    return {**final, "run_id": run_id}


def brief_next_call(con: sqlite3.Connection, rep_id: str, day: str, now_min: int, col=None,
                    llm: LLM | None = None, drive: DriveTimes | None = None, log_path=None) -> dict:
    """The just-in-time trigger: brief the next booked call once its brief time has come.

    Call it on a timer (or when the rep opens the app). It writes at most one brief, and
    never one for a cancelled call or a call that already has one.
    """
    return run(con, {"flow": "brief_next", "rep_id": rep_id, "day": day, "now_min": now_min}, col=col, llm=llm,
               drive=drive, log_path=log_path)


def _show(final: dict) -> None:
    """Print a run for the command line."""
    print("Trace:")
    for line in final["trace"]:
        print(f"  {line}")
    for b in final.get("briefs", {}).values():
        print(f"\nBrief [{b['status']}]:\n{b['text']}")
    for l in final.get("lanes", []):
        print(f"  [{l['lane_name']}] {l['item']}: {l['why']}")
    print(f"\nStatus: {final['status']}\n{final.get('message', '')}\n")


if __name__ == "__main__":
    import sys

    from navinib import config as cfg
    from navinib.data.generate import generate
    from navinib.db import connect
    from navinib.log import setup_logging
    from navinib.rag.retrieval import build_index
    from navinib.tools.schedule import to_min

    setup_logging()
    args = sys.argv[1:] or ["morning"]
    day = cfg.DEMO_DATE
    if args[0] == "demo":
        # Dr. Smith's 2pm cancels at 13:05; the rep approves the replan; each call is briefed when its drive and reading time come up.
        generate()
        con, col = connect(), build_index()
        print("=== 08:20 brief check (Dr. Okafor's 8:45 call; a memo in his notes tries to steer the model) ===")
        _show(brief_next_call(con, "rep-01", day, to_min("08:20"), col=col))
        print("=== 13:05 Dr. Smith's 2pm cancels: replan ===")
        final = run(con, {"flow": "cancellation", "rep_id": "rep-01", "day": day, "event_id": "ev-04",
                          "now_min": to_min("13:05")}, col=col)
        _show(final)
        print(f"=== Rep approves: booked {approve_plan(con, final)} ===\n")
        for t in ["13:05", "13:30", "14:05", "15:50"]:
            print(f"=== {t} brief check ===")
            _show(brief_next_call(con, "rep-01", day, to_min(t), col=col))
        written = [r[0] for r in con.execute("SELECT event_id FROM call_brief")]
        print(f"Briefs written today: {written} (none for the cancelled ev-04)")
    elif args[0] == "brief":
        _show(brief_next_call(connect(), "rep-01", day, to_min(args[1] if len(args) > 1 else "08:20"),
                              col=build_index()))
    else:
        if args[0] == "cancel":
            req = {"flow": "cancellation", "rep_id": "rep-01", "day": day,
                   "event_id": args[1] if len(args) > 1 else "ev-04", "now_min": to_min(args[2] if len(args) > 2
                                                                                        else "14:00")}
        else:
            req = {"flow": "morning", "rep_id": "rep-01", "day": day}
        _show(run(connect(), req, col=build_index()))
