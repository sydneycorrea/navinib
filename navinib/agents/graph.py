"""The multi-agent graph (Checkpoint 5.1): a LangGraph supervisor with the Coordinator at the hub.

Every agent is a node that returns to the Coordinator; no agent calls another.
Briefs fan out in parallel (one per call) and join back at the Coordinator.

Run:
    python -m navinib.agents.graph morning               # plan the demo day
    python -m navinib.agents.graph cancel ev-04 14:00    # Dr. Smith's 2pm cancels on arrival
"""
import sqlite3

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from navinib.agents.brief import brief
from navinib.agents.compliance import compliance
from navinib.agents.coordinator import briefs_to_write, coordinator, next_step
from navinib.agents.llm import LLM, get_llm
from navinib.agents.route_replan import route_replan
from navinib.agents.specs import AGENTS
from navinib.agents.state import NavinibState
from navinib.agents.targeting import targeting
from navinib.log import get_logger
from navinib.tools.drive import DriveTimes

log = get_logger("navinib.agents.graph")

NODES = {"coordinator": coordinator, "targeting": targeting, "route_replan": route_replan, "brief": brief,
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
    """After the Coordinator: one Brief run per call that needs one, or the single next agent, or the end."""
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
    g.add_conditional_edges("coordinator", _route, ["targeting", "route_replan", "brief", "compliance", END])
    for name in NODES:
        if name != "coordinator":
            g.add_edge(name, "coordinator")
    return g.compile()


def run(con: sqlite3.Connection, request: dict, col=None, llm: LLM | None = None,
        drive: DriveTimes | None = None) -> dict:
    """Run one request through the agents and return the final shared state.

    request: {'flow': 'morning' | 'cancellation', 'rep_id', 'day', and for a cancellation
    'event_id' and 'now_min' (plus optional 'at_location_id')}. col is the retrieval index;
    without it every brief is a partial one from structured data and the plan still goes through.
    """
    config = {"configurable": {"con": con, "col": col, "llm": llm or get_llm(), "drive": drive or DriveTimes()},
              "recursion_limit": 50}
    log.info("run %s", request)
    return build_graph().invoke({"request": request}, config=config)


if __name__ == "__main__":
    import sys

    from navinib import config as cfg
    from navinib.db import connect
    from navinib.log import setup_logging
    from navinib.rag.retrieval import build_index

    setup_logging()
    args = sys.argv[1:] or ["morning"]
    if args[0] == "cancel":
        h, m = (args[2] if len(args) > 2 else "14:00").split(":")
        req = {"flow": "cancellation", "rep_id": "rep-01", "day": cfg.DEMO_DATE,
               "event_id": args[1] if len(args) > 1 else "ev-04", "now_min": int(h) * 60 + int(m)}
    else:
        req = {"flow": "morning", "rep_id": "rep-01", "day": cfg.DEMO_DATE}
    final = run(connect(), req, col=build_index())
    print("Trace:")
    for line in final["trace"]:
        print(f"  {line}")
    print("\nBriefs:")
    for v in final["plan"]["visits"]:
        b = final["briefs"].get(v["provider_id"], {})
        print(f"--- {v['start']} {v['full_name']} [{b.get('status')}]\n{b.get('text', '')}\n")
    print(f"Status: {final['status']}\n{final['message']}")
