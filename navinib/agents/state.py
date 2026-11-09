"""The shared state the agents work on (Checkpoint 5.1: one typed state, hub-and-spoke).

Every agent reads the whole state but writes only its own part (OWNS below);
graph.py enforces that. The state holds plain JSON-like data, so it can move to
the planned MCP state server later without changing the agents. Resources that
aren't data (database connection, retrieval index, LLM, drive times) come in
through the graph's config instead.
"""
import operator
from typing import Annotated, TypedDict


def merge(old: dict | None, new: dict | None) -> dict:
    """Reducer for keyed results written in parallel (one brief per provider)."""
    return {**(old or {}), **(new or {})}


class NavinibState(TypedDict, total=False):
    """One run of the multi-agent planner, from the rep's request to the plan awaiting approval."""
    request: dict           # the rep's ask: flow (morning | cancellation), rep_id, day, now_min, event_id, ...
    schedule: dict          # Coordinator: schedule_lookup result (after any cancellation)
    targets: dict           # Targeting: ranked provider locations open today, with reasons, and its refill round
    plan: dict              # Route and Replan: visits (and ToT alternatives), with the round it was made in
    briefs: Annotated[dict, merge]   # Brief: {provider_id: brief}, written in parallel
    compliance: dict        # Compliance: passed, issues, and the round it checked
    round: int              # Coordinator: 0, then +1 for each retry after a compliance fail
    route_round: int        # Coordinator: the round in which the route must be redone
    refill: dict            # Coordinator: the one Route -> Targeting round for an underfilled day
    brief_rounds: dict      # Coordinator: {provider_id: round} briefs that must be rewritten
    feedback: dict          # Coordinator: what Compliance found, passed back to the agent that caused it
    flags: list             # Coordinator: problems still open after the retries, shown to the rep
    next: str               # Coordinator: the agent to run next
    status: str             # Coordinator: working | awaiting_approval
    message: str            # Coordinator: the summary for the rep
    trace: Annotated[list, operator.add]   # every agent appends what it did, for review and the report


# Which part of the state each agent may write (plus 'trace', which everyone appends to).
OWNS = {
    "coordinator": {"schedule", "round", "route_round", "refill", "brief_rounds", "feedback", "flags", "next",
                    "status", "message"},
    "targeting": {"targets"},
    "route_replan": {"plan"},
    "brief": {"briefs"},
    "compliance": {"compliance"},
}
