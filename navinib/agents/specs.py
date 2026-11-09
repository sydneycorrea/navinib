"""The five agents, defined CrewAI-style (role, goal, tools) and run as LangGraph nodes.

These specs are the single description of each agent: the graph checks that an
agent writes only what it owns, and the README and report quote them.
"""
from dataclasses import dataclass

from navinib.agents.state import OWNS


@dataclass(frozen=True)
class AgentSpec:
    """Who an agent is, what it is for, which tools it may use, and whether it calls the LLM."""
    name: str
    role: str
    goal: str
    tools: tuple[str, ...]
    uses_llm: bool

    @property
    def writes(self) -> set[str]:
        """The parts of the shared state this agent may write."""
        return OWNS[self.name]


AGENTS = {a.name: a for a in [
    AgentSpec("coordinator", "Coordinator",
              "Understand the rep's request, route work to the specialists, merge their results and ask the "
              "rep to approve the plan.", ("schedule_lookup", "cancel_event"), uses_llm=True),
    AgentSpec("targeting", "Targeting",
              "Rank the providers worth seeing, with plain reasons for each rank: the whole territory for the "
              "morning plan, or only those near the rep (plus phone fallbacks) when a call cancels.",
              ("provider_lookup", "provider_scoring", "candidates"), uses_llm=False),
    AgentSpec("routing", "Routing",
              "Build a drivable plan that fits office hours: OR-Tools for the morning plan, the Tree-of-Thought "
              "replan when a call cancels.", ("free_windows", "plan_route", "tree_of_thought"), uses_llm=False),
    AgentSpec("brief", "Brief",
              "Write a short pre-call brief for the next due call, using only retrieved, cited sources.",
              ("build_brief", "label_lookup", "call_note_recall", "trial_search"), uses_llm=True),
    AgentSpec("compliance", "Compliance",
              "Check the plan and every brief before the rep sees them: rules first, then the LLM for claims.",
              ("plan_validator",), uses_llm=True),
]}
