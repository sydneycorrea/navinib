# Navinib

**An AI co-pilot for oncology field reps.** Navinib ranks which providers a rep should see, builds a drivable route, writes a short pre-call brief and replans the day when a call cancels. A regional manager view shows goals and coverage.

Capstone project for the CMU Agentic AI program. *Navinib is an app, not a medicine.*

> **Data notice.** Everything in this repo is synthetic or public. Provider names are fictional, "Product X" is a fictional drug, and lab alerts are provider-level counts with small counts suppressed. No company, client or patient data is used.

## Status

Phase 1 (data, tools, retrieval, routing, Tree-of-Thought replan) is done. Phase 2 has started: the five agents from Checkpoint 5.1 run as a LangGraph graph, with a stub LLM until a free model is connected.

| Piece | Status |
| --- | --- |
| Synthetic world generator + SQLite schema | Done |
| Schedule lookup, provider lookup, provider scoring | Done |
| Retrieval for pre-call briefs (Chroma + all-MiniLM-L6-v2) | Done (ported from 3.1) |
| Pre-call brief inputs: notes, publications, label points, do-not-raise list, contacts | Done (LLM write-up in phase 2) |
| Route optimizer (OR-Tools, time windows) + drive-time cache | Done |
| Tree-of-Thought replan (Checkpoint 4.1) | Done |
| Multi-agent graph: Coordinator, Targeting, Route and Replan, Brief, Compliance (LangGraph) | Done, with a stub LLM |
| Free OpenRouter model for Coordinator, Brief and Compliance | Phase 2, next |
| Coordinator reads free-text requests (ReAct over the allowed steps) | Phase 2 |
| Plan validator and guardrails | Phase 3 (Module 6) |
| Streamlit rep app and manager view | Phase 4 |
| Evaluation vs prompt-only baseline | Phase 5 |

## Layout

```
navinib/
  config.py            settings, scoring weights, demo date
  log.py               logging setup (logs/navinib.log, --verbose)
  db.py                SQLite schema (providers, their locations and weekday hours, visits, schedule)
  data/export_csv.py   CSV export for reviewing the data in a spreadsheet
  data/generate.py     seeded synthetic world (rep, 40 providers, visits, demo-day schedule)
                       Dr. Leigh (p-002) works at two locations on different days;
                       lab tech Kevin Tran (c-001) is a non-prescribing contact at L-001
  tools/               specialist tools the planner calls
    schedule.py        schedule lookup, cancel a call
    providers.py       provider lookup, scoring with explanations, location contacts
    brief.py           pre-call brief inputs, each with its source id
    route.py           free windows in the day, OR-Tools route with office-hour windows
    drive.py           drive-time cache (OSRM road times, straight-line fallback)
  agents/              the five agents (Checkpoint 5.1) and the graph that runs them
    state.py           shared state; each agent may write only its own part
    specs.py           each agent's role, goal and tools (CrewAI-style)
    coordinator.py     supervisor: routes work, retries after a compliance fail, summary for the rep
    targeting.py       ranks providers (no LLM)
    route_replan.py    morning plan with OR-Tools, cancellation replan with Tree-of-Thought (no LLM)
    brief.py           one brief per new call, in parallel; "brief unavailable" if retrieval fails
    compliance.py      plan validator: rules first, then an LLM claims check
    llm.py             the model interface and a deterministic stub
    graph.py           LangGraph hub-and-spoke graph; run it from here
  planner/             planning strategies on top of the tools
    tot.py             Tree-of-Thought replan: compares alternative plans for a freed window
  rag/                 retrieval for pre-call briefs
    corpus.py          fictional Product X label, trials, call notes, provider publications
    retrieval.py       label lookup, trial search, call-note recall, publication search
tests/                 pytest suite
```

## Run it

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m navinib.data.generate  # writes data/navinib.db
python -m navinib.data.export_csv  # optional: CSVs for review in data/csv/
python -m navinib.rag.retrieval  # Checkpoint 3.1 retrieval demo
python -m navinib.tools.brief p-001  # pre-call brief for Dr. Lee
python -m navinib.tools.drive    # optional, needs internet: cache real road times from OSRM
python -m navinib.tools.route    # the 2.1 replan: 2pm cancels, who fits?
python -m navinib.planner.tot    # the 4.1 Tree-of-Thought replan: compare plans, pick one, say why
python -m navinib.planner.tot 12:00  # same, but the rep hears of the cancellation at noon
python -m navinib.agents.graph morning             # the five agents plan the demo day
python -m navinib.agents.graph cancel ev-04 14:00  # the agents handle Dr. Smith's 2pm cancellation
pytest                           # runs the tests
```

The first retrieval run downloads the free all-MiniLM-L6-v2 embedding model (about 80 MB).

## Logs

Every tool logs what it was asked and what it returned. Each run appends to `logs/navinib.log` (full detail, never committed). The terminal shows only warnings unless you add `--verbose`:

```bash
python -m navinib.tools.route --verbose
```

A run of the replan demo logs the cancellation, the open window, the provider lookup and ranking, the drive-time source, and each planned or dropped visit with its reason.

## Demo day

The scripted demo day (`2026-10-27`) follows the Checkpoint 2.1 example: the rep's 2:00 call with Dr. Helen Smith cancels. Provider scoring ranks Hillcrest Cancer Center first, but the route optimizer finds it closes at 2:30 before a 30-minute visit can finish, so the next candidate, Dr. Alicia Lee, fills the slot and gets a brief from the retrieval layer.

The Tree-of-Thought replan (`python -m navinib.planner.tot`, Checkpoint 4.1) looks past the first fill. It grows a small tree of alternative plans for the 2:00 to 3:30 gap (in-person visits nearby, plus at most one phone call to a provider no visit can reach, such as Hillcrest before it closes), rules out branches that break office hours or the window, ranks the rest with a fixed value function, and keeps the best few of each shape at every level. It chooses two visits, Dr. Lee 2:10 to 2:40 and Dr. Irene Marchetti 2:57 to 3:27 (value 1.39), over phoning Hillcrest and then visiting Dr. Lee (1.20) or Dr. Lee alone (0.73), and prints why.

Drive times use a straight-line estimate until you run `python -m navinib.tools.drive` once with internet access; that saves real road times from the free OSRM service to `data/drive_times.json`, which you can commit so the demo works offline.

## Agents

The multi-agent design from Checkpoint 5.1 runs in `navinib/agents/`. The Coordinator is the hub: every agent reports back to it and it decides what runs next, so no agent calls another.

- **Morning plan:** Coordinator, Targeting, Route and Replan (OR-Tools), Brief (one per new call, in parallel), Compliance, Coordinator.
- **Cancellation:** the Coordinator cancels the call and skips Targeting; Route and Replan runs the Tree-of-Thought replan; Brief writes only the new briefs; Compliance checks them.
- **Two-way links (the only ones):**
  - Route to Targeting, at most once: if the morning route drops providers that don't fit (office hours or drive time) and leaves room in the day, Targeting re-ranks without them and Route runs again.
  - Compliance to the agent that caused a problem, at most twice. Anything still wrong is flagged to the rep, never silently fixed.

  Every other handoff is one-way, and its output is still checked by Compliance and approved by the rep.
- **Shared state:** one typed state (`agents/state.py`); each agent may write only its own part, and the graph refuses anything else.
- **LLM:** a deterministic stub stands in for the model, so the graph and tests run offline. Only the Coordinator (summary), Brief (writing) and Compliance (claims check) use it.

## License

MIT
