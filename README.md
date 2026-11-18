# Navinib

**An AI co-pilot for oncology field reps.** Navinib ranks which providers a rep should see, builds a drivable route, writes a short pre-call brief and replans the day when a call cancels. A regional manager view shows goals and coverage.

Capstone project for the CMU Agentic AI program. *Navinib is an app, not a medicine.*

> **Data notice.** Everything in this repo is synthetic or public. Provider names are fictional, "Product X" is a fictional drug, and lab alerts are provider-level counts with small counts suppressed. No company, client or patient data is used.

## Status

Phase 1 (data, tools, retrieval, routing, Tree-of-Thought replan) is done. The five agents from Checkpoint 5.1 run as a LangGraph graph, and the Checkpoint 6.1 guardrails are built. A free OpenRouter model is used when `OPENROUTER_API_KEY` is set; without it, a deterministic stub runs everything offline.

| Piece | Status |
| --- | --- |
| Synthetic world generator + SQLite schema | Done |
| Schedule lookup, provider lookup, provider scoring | Done |
| Retrieval for pre-call briefs (Chroma + all-MiniLM-L6-v2) | Done (ported from 3.1) |
| Pre-call brief inputs: notes, publications, label points, do-not-raise list, contacts | Done (LLM write-up in phase 2) |
| Route optimizer (OR-Tools, time windows) + drive-time cache | Done |
| Tree-of-Thought replan (Checkpoint 4.1) | Done |
| Multi-agent graph: Coordinator, Targeting, Routing, Brief, Compliance (LangGraph) | Done, with a stub LLM |
| Free OpenRouter model for Coordinator, Brief and Compliance | Client done; first live run pending |
| Guardrails (Checkpoint 6.1): policy file, input checks, claim-by-claim check, escalation, four lanes, JSON run log, kill switches | Done |
| Streamlit rep app and manager view | Phase 4 |
| Evaluation vs prompt-only baseline | Phase 5 |

## Layout

```
navinib/
  config.py            settings, scoring weights, demo date; reads the limits in policy.yaml
  policy.yaml          guardrail policy owned by compliance: limits, model settings, kill switches, check patterns
  guardrails.py        input checks (PII, injection, adverse events, off-label), claim-by-claim check, escalation, lanes
  audit.py             one JSON line per run in logs/runs.jsonl, with a trace id
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
    routing.py         morning plan with OR-Tools, cancellation replan with Tree-of-Thought (no LLM)
    brief.py           the brief for the next due call (just in time); a partial brief from structured data if retrieval fails
    compliance.py      plan validator: rules first, then an LLM claims check
    llm.py             the model interface: free OpenRouter model, deterministic stub, kill switch
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
python -m navinib.agents.graph demo                # the whole day, including the guardrails firing
pytest                           # runs the tests
```

To use a real model, get a free key at [openrouter.ai/keys](https://openrouter.ai/keys) and set `OPENROUTER_API_KEY` as a Codespaces secret, or copy `.env.example` to `.env`. Only free models are allowed (ids ending in `:free`); pick one with `NAVINIB_MODEL` or change the default in `navinib/policy.yaml`. Tests always use the stub.

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

- **Morning plan:** Coordinator, Targeting (territory mode), Routing (OR-Tools), Compliance (providers and times), then the rep approves. No briefs yet.
- **Cancellation:** the Coordinator cancels the call; Targeting (nearby mode) ranks only providers within a 20-minute drive of where the rep will be when the freed window opens, plus phone candidates for top providers no visit can reach; Routing runs the Tree-of-Thought replan over that shortlist; Compliance checks it; the rep approves.
- **Briefs, just in time:** `brief_next_call(con, rep_id, day, now_min)` runs on a timer or when the rep opens the app. When the next booked call's brief is due, Brief writes that call's brief only and Compliance checks its claims. The timing is set per call: due = call start - drive there (from the last booked call, or home) - reading time (`BRIEF_READ_MIN`, 15 min), written `BRIEF_GEN_MARGIN_MIN` (5 min) before that, or right away if that time has passed. A call 20 minutes away after a replan is briefed as soon as the rep approves; a call 45 minutes away is briefed earlier than one next door. `PREP_BUFFER_MIN` stays a planning rule, not the brief timer. A brief is saved once per call, and a cancelled call never gets one, so no tokens go to calls that cancel or move.
- **Two-way links (the only ones):**
  - Routing to Targeting, at most once: if the morning route drops providers that don't fit (office hours or drive time) and leaves room in the day, Targeting re-ranks without them and Routing runs again.
  - Compliance to the agent that caused a problem, at most twice. Anything still wrong is flagged to the rep, never silently fixed.

  Every other handoff is one-way, and its output is still checked by Compliance and approved by the rep.
- **When something fails:**
  - Retrieval or the model down: Brief writes a partial brief from structured data (why the provider ranked high, goal gap, lab alerts, recent calls, who is at the office). It is marked "label and trial sources unavailable" and makes no product claims.
  - No live road times: Routing uses cached or straight-line times, and the plan says they are estimates.
  - Model down: the rep still gets the ranked plan, with a summary built in code instead of written by the model. If the claims check can't run, the rep sees a warning.
- **Shared state:** one typed state (`agents/state.py`); each agent may write only its own part, and the graph refuses anything else.
- **LLM:** a free OpenRouter model (temperature 0.2, token cap from the policy) when a key is set, otherwise a deterministic stub, so the graph and tests run offline. Only the Coordinator (plan summary), Brief (writing), Compliance (claims check) and the adverse-event classifier use it.

## Guardrails

The safety plan from Checkpoint 6.1. Hard limits live in `navinib/policy.yaml` and in code, not in prompts: a model can be talked out of a prompt rule but cannot use a tool it was never given.

| Stage | What runs | Where |
| --- | --- | --- |
| Configuration | No outbound tools (nothing emails, texts or calls an office or writes to a CRM); the only write is `approve_plan` after the rep approves. Each agent writes only its own state. Retry, refill, phone-call, radius and lab-alert limits; free models only, temperature 0.2, token and step caps | `policy.yaml`, `agents/specs.py`, `agents/graph.py` |
| Rep message | Unknown request: a clarifying question. Possible adverse event: stop, log for safety reporting. Off-label question: stop, refer to Medical Information | `agents/coordinator.py` |
| Call notes and retrieved text | Personal data redacted; instruction-like text quarantined for admin review; a possible adverse event escalated once and left out of the brief; label passages from a superseded version dropped | `agents/brief.py`, `guardrails.py` |
| Brief output | Claim by claim: a dose, indication or efficacy sentence must cite the label or an approved trial, and its numbers must appear in that source; otherwise it is removed and marked. Then Compliance re-checks citations, do-not-raise items, label version and claims, and runs the LLM claims check | `guardrails.claim_check`, `agents/compliance.py` |
| Human review | Every output gets a lane: **Auto, logged** (ranking, route drafts, a clean brief), **Rep approves** (every calendar change), **Flag to rep** (failures after retries, partial briefs, estimated drive times, close replan options), **Stop and escalate** (adverse events, off-label questions, quarantined documents) | `state["lanes"]` |
| Monitoring | One JSON line per run in `logs/runs.jsonl`: trace id, model, policy and label versions, prompt hashes, checks, lanes, escalations, trace. Kill switches: `agents.brief: false` gives plan-only mode; `agents.llm: false` makes every agent use its coded fallback | `audit.py`, `policy.yaml` |

The demo data includes two notes that trip these checks: Dr. Lee's office reports a patient hospitalized with a rash (an adverse event that also names the patient), and a memo in Dr. Okafor's notes tries to steer the model toward first-line use (injection). `python -m navinib.agents.graph demo` shows both being handled.

## License

MIT
