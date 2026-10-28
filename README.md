# Navinib

**An AI co-pilot for oncology field reps.** Navinib ranks which providers a rep should see, builds a drivable route, writes a short pre-call brief and replans the day when a call cancels. A regional manager view shows goals and coverage.

Capstone project for the CMU Agentic AI program. *Navinib is an app, not a medicine.*

> **Data notice.** Everything in this repo is synthetic or public. Provider names are fictional, "Product X" is a fictional drug, and lab alerts are provider-level counts with small counts suppressed. No company, client or patient data is used.

## Status

Phase 1 of the MVP plan (Oct 27 to Nov 2): synthetic data, SQLite, the first tools and the retrieval layer from Checkpoint 3.1.

| Piece | Status |
| --- | --- |
| Synthetic world generator + SQLite schema | Done |
| Schedule lookup, provider lookup, provider scoring | Done |
| Retrieval for pre-call briefs (Chroma + all-MiniLM-L6-v2) | Done (ported from 3.1) |
| Pre-call brief inputs: notes, publications, label points, do-not-raise list, contacts | Done (LLM write-up in phase 2) |
| Route optimizer (OR-Tools, time windows) + drive-time cache | Done |
| Tree-of-Thought replan | Phase 1 |
| ReAct planner (LangGraph + free OpenRouter model) | Phase 2 |
| Multi-agent split | Phase 2 (Module 5) |
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

Drive times use a straight-line estimate until you run `python -m navinib.tools.drive` once with internet access; that saves real road times from the free OSRM service to `data/drive_times.json`, which you can commit so the demo works offline.

## License

MIT
