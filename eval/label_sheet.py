"""Hand labels for the judge-agreement metric (Cohen's kappa).

The "judge" is Navinib's own claim check: for every product-claim sentence in a brief
it decides supported (kept) or unsupported (removed). A person labels the same
sentences against the cited source text, without seeing the check's verdict, and
kappa says how far the two agree.

    python -m eval.label_sheet            # 1. write eval/labels/sheet.json from the real-model briefs
    python -m streamlit run eval/label_app.py   # 2. label each sentence (about 20 minutes)
    python -m eval.label_sheet --score    # 3. kappa, agreement, hand-labeled faithfulness

The briefs come from the same 40 eval briefs as python -m eval.run, so with saved model
answers in .cache/llm/ step 1 costs nothing.
"""
import json
import random
import sys
from pathlib import Path

from navinib import guardrails

LABELS = Path(__file__).resolve().parent / "labels"
SHEET, ANSWERS = LABELS / "sheet.json", LABELS / "answers.json"
SIZE = 40


def _cited(sentence: str, carried: list[str]) -> list[str]:
    return guardrails._CITE.findall(sentence) or carried


def make(size: int = SIZE) -> list[dict]:
    """Every claim sentence from the eval briefs, with the check's verdict and the cited text."""
    from navinib.agents.llm import get_llm
    from navinib.log import setup_logging
    from navinib.rag.retrieval import build_index

    from eval import run as ev
    setup_logging(verbose=False, log_file=ev.TMP / "labels.log")
    llm, col = get_llm(), build_index(ev.TMP / "chroma")
    con = ev.fresh("labels")
    rows = []
    for pid in [r[0] for r in con.execute("SELECT provider_id FROM provider ORDER BY provider_id")]:
        f = ev.brief_for(con, col, llm, pid)
        if not f or not f.get("briefs"):
            continue
        b = f["briefs"][pid]
        sources = b.get("sources", {})
        def show(refs):
            refs = [guardrails.normalize_citations(f"[{r}]", sources)[0][1:-1] for r in refs]
            return {r: sources.get(r, "(source not given to the brief)")[:700] for r in refs}
        for line in b["text"].split("\n"):
            carried: list[str] = []
            for s in [x for x in guardrails._SENTENCE.split(line) if x and x.strip()]:
                carried = _cited(s, carried)
                if guardrails.is_claim(s) and s.strip() != guardrails.UNSUPPORTED_NOTE:
                    rows.append({"provider": pid, "sentence": s.strip(), "sources": show(carried),
                                 "check": "supported", "why": ""})
        for c in b.get("checks", {}).get("claims_removed", []):
            rows.append({"provider": pid, "sentence": c["sentence"], "sources": show(_cited(c["sentence"], [])),
                         "check": "unsupported", "why": c["why"]})
    model = llm.name
    rng = random.Random(7)
    removed = [r for r in rows if r["check"] == "unsupported"]
    kept = [r for r in rows if r["check"] == "supported"]
    # Half and half where possible, so kappa isn't dominated by one class.
    pick = rng.sample(removed, min(len(removed), size // 2))
    pick += rng.sample(kept, min(len(kept), size - len(pick)))
    rng.shuffle(pick)
    for i, r in enumerate(pick, 1):
        r["id"] = f"c{i:02d}"
    LABELS.mkdir(exist_ok=True)
    if ANSWERS.exists() and SHEET.exists():
        # Keep an answer when the same sentence is in the new sheet (and it showed its source text);
        # the old file is kept aside as answers.old.json.
        old_rows = {r["id"]: r for r in json.loads(SHEET.read_text(encoding="utf-8"))["rows"]}
        old = json.loads(ANSWERS.read_text(encoding="utf-8"))
        ANSWERS.replace(LABELS / "answers.old.json")
        by_text = {old_rows[i]["sentence"]: a for i, a in old.items() if i in old_rows and a != "skip"
                   and "(source not given to the brief)" not in old_rows[i]["sources"].values()}
        kept = {r["id"]: by_text[r["sentence"]] for r in pick if r["sentence"] in by_text}
        if kept:
            ANSWERS.write_text(json.dumps(kept, indent=1), encoding="utf-8")
        print(f"Kept {len(kept)} of your {len(old)} earlier answers (same sentence, source shown).")
    SHEET.write_text(json.dumps({"model": model, "rows": pick}, indent=1), encoding="utf-8")
    print(f"{len(pick)} sentences ({len(removed)} removed and {len(kept)} kept in all) from {model} -> {SHEET}")
    print("Next: python -m streamlit run eval/label_app.py")
    return pick


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two binary raters."""
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def score() -> dict | None:
    """Compare the hand labels with the check; also the faithfulness a person sees."""
    if not (SHEET.exists() and ANSWERS.exists()):
        return None
    rows = json.loads(SHEET.read_text(encoding="utf-8"))["rows"]
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    done = [r for r in rows if answers.get(r["id"]) in ("supported", "unsupported")]
    human = [answers[r["id"]] == "supported" for r in done]
    check = [r["check"] == "supported" for r in done]
    out = {"labeled": len(done), "of": len(rows), "kappa": kappa(human, check),
           "agreement": round(sum(h == c for h, c in zip(human, check)) / len(done), 3) if done else None,
           "check_too_strict": [r["id"] for r, h, c in zip(done, human, check) if h and not c],
           "check_too_lenient": [r["id"] for r, h, c in zip(done, human, check) if c and not h]}
    return out


if __name__ == "__main__":
    if "--score" in sys.argv:
        s = score()
        print(json.dumps(s, indent=1) if s else "No labels yet: run python -m streamlit run eval/label_app.py")
    else:
        make()
