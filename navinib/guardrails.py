"""Safety guardrails from Checkpoint 6.1, driven by navinib/policy.yaml.

Code decides limits and routing; the LLM only reads and writes text. Every check
here is plain Python, so it runs before any model call, costs no tokens and is
tested in tests/test_guardrails.py.

  Input (pre-generation):  screen() on every rep message, call note and retrieved chunk:
                           PII redaction, injection scan, adverse-event check
                           off_label_question() on rep messages
  Sources:                 current_label() blocks passages from a superseded label version
  Output (post-generation): claim_check() on every brief, sentence by sentence (QAG style)
  Escalation:              escalate() records what left Navinib and where it went
  Human review:            lanes() sends each output to one of four lanes by the cost of a mistake
"""
import re
import sqlite3
from datetime import datetime, timezone

from navinib.config import POLICY
from navinib.log import get_logger

log = get_logger("navinib.guardrails")

_I = re.IGNORECASE
AE_PATTERNS = [re.compile(p, _I) for p in POLICY["adverse_event"]["keywords"]]
OFF_LABEL_PATTERNS = [re.compile(p, _I) for p in POLICY["off_label"]["patterns"]]
INJECTION_PATTERNS = [re.compile(p, _I) for p in POLICY["injection"]["patterns"]]
PII_PATTERNS = {k: re.compile(p) for k, p in POLICY["pii"]["patterns"].items()}
CLAIM_PATTERNS = [re.compile(p, _I) for p in POLICY["claims"]["claim_words"]]
APPROVED_PREFIXES = tuple(POLICY["claims"]["approved_ref_prefixes"])
UNSUPPORTED_NOTE = POLICY["claims"]["unsupported_note"]
OFF_LABEL_NOTE = POLICY["claims"]["off_label_note"]

AE_SYSTEM = """You screen notes from a pharmaceutical field rep for possible adverse events: any harm,
side effect, hospital visit or death a patient had while on a medicine. Reply with exactly YES or NO."""

LANES = {
    "auto": "Auto, logged",
    "rep_approves": "Rep approves",
    "flag": "Flag to rep",
    "escalate": "Stop and escalate",
}


# ---------- input checks ----------

def redact(text: str) -> tuple[str, list[str]]:
    """Replace patient names, record numbers, phone numbers, emails and dates of birth with [REDACTED kind]."""
    kinds = []
    for kind, pat in PII_PATTERNS.items():
        if pat.search(text):
            kinds.append(kind)
            text = pat.sub(f"[REDACTED {kind}]", text)
    return text, kinds


def injection(text: str) -> list[str]:
    """Instruction-like phrases in data (call notes, retrieved documents). Any hit quarantines the text."""
    return [m.group(0) for p in INJECTION_PATTERNS if (m := p.search(text))]


def adverse_event(text: str, llm=None) -> list[str]:
    """Possible adverse event: keywords first (cheap, high recall), then the LLM for what keywords miss.

    Returns the reasons it fired, empty if it didn't. If the model is down, the keyword
    result stands on its own.
    """
    hits = [m.group(0) for p in AE_PATTERNS if (m := p.search(text))]
    if hits or llm is None:
        return hits
    try:
        answer = llm.complete("ae", AE_SYSTEM, text, data={"text": text})
    except Exception as e:  # model down: keywords only, and say so in the log
        log.warning("adverse-event classifier unavailable, keywords only: %s", e)
        return []
    return ["model classifier"] if answer.strip().upper().startswith("YES") else []


def off_label_question(text: str) -> list[str]:
    """A rep asking about an unapproved use: Navinib does not answer, Medical Information does."""
    return [m.group(0) for p in OFF_LABEL_PATTERNS if (m := p.search(text))]


def screen(text: str, llm=None) -> dict:
    """All input checks on one piece of text. The redacted text is the only version passed on."""
    clean, pii = redact(text)
    return {"text": clean, "pii": pii, "injection": injection(text), "adverse_event": adverse_event(clean, llm)}


# ---------- source checks ----------

def current_label() -> str:
    """The only label version briefs may be built on."""
    return POLICY["label"]["current_version"]


def superseded(passages: list[dict]) -> list[dict]:
    """Label passages from any version other than the current one."""
    return [p for p in passages if p.get("version") and p["version"] != current_label()]


# ---------- output check: claim by claim ----------

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])|\n+")
_CITE = re.compile(r"\[([^\]]+)\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def is_claim(sentence: str) -> bool:
    """A statement about dose, indication or efficacy, which must be backed by the label."""
    bare = _CITE.sub("", sentence)
    return any(p.search(bare) for p in CLAIM_PATTERNS)


def claim_check(text: str, sources: dict[str, str]) -> tuple[str, list[dict]]:
    """Check every claim sentence against the sources it cites; remove the ones that don't hold.

    A claim sentence is supported when it cites at least one approved source (label or
    approved trial) and every number in it appears in one of the cited approved texts.
    Sentences that cite only call notes or publications report history, not product
    claims, and pass. A sentence with no citation of its own is covered by the last
    citation before it on the same line ("Talking point [Label §5.1]: ... . ..."). Anything
    removed is replaced by a visible note, never silently.
    """
    kept, removed = [], []
    for line in text.split("\n"):
        out, carried = [], []
        for s in [x for x in _SENTENCE.split(line) if x and x.strip()]:
            refs = [r for r in _CITE.findall(s) if r in sources or r.startswith(APPROVED_PREFIXES)] \
                or _CITE.findall(s) or carried
            carried = refs
            if not is_claim(s):
                out.append(_mark_off_label(s, sources, refs))
                continue
            approved = [r for r in refs if r.startswith(APPROVED_PREFIXES)]
            history_only = refs and not approved and all(r in sources for r in refs)
            if history_only and not off_label_question(s):
                out.append(s)
                continue
            cited_text = " ".join(sources.get(r, "") for r in approved)
            numbers = _NUMBER.findall(_CITE.sub("", s))
            missing = [n for n in numbers if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", cited_text)]
            why = ("cites no approved source" if not approved else
                   f"cited source not provided ({', '.join(r for r in approved if r not in sources)})"
                   if any(r not in sources for r in approved) else
                   f"numbers not in the cited source: {', '.join(missing)}" if missing else
                   "off-label wording" if any(not re.search(re.escape(w), cited_text, re.IGNORECASE)
                                              for w in off_label_question(s)) else None)
            if why:
                removed.append({"sentence": s.strip(), "why": why})
                out.append(UNSUPPORTED_NOTE)
            else:
                out.append(s)
        kept.append(" ".join(out))
    return "\n".join(l for l in kept if l.strip()), removed


def _mark_off_label(sentence: str, sources: dict, refs: list) -> str:
    """History that touches an unapproved use (a doctor asked about first-line data) stays, with the referral.

    Found by the evaluation: a call note quoted in a brief can mention an off-label topic
    without making a claim, so the rep needs the referral right there, not only at the end.
    """
    words = off_label_question(sentence)
    cited = " ".join(sources.get(r, "") for r in refs if r.startswith("Label"))
    if not words or "Medical Information" in sentence or \
            all(re.search(re.escape(w), cited, re.IGNORECASE) for w in words):
        return sentence
    return f"{sentence} {OFF_LABEL_NOTE}"


# ---------- escalation and the four lanes ----------

def escalate(con: sqlite3.Connection, kind: str, source_ref: str, excerpt: str, rep_id: str) -> dict:
    """Record a case that leaves Navinib: an adverse event, an off-label question or a quarantined document.

    The excerpt is redacted before it is stored. Each source is escalated once, so a
    note that was already reported is not reported again on the next brief.
    """
    con.execute("""CREATE TABLE IF NOT EXISTS escalation (
                       source_ref TEXT NOT NULL, kind TEXT NOT NULL, rep_id TEXT NOT NULL,
                       routed_to TEXT NOT NULL, excerpt TEXT NOT NULL, created_at TEXT NOT NULL,
                       PRIMARY KEY (source_ref, kind))""")
    routed_to = {"adverse_event": POLICY["adverse_event"]["route_to"],
                 "off_label": POLICY["off_label"]["route_to"],
                 "injection": "admin review (document quarantined)"}[kind]
    excerpt = redact(excerpt)[0][:240]
    new = con.execute("INSERT OR IGNORE INTO escalation VALUES (?, ?, ?, ?, ?, ?)",
                      (source_ref, kind, rep_id, routed_to, excerpt,
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))).rowcount == 1
    con.commit()
    log.warning("escalate %s %s -> %s%s", kind, source_ref, routed_to, "" if new else " (already reported)")
    return {"kind": kind, "source_ref": source_ref, "routed_to": routed_to, "new": new}


def quarantined(con: sqlite3.Connection) -> set[str]:
    """Documents held back from retrieval until an admin reviews them."""
    try:
        return {r[0] for r in con.execute("SELECT source_ref FROM escalation WHERE kind = 'injection'")}
    except sqlite3.OperationalError:  # no escalation yet
        return set()


def close_options(plan: dict) -> bool:
    """The two best replan options are within close_options_pct of each other: the rep should choose."""
    alts = plan.get("alternatives") or []
    best = plan.get("value")
    if not alts or not best:
        return False
    return abs(best - alts[0]["value"]) <= abs(best) * POLICY["limits"]["close_options_pct"] / 100


def lanes(items: list[tuple[str, str, str]]) -> list[dict]:
    """Label each output with its lane: (item, lane key, why) -> dicts the app and the log can show."""
    return [{"item": item, "lane": lane, "lane_name": LANES[lane], "why": why} for item, lane, why in items]
