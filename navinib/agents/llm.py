"""The language model the agents use, behind one small interface.

Three agents use an LLM: the Coordinator (summary for the rep), Brief (writes the
brief) and Compliance (checks claims). They all call llm.complete(task, system,
user, data). A real model reads `system` and `user`; the stub ignores them and
builds a deterministic answer from `data`, so the graph and its tests run with
no model, no key and no network. A free OpenRouter model plugs in here next.
"""
import re
from typing import Protocol


class LLM(Protocol):
    """Anything with a name and a complete() method can be the agents' model."""
    name: str

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """Answer one prompt. `task` is 'brief', 'claims' or 'summary'."""
        ...


class StubLLM:
    """Deterministic stand-in for a model: same input, same output, no network."""
    name = "stub"

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """Route to the stub answer for each task."""
        return {"brief": self._brief, "claims": self._claims, "summary": self._summary}[task](data or {})

    def _brief(self, d: dict) -> str:
        """A short brief built only from the inputs, citing a source id for each point."""
        b, v = d["inputs"], d["visit"]
        how = "Visit" if v["call_type"] == "face_to_face" else "Phone call with"
        lines = [f"{how} {b['full_name']} (tier {b['tier']}) at {v['start']}."]
        for n in b["call_notes"][:1]:
            lines.append(f"Last call [{n['ref']}]: {n['text'].split(': ', 1)[-1][:160]}")
        for t in b["talking_points"][:2]:
            lines.append(f"Talking point [{t['ref']}]: {t['text'][:160]}")
        pubs = [p for p in b["publications"] if p["may_discuss"]]
        if pubs:
            lines.append(f"Their work [{pubs[0]['pub_id']}]: {pubs[0]['title']}.")
        lines.append(f"Open lab alerts: {b['lab_alerts']}.")
        lines += [f"Also at the office: {c['full_name']} ({c['role'].replace('_', ' ')})." for c in b["contacts"]]
        if b["do_not_raise"]:
            lines.append("Some trial and publication topics are off-label: if asked, refer to Medical Information.")
        return "\n".join(lines)

    def _claims(self, d: dict) -> str:
        """The stub trusts the rule checks and finds no unsupported claims."""
        return "PASS"

    def _summary(self, d: dict) -> str:
        """One plain paragraph for the rep: what's planned, what needs a look."""
        visits = d["visits"]
        what = "; ".join(f"{v['start']} {'visit' if v['call_type'] == 'face_to_face' else 'call'} "
                         f"{v['full_name']}" for v in visits) or "no new calls"
        text = f"Proposed: {what}."
        if d.get("reason"):
            text += f" Why: {d['reason']}."
        if d.get("unavailable"):
            text += f" Brief unavailable for {', '.join(d['unavailable'])}."
        if d.get("flags"):
            text += f" Needs your review: {len(d['flags'])} compliance issue(s) not fixed automatically."
        return text + " Approve to book."


CITATION = re.compile(r"\[([^\]]+)\]")


def citations(text: str) -> list[str]:
    """Source ids cited in a brief, e.g. ['N-0388', 'Label §5.1']."""
    return CITATION.findall(text)


def get_llm() -> LLM:
    """The model to use. Only the stub for now; the OpenRouter client is the next step."""
    return StubLLM()
