"""The language model the agents use, behind one small interface.

Three agents use an LLM: the Coordinator (summary for the rep), Brief (writes the
brief) and Compliance (checks claims). They all call llm.complete(task, system,
user, data). A real model reads `system` and `user`; the stub ignores them and
builds a deterministic answer from `data`, so the graph and its tests run with
no model, no key and no network.

get_llm() picks the model: a free OpenRouter model when OPENROUTER_API_KEY is set
(environment, Codespaces secret or .env), the stub otherwise, and a disabled model
when the policy's kill switch turns the LLM off (every agent then uses its coded
fallback). Model settings (free models only, temperature 0.2, token cap, timeout)
come from navinib/policy.yaml.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from navinib.config import POLICY, ROOT


class LLM(Protocol):
    """Anything with a name and a complete() method can be the agents' model."""
    name: str

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """Answer one prompt. `task` is 'brief', 'claims', 'summary' or 'ae'."""
        ...


class StubLLM:
    """Deterministic stand-in for a model: same input, same output, no network."""
    name = "stub"

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """Route to the stub answer for each task."""
        return {"brief": self._brief, "claims": self._claims, "summary": self._summary,
                "ae": self._ae}[task](data or {})

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
        """The stub's summary is the plain one."""
        return plain_summary(d)

    def _ae(self, d: dict) -> str:
        """Called only when no adverse-event keyword matched; the stub agrees with the keywords."""
        return "NO"


class DisabledLLM:
    """The kill switch: every call fails, so each agent falls back to what it can do in code."""
    name = "disabled"

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """Refuse every call."""
        raise RuntimeError("LLM turned off by the kill switch in policy.yaml")


class OpenRouterLLM:
    """A free model on OpenRouter, through its OpenAI-compatible chat API. Standard library only."""
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str | None = None):
        """Use the policy's settings; refuse any model that isn't free."""
        m = POLICY["model"]
        self.model = model or m["default"]
        if m["free_only"] and not self.model.endswith(":free"):
            raise ValueError(f"{self.model} is not a free model; policy.yaml allows free models only")
        self.name = f"openrouter:{self.model}"
        self.api_key = api_key
        self.temperature, self.max_tokens, self.timeout = m["temperature"], m["max_tokens"], m["timeout_s"]

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """One chat completion. Retrieved text arrives inside `user` as quoted JSON data, never as instructions."""
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
            "X-Title": "Navinib capstone"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenRouter {e.code}: {e.read().decode()[:200]}") from e
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content")
        if not text:
            raise RuntimeError(f"OpenRouter returned no text: {str(out)[:200]}")
        return text.strip()


def plain_summary(d: dict) -> str:
    """One plain paragraph for the rep, built in code: what's planned and what needs a look.

    Also the fallback when the model is down, so the rep always gets the plan.
    """
    visits = d["visits"]
    what = "; ".join(f"{v['start']} {'visit' if v['call_type'] == 'face_to_face' else 'call'} "
                     f"{v['full_name']}" for v in visits) or "no new calls"
    text = f"Proposed: {what}."
    if d.get("reason"):
        text += f" Why: {d['reason']}."
    if d.get("drive_estimated"):
        text += " Drive times are estimates (live road times unavailable)."
    if d.get("partial"):
        text += (f" Partial brief for {', '.join(d['partial'])}: label and trial sources unavailable, "
                 "so no product claims.")
    for w in d.get("warnings", []):
        text += f" Note: {w}."
    if d.get("flags"):
        text += f" Needs your review: {len(d['flags'])} compliance issue(s) not fixed automatically."
    return text + " Approve to book."


CITATION = re.compile(r"\[([^\]]+)\]")


def citations(text: str) -> list[str]:
    """Source ids cited in a brief, e.g. ['N-0388', 'Label §5.1']."""
    return CITATION.findall(text)


def _dotenv() -> dict:
    """Read KEY=value lines from .env at the repo root, if there is one (never committed)."""
    path = ROOT / ".env"
    if not path.exists():
        return {}
    pairs = (l.split("=", 1) for l in path.read_text(encoding="utf-8").splitlines() if "=" in l
             and not l.lstrip().startswith("#"))
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def get_llm() -> LLM:
    """The model to use: disabled by the kill switch, OpenRouter when a key is set, else the stub."""
    if not POLICY["agents"].get("llm", True):
        return DisabledLLM()
    env = {**_dotenv(), **os.environ}
    key = env.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return OpenRouterLLM(key, env.get("NAVINIB_MODEL") or None)
    return StubLLM()
