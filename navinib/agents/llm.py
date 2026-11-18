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
    """A free model on OpenRouter, through its OpenAI-compatible chat API. Standard library only.

    OpenRouter's free models come and go, so the client keeps a list: NAVINIB_MODEL if set,
    then the policy's preferred models, then whatever OpenRouter currently lists as free.
    A model that is gone, no longer free or rate-limited is skipped for the rest of the run.
    Paid models are never used.
    """
    URL = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    # Retired, not free, refused for this client (403, e.g. "only available on agentic harnesses"),
    # rate-limited, timed out or overloaded: try the next model. 401 (bad key) is not skipped.
    SKIP = (400, 402, 403, 404, 408, 409, 429, 500, 502, 503, 504)

    def __init__(self, api_key: str, model: str | None = None):
        """Use the policy's settings; refuse any model that isn't free."""
        m = POLICY["model"]
        self.queue = [model] if model else list(m["preferred"])
        bad = [q for q in self.queue if m["free_only"] and not q.endswith(":free")]
        if bad:
            raise ValueError(f"{bad[0]} is not a free model; policy.yaml allows free models only")
        self.api_key = api_key
        self.temperature, self.max_tokens, self.timeout = m["temperature"], m["max_tokens"], m["timeout_s"]
        self.tried: list[str] = []
        self.discovered = False

    @property
    def model(self) -> str:
        """The model in use now."""
        return self.queue[0] if self.queue else "none"

    @property
    def name(self) -> str:
        return f"openrouter:{self.model}"

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """One chat completion. Retrieved text arrives inside `user` as quoted JSON data, never as instructions."""
        errors = []
        while True:
            if not self.queue and not self.discovered:
                self.discovered = True
                self.queue = [m for m in free_models(self.api_key) if m not in self.tried]
            if not self.queue:
                raise RuntimeError("no free OpenRouter model answered: " + "; ".join(errors[-3:]))
            try:
                return self._call(self.queue[0], system, user)
            except _Skip as e:
                errors.append(str(e))
                self.tried.append(self.queue.pop(0))

    def _call(self, model: str, system: str, user: str) -> str:
        body = {"model": model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
            "X-Title": "Navinib capstone"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            msg = f"{model} -> {e.code}: {e.read().decode()[:160]}"
            if e.code in self.SKIP:
                raise _Skip(msg) from e
            raise RuntimeError(f"OpenRouter {msg}") from e
        except (TimeoutError, json.JSONDecodeError) as e:
            raise _Skip(f"{model} -> {type(e).__name__}") from e
        if out.get("error"):
            raise _Skip(f"{model} -> {str(out['error'])[:160]}")
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content")
        if not text:
            raise _Skip(f"{model} returned no text")
        return text.strip()


class _Skip(Exception):
    """This model can't answer now; try the next free one."""


def free_models(api_key: str = "") -> list[str]:
    """OpenRouter's current free models (id ends in ':free' and both prices are zero), largest context first."""
    req = urllib.request.Request(OpenRouterLLM.MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}
                                 if api_key else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)["data"]
    free = [m for m in data if m["id"].endswith(":free")
            and all(float(m.get("pricing", {}).get(k, 0) or 0) == 0 for k in ("prompt", "completion"))]
    free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    return [m["id"] for m in free]


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


if __name__ == "__main__":
    # Smoke test: which model is in use, and does it answer?  python -m navinib.agents.llm [--list]
    import sys
    if "--list" in sys.argv:
        print("\n".join(free_models({**_dotenv(), **os.environ}.get("OPENROUTER_API_KEY", ""))))
        raise SystemExit
    model = get_llm()
    print(f"Model: {model.name}")
    if isinstance(model, StubLLM):
        print("No OPENROUTER_API_KEY found (environment, Codespaces secret or .env), so the stub is used.")
    else:
        reply = model.complete("summary", "Reply in five words or fewer.", "Say hello to a field rep.")
        print(f"Answered by: {model.name}" + (f" (skipped {', '.join(model.tried)})" if model.tried else ""))
        print("Reply:", reply)
