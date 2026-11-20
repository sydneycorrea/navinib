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
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
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
    """A model on OpenRouter, through its OpenAI-compatible chat API. Standard library only.

    OpenRouter's free models come and go, so the client keeps a list: NAVINIB_MODEL if set,
    then the policy's preferred models, then whatever OpenRouter currently lists as free.
    Each call tries up to MAX_MODELS_PER_CALL models. A model that is gone, not free or refused
    (400/402/403/404) is dropped for the run; one that is busy, rate-limited upstream or returns
    no text is only passed over for that call and tried again on the next one. The model that
    last answered goes first. A per-minute rate limit waits and retries; the account's daily
    free limit stops the provider at once (every free model shares it).
    Paid models are refused unless NAVINIB_ALLOW_PAID=1 (the course allows a paid model).
    """
    provider = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    DISCOVER = True                      # ask the provider for its current free models when the list runs out
    PERMANENT = (400, 402, 403, 404)     # retired, not free, refused for this client: drop for the run
    TRANSIENT = (408, 409, 429, 500, 502, 503, 504)   # busy or overloaded: pass over for this call only
    RETRIES_429 = 2                      # waits on a rate limit before passing the model over
    MAX_MODELS_PER_CALL = 4

    def __init__(self, api_key: str, model: str | None = None, allow_paid: bool = False):
        """Use the policy's settings; refuse any model that isn't free unless paid use is allowed."""
        m = POLICY["model"]
        self.allow_paid = allow_paid
        self.queue = [model] if model else list(self.preferred())
        bad = [q for q in self.queue if not (allow_paid or self.is_free(q))]
        if bad:
            raise ValueError(f"{bad[0]} is not a free model; policy.yaml allows free models only "
                             "(set NAVINIB_ALLOW_PAID=1 to allow a paid one)")
        self.api_key = api_key
        self.temperature, self.max_tokens, self.timeout = m["temperature"], m["max_tokens"], m["timeout_s"]
        self.tried: list[str] = []       # every model that failed at least once (for the report)
        self.dead: set[str] = set()      # dropped for the rest of the run
        self.discovered = not self.DISCOVER
        self.exhausted = ""              # set when the daily free limit is hit
        self.sleep = time.sleep

    @staticmethod
    def preferred() -> list[str]:
        return POLICY["model"]["preferred"]

    @staticmethod
    def is_free(model: str) -> bool:
        return not POLICY["model"]["free_only"] or model.endswith(":free")

    @property
    def model(self) -> str:
        """The model in use now: the first one not dropped."""
        return next((q for q in self.queue if q not in self.dead), "none")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        """One chat completion. Retrieved text arrives inside `user` as quoted JSON data, never as instructions."""
        errors, now = [], []
        while len(now) < self.MAX_MODELS_PER_CALL:
            if self.exhausted:
                raise RuntimeError(self.exhausted)
            cands = [q for q in self.queue if q not in self.dead and q not in now]
            if not cands:
                if self.discovered:
                    break
                self.discovered = True
                self.queue += [q for q in free_models(self.api_key) if q not in self.queue]
                continue
            model = cands[0]
            now.append(model)
            try:
                text = self._call(model, system, user)
            except _Skip as e:
                errors.append(str(e))
                if model not in self.tried:
                    self.tried.append(model)
                if e.permanent:
                    self.dead.add(model)
                continue
            self.queue.remove(model)
            self.queue.insert(0, model)  # the model that answered goes first next time
            return text
        raise RuntimeError(f"no {self.provider} model answered: " + "; ".join(errors[-3:]))

    def _call(self, model: str, system: str, user: str) -> str:
        body = {"model": model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
            "X-Title": "Navinib capstone"})
        for attempt in range(self.RETRIES_429 + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.load(r)
                break
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                msg = f"{model} -> {e.code}: {text[:160]}"
                if e.code == 429 and re.search(r"per[- ]day|daily", text, re.I):
                    self.exhausted = f"{self.provider} daily free limit reached ({msg})"
                    raise RuntimeError(self.exhausted) from e
                if e.code == 429 and attempt < self.RETRIES_429:
                    wait = e.headers.get("retry-after") if e.headers else None
                    self.sleep(min(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 8.0, 30.0))
                    continue
                if e.code in self.PERMANENT:
                    raise _Skip(msg, permanent=True) from e
                if e.code in self.TRANSIENT:
                    raise _Skip(msg) from e
                raise RuntimeError(f"{self.provider} {msg}") from e
            except (TimeoutError, json.JSONDecodeError) as e:
                raise _Skip(f"{model} -> {type(e).__name__}") from e
        if out.get("error"):
            raise _Skip(f"{model} -> {str(out['error'])[:160]}")
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content")
        if not text:   # e.g. a reasoning model that spent the token cap thinking
            raise _Skip(f"{model} returned no text")
        return text.strip()


class GroqLLM(OpenRouterLLM):
    """Groq's free tier (no card needed), same OpenAI-compatible API. Its model ids carry no
    ":free" suffix: a free-tier key can only use free models, so every listed model is free."""
    provider = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"
    DISCOVER = False

    @staticmethod
    def preferred() -> list[str]:
        return POLICY["model"]["groq"]["preferred"]

    @staticmethod
    def is_free(model: str) -> bool:
        return True


class ChainLLM:
    """Providers tried in order (policy model.providers). When one gives up, for example at its
    daily free limit, the rest of the run uses the next one."""

    def __init__(self, models: list):
        self.models, self.i = models, 0

    @property
    def current(self):
        return self.models[min(self.i, len(self.models) - 1)]

    @property
    def name(self) -> str:
        return self.current.name

    @property
    def tried(self) -> list[str]:
        return [t for m in self.models for t in getattr(m, "tried", [])]

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        errors = []
        while self.i < len(self.models):
            try:
                return self.models[self.i].complete(task, system, user, data)
            except RuntimeError as e:
                errors.append(str(e)[:200])
                if len(self.models) == 1:
                    raise
                self.i += 1
        self.i = len(self.models) - 1
        raise RuntimeError("no provider answered: " + " | ".join(errors))


class CachedLLM:
    """Saves each real-model answer under .cache/llm/, keyed by task and prompt, so rerunning the
    demo or the evaluation does not spend the free daily quota again. Delete the folder to refresh."""

    def __init__(self, inner, path: Path | None = None):
        self.inner, self.path = inner, path or ROOT / ".cache" / "llm"
        self.hits = self.misses = 0

    def __getattr__(self, attr):
        return getattr(self.inner, attr)

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, task: str, system: str, user: str, data: dict | None = None) -> str:
        key = hashlib.sha256(json.dumps([task, system, user]).encode()).hexdigest()[:32]
        f = self.path / f"{key}.json"
        if f.exists():
            self.hits += 1
            return json.loads(f.read_text(encoding="utf-8"))["text"]
        text = self.inner.complete(task, system, user, data)
        self.misses += 1
        self.path.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"model": self.inner.name, "task": task, "text": text}), encoding="utf-8")
        return text


class _Skip(Exception):
    """This model can't answer now; try the next one. `permanent` drops it for the run."""

    def __init__(self, msg: str, permanent: bool = False):
        super().__init__(msg)
        self.permanent = permanent


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


PROVIDERS = {"groq": ("GROQ_API_KEY", GroqLLM), "openrouter": ("OPENROUTER_API_KEY", OpenRouterLLM)}


def get_llm() -> LLM:
    """The model to use: disabled by the kill switch; else each provider in policy model.providers
    that has a key (Groq, then OpenRouter), answers cached; else the stub."""
    if not POLICY["agents"].get("llm", True):
        return DisabledLLM()
    env = {**_dotenv(), **os.environ}
    models = []
    for p in POLICY["model"]["providers"]:
        var, cls = PROVIDERS[p]
        key = env.get(var, "").strip()
        if key:
            pinned = env.get("NAVINIB_MODEL") if p == "openrouter" else env.get("NAVINIB_GROQ_MODEL")
            paid = env.get("NAVINIB_ALLOW_PAID", "").strip().lower() in ("1", "true", "yes")
            models.append(cls(key, pinned or None, allow_paid=paid))
    if not models:
        return StubLLM()
    llm = models[0] if len(models) == 1 else ChainLLM(models)
    return CachedLLM(llm) if POLICY["model"].get("cache", True) else llm


if __name__ == "__main__":
    # Smoke test: which model is in use, and does it answer?  python -m navinib.agents.llm [--list]
    import sys
    if "--list" in sys.argv:
        print("\n".join(free_models({**_dotenv(), **os.environ}.get("OPENROUTER_API_KEY", ""))))
        raise SystemExit
    model = get_llm()
    print(f"Model: {model.name}")
    if isinstance(model, StubLLM):
        print("No GROQ_API_KEY or OPENROUTER_API_KEY found (environment, Codespaces secret or .env), "
              "so the stub is used.")
    else:
        reply = model.complete("summary", "Reply in five words or fewer.",
                               f"Say hello to a field rep. ({time.strftime('%H:%M:%S')})")
        print(f"Answered by: {model.name}" + (f" (skipped {', '.join(model.tried)})" if model.tried else ""))
        print("Reply:", reply)
