"""Tests for the Checkpoint 6.1 guardrails: policy limits, input checks, claim checks, escalation, lanes and logs.

The scenarios come from the 6.1 plan: Dr. Lee's office reports a patient hospitalized with a
rash (adverse event, with a patient name to redact), a memo in Dr. Okafor's notes tries to
steer the model (injection), and a rep asks about first-line use (off-label question).
"""
import json
import pytest

from navinib import audit, config, guardrails
from navinib.agents import llm as llm_mod
from navinib.agents.coordinator import MAX_RETRIES
from navinib.agents.graph import brief_next_call, run
from navinib.agents.llm import DisabledLLM, OpenRouterLLM, StubLLM, get_llm
from navinib.agents.specs import AGENTS
from navinib.planner import tot
from navinib.rag.corpus import CALL_NOTES, LABEL_VERSION
from navinib.tools.schedule import to_min

retrieval = pytest.importorskip("navinib.rag.retrieval")

DAY = config.DEMO_DATE
MORNING = {"flow": "morning", "rep_id": "rep-01", "day": DAY}
NOTES = {n[0]: n[4] for n in CALL_NOTES}
AE_NOTE, INJECTED_NOTE = "N-0431", "N-0433"


@pytest.fixture(scope="module")
def col(tmp_path_factory):
    """One retrieval index for the file (skipped if the embedding model can't download)."""
    try:
        return retrieval.build_index(tmp_path_factory.mktemp("chroma"))
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {e}")


# ---------- the policy file drives the limits ----------

def test_limits_come_from_the_policy_file():
    """Hard limits are numbers in policy.yaml, not constants scattered in code or prompts."""
    lim = config.POLICY["limits"]
    assert MAX_RETRIES == lim["compliance_retries"] == 2
    assert config.MAX_REFILLS == lim["refill_rounds"] == 1
    assert config.TOT_MAX_PHONE == lim["phone_calls_per_window"] == 1
    assert config.TOT_MAX_STEPS == lim["calls_added_per_window"] == 3
    assert tot.VISIT_RADIUS_MIN == lim["replan_radius_min"] == 20
    assert config.ALERT_SUPPRESS_BELOW == lim["lab_alert_suppress_below"] == 5
    assert config.POLICY["model"]["temperature"] == 0.2


def test_no_agent_has_an_outbound_tool():
    """Nothing can email, text or call an office, or write to a CRM: those tools don't exist."""
    tools = {t for a in AGENTS.values() for t in a.tools}
    assert not {t for t in tools if any(w in t for w in ("email", "sms", "text", "call_office", "crm", "send"))}


# ---------- input checks ----------

def test_personal_data_is_redacted():
    """Patient names, record numbers, phone numbers and emails never reach the model or the logs."""
    text, kinds = guardrails.redact(NOTES[AE_NOTE] + " Call 412-555-0199 or nurse@example.org.")
    assert "Robert Hale" not in text and "448812" not in text and "412-555-0199" not in text
    assert "nurse@example.org" not in text
    assert {"mrn", "patient_name", "phone", "email"} <= set(kinds)
    assert guardrails.redact(NOTES["N-0412"])[1] == []      # an ordinary note is left alone


def test_the_adverse_event_check_catches_the_rash_note_and_spares_ordinary_notes():
    """Tuned for recall, but the original call notes raise no false alarm."""
    assert guardrails.adverse_event(NOTES[AE_NOTE])
    ordinary = ["N-0388", "N-0412", "N-0371", "N-0405", "N-0420"]
    assert [n for n in ordinary if guardrails.adverse_event(NOTES[n])] == []


class YesLLM(StubLLM):
    """A classifier model that recognizes an adverse event the keywords miss."""

    def _ae(self, d):
        """Always YES."""
        return "YES"


class DownLLM(StubLLM):
    """A model that is down."""

    def complete(self, task, system, user, data=None):
        """Always fail."""
        raise ConnectionError("model offline")


def test_the_model_catches_what_keywords_miss_and_keywords_stand_when_it_is_down():
    """Keywords first; the LLM classifier only adds recall, and its outage never crashes the check."""
    subtle = "Her patient felt dizzy and fell at home a week after starting Product X."
    assert guardrails.adverse_event(subtle) == []
    assert guardrails.adverse_event(subtle, YesLLM()) == ["model classifier"]
    assert guardrails.adverse_event(subtle, DownLLM()) == []
    assert guardrails.adverse_event(NOTES[AE_NOTE], DownLLM())


def test_instruction_like_text_is_caught():
    """The memo in Dr. Okafor's notes tries to steer the model; ordinary notes don't trip the scan."""
    assert guardrails.injection(NOTES[INJECTED_NOTE])
    assert [n for n in NOTES if n != INJECTED_NOTE and guardrails.injection(NOTES[n])] == []


def test_off_label_questions_are_recognized():
    """Unapproved uses go to Medical Information; on-label dosing questions don't."""
    assert guardrails.off_label_question("Can I show Dr. Leigh the first-line results?")
    assert guardrails.off_label_question("Does it work combined with immunotherapy in small cell?")
    assert not guardrails.off_label_question("What is the dose after Grade 3 neutropenia?")


# ---------- output check: claim by claim ----------

LABEL_51 = ("5.1 Neutropenia. Neutropenia occurred in 31% of patients treated with Product X, including "
            "Grade 3 or 4 in 12%.")
SOURCES = {"Label §5.1": LABEL_51, "N-0412": NOTES["N-0412"],
           "Label §1": "Product X is not indicated for first-line treatment."}


def test_a_supported_claim_stays():
    text, removed = guardrails.claim_check("Neutropenia occurred in 31% of patients [Label §5.1].", SOURCES)
    assert removed == [] and "31%" in text


def test_a_claim_with_a_wrong_number_is_removed_and_marked():
    """The number isn't in the cited label passage: removed, with a visible note, never silently."""
    text, removed = guardrails.claim_check("Neutropenia occurred in only 8% of patients [Label §5.1].", SOURCES)
    assert "8%" not in text and guardrails.UNSUPPORTED_NOTE in text
    assert removed[0]["why"] == "numbers not in the cited source: 8"


def test_a_claim_without_an_approved_source_is_removed():
    """Efficacy talk with no label citation, or citing only a call note, is not allowed."""
    _, removed = guardrails.claim_check("Response rates are the best in its class.", SOURCES)
    assert removed and removed[0]["why"] == "cites no approved source"
    _, removed = guardrails.claim_check("It is very effective first-line [N-0412].", SOURCES)
    assert removed


def test_history_and_label_limitations_pass():
    """What a provider said last time is history, and quoting the label's own limitation is on-label."""
    text = ("Last call [N-0412]: Left the dose-modification card.\n"
            "Product X is not indicated for first-line treatment [Label §1].")
    assert guardrails.claim_check(text, SOURCES)[1] == []


# ---------- the guardrails inside the agents ----------

def _lee_brief(con, col, **kw):
    """Cancel the 2pm at 13:05, book the replan, brief Hillcrest (next), then Dr. Lee at 13:30."""
    from navinib.agents.coordinator import approve_plan
    from navinib.tools.drive import DriveTimes
    drive = DriveTimes()
    if drive.source != "osrm":
        pytest.skip("needs the committed OSRM drive-time cache")
    plan = run(con, {"flow": "cancellation", "rep_id": "rep-01", "day": DAY, "event_id": "ev-04",
                     "now_min": to_min("13:05")}, col=col, drive=drive)
    approve_plan(con, plan)
    brief_next_call(con, "rep-01", DAY, to_min("13:05"), col=col, drive=drive, **kw)
    return brief_next_call(con, "rep-01", DAY, to_min("13:30"), col=col, drive=drive, **kw)


def test_an_adverse_event_note_is_escalated_once_and_kept_out_of_the_brief(con, col):
    """The 6.1 example: the rash note stops, goes to safety reporting, and the brief resumes without it."""
    final = _lee_brief(con, col)
    lee = final["briefs"]["p-001"]
    assert AE_NOTE not in lee["text"] and "rash" not in lee["text"].lower()
    assert [e["source_ref"] for e in final["escalations"]] == [AE_NOTE]
    assert any(l["lane"] == "escalate" and AE_NOTE in l["item"] for l in final["lanes"])
    assert "possible adverse event" in final["message"]
    row = con.execute("SELECT * FROM escalation WHERE source_ref = ?", (AE_NOTE,)).fetchone()
    assert row["routed_to"].startswith("safety reporting")
    assert "Robert Hale" not in row["excerpt"] and "448812" not in row["excerpt"]   # stored redacted
    again = guardrails.escalate(con, "adverse_event", AE_NOTE, NOTES[AE_NOTE], "rep-01")
    assert again["new"] is False                                                     # reported once


def test_an_injected_note_is_quarantined_and_never_reaches_the_brief(con, col):
    """Dr. Okafor's 8:45 brief leaves out the memo, flags it for admin review, and keeps it out afterwards."""
    final = brief_next_call(con, "rep-01", DAY, to_min("08:20"), col=col)
    okafor = final["briefs"]["p-004"]
    assert INJECTED_NOTE not in okafor["text"] and "first-line" not in okafor["text"]
    assert okafor["checks"]["quarantined"] == [INJECTED_NOTE]
    assert any(l["lane"] == "escalate" for l in final["lanes"])
    assert INJECTED_NOTE in guardrails.quarantined(con)


def test_a_superseded_label_version_is_blocked(con, col, monkeypatch):
    """When the label is revised, passages from the old version are dropped from briefs."""
    monkeypatch.setitem(config.POLICY["label"], "current_version", "2026-11-01")
    final = brief_next_call(con, "rep-01", DAY, to_min("08:20"), col=col)
    okafor = final["briefs"]["p-004"]
    assert okafor["checks"]["superseded_label"] and "Label" not in okafor["text"]
    assert LABEL_VERSION != guardrails.current_label()


def test_compliance_blocks_a_brief_built_on_an_old_label(con, col, monkeypatch):
    """Backstop: even if Brief let an old passage through, Compliance fails it."""
    from navinib.agents.compliance import _brief_issues
    b = {"text": "Dose is 160 mg [Label §2.1].", "allowed_refs": ["Label §2.1"], "do_not_raise": [],
         "status": "ok", "label_versions": ["2025-01-01"], "sources": {"Label §2.1": "160 mg once daily"}}
    issues, _ = _brief_issues(StubLLM(), "p-001", b)
    assert any("label version 2025-01-01" in i["problem"] for i in issues)


# ---------- the rep's own words ----------

def test_a_rep_message_about_an_adverse_event_stops_the_run(con):
    """No agent runs; the case goes to safety reporting and the rep is told what to do."""
    final = run(con, {**MORNING, "message": "Dr. Smith said a patient ended up in the emergency room on Product X"})
    assert final["status"] == "escalated" and "targets" not in final
    assert final["lanes"][0]["lane"] == "escalate" and "adverse event" in final["message"]


def test_an_off_label_question_goes_to_medical_information(con):
    final = run(con, {**MORNING, "message": "Can I share the first-line data with Dr. Leigh?"})
    assert final["status"] == "escalated" and "Medical Information" in final["message"]
    assert "plan" not in final


def test_an_unknown_request_gets_a_clarifying_question(con):
    final = run(con, {"flow": "email_the_office", "rep_id": "rep-01", "day": DAY})
    assert final["status"] == "needs_clarification" and final["lanes"][0]["lane"] == "flag"


# ---------- four lanes ----------

def test_every_plan_waits_for_the_rep(con):
    """Ranking and route drafts run on their own; the calendar change goes to the rep."""
    final = run(con, MORNING)
    lanes = {l["lane"] for l in final["lanes"]}
    assert {"auto", "rep_approves"} <= lanes and final["status"] == "awaiting_approval"


def test_close_replan_options_are_flagged():
    assert guardrails.close_options({"value": 1.00, "alternatives": [{"value": 0.97}]})
    assert not guardrails.close_options({"value": 1.39, "alternatives": [{"value": 1.20}]})


# ---------- kill switches and the model client ----------

def test_brief_kill_switch_gives_plan_only_mode(con, monkeypatch):
    monkeypatch.setitem(config.POLICY["agents"], "brief", False)
    final = brief_next_call(con, "rep-01", DAY, to_min("08:20"))
    assert final["status"] == "plan_only" and not final.get("briefs")
    monkeypatch.setitem(config.POLICY["agents"], "brief", True)
    assert run(con, MORNING)["status"] == "awaiting_approval"      # planning was never affected


def test_llm_kill_switch_falls_back_to_code(con, monkeypatch):
    """With the model off, the rep still gets a plan, with the summary written in code."""
    monkeypatch.setitem(config.POLICY["agents"], "llm", False)
    model = get_llm()
    assert isinstance(model, DisabledLLM)
    final = run(con, MORNING, llm=model)
    assert final["status"] == "awaiting_approval" and final["message"].startswith("Written summary unavailable")


def test_only_free_models_at_low_temperature(monkeypatch):
    """The client refuses paid models and uses the policy's settings."""
    with pytest.raises(ValueError):
        OpenRouterLLM("key", "openai/gpt-4o")
    m = OpenRouterLLM("key")
    assert all(q.endswith(":free") for q in m.queue) and m.temperature == 0.2
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm_mod, "_dotenv", lambda: {})
    llm = get_llm()
    assert isinstance(llm, llm_mod.CachedLLM) and isinstance(llm.inner, OpenRouterLLM)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    chain = get_llm().inner
    assert isinstance(chain, llm_mod.ChainLLM) and [m.provider for m in chain.models] == ["groq", "openrouter"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert isinstance(get_llm(), StubLLM)


# ---------- structured logs ----------

def test_every_run_is_logged_as_json_with_a_trace_id(con, tmp_path):
    path = tmp_path / "runs.jsonl"
    final = run(con, {**MORNING, "message": "Plan my day. Patient Robert Hale MRN 448812 called."}, log_path=path)
    entry = audit.read(path)[-1]
    assert entry["run_id"] == final["run_id"] and entry["status"] == final["status"]
    assert entry["policy_version"] == config.POLICY["version"] and entry["label_version"] == LABEL_VERSION
    assert set(entry["prompts"]) == {"brief", "chat", "claims", "summary", "adverse_event"}
    assert entry["lanes"] and entry["trace"]
    assert "Robert Hale" not in entry["request"]["message"] and "448812" not in entry["request"]["message"]


def test_a_retired_free_model_is_skipped_for_the_next_one(monkeypatch):
    """OpenRouter retires free models often: a 404 moves on to the next free model, never a paid one."""
    m = OpenRouterLLM("key")
    first, second = m.queue[0], m.queue[1]
    calls = []

    def fake(model, system, user):
        calls.append(model)
        if model == first:
            raise llm_mod._Skip(f"{model} -> 404: This model is unavailable for free")
        return "hello"
    monkeypatch.setattr(m, "_call", fake)
    assert m.complete("summary", "s", "u") == "hello"
    assert calls == [first, second] and m.model == second and m.tried == [first]
    assert m.complete("summary", "s", "u") == "hello" and calls[-1] == second   # stays on the working one


def test_a_model_refused_for_this_client_is_skipped(monkeypatch):
    """A 403 such as "only available on agentic harnesses" moves on; a 401 (bad key) does not."""
    import io
    import urllib.error
    m = OpenRouterLLM("key", "harness/only:free")
    monkeypatch.setattr(llm_mod, "free_models", lambda key="": ["harness/only:free", "works/model:free"])

    def fake_open(req, timeout=None):
        model = json.loads(req.data)["model"]
        if model == "harness/only:free":
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b"agentic harnesses only"))
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_open)
    with pytest.raises(RuntimeError, match="401"):
        m.complete("summary", "s", "u")
    assert m.tried == ["harness/only:free"] and m.model == "works/model:free"


def _http_error(code, body, headers=None):
    import io
    import urllib.error
    return urllib.error.HTTPError("https://x", code, "err", headers or {}, io.BytesIO(body.encode()))


def test_the_daily_free_limit_stops_at_once_and_the_next_provider_takes_over(monkeypatch):
    """Every OpenRouter free model shares one daily limit: hitting it must not burn calls on the others."""
    calls = []

    def fake_open(req, timeout=None):
        calls.append(json.loads(req.data)["model"])
        raise _http_error(429, "Rate limit exceeded: free-models-per-day")
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_open)
    router = OpenRouterLLM("key")
    backup = StubLLM()
    chain = llm_mod.ChainLLM([router, backup])
    assert chain.complete("claims", "s", "u") == "PASS"          # answered by the next provider
    assert len(calls) == 1 and router.exhausted and chain.name == "stub"
    with pytest.raises(RuntimeError, match="daily free limit"):
        router.complete("summary", "s", "u")
    assert len(calls) == 1                     # no more calls once exhausted


def test_a_per_minute_rate_limit_waits_and_retries_the_same_model(monkeypatch):
    m = llm_mod.GroqLLM("key")
    waits, n = [], {"calls": 0}
    m.sleep = waits.append

    class Ok:
        def __enter__(self):
            import io
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        n["calls"] += 1
        if n["calls"] == 1:
            raise _http_error(429, "Rate limit reached for tokens per minute", {"retry-after": "2"})
        return Ok()
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_open)
    assert m.complete("summary", "s", "u") == "hi" and waits == [2.0] and m.tried == []


def test_a_busy_model_is_passed_over_for_one_call_not_dropped(monkeypatch):
    """Upstream rate limits and empty answers are temporary: the model is tried again next call."""
    m = OpenRouterLLM("key", "busy/model:free")
    monkeypatch.setattr(llm_mod, "free_models", lambda key="": ["backup/model:free"])
    state = {"busy": True}

    def fake(model, system, user):
        if model == "busy/model:free" and state["busy"]:
            raise llm_mod._Skip("busy/model:free returned no text")
        return model
    monkeypatch.setattr(m, "_call", fake)
    assert m.complete("summary", "s", "u") == "backup/model:free"
    state["busy"] = False
    assert "busy/model:free" not in m.dead
    assert m.complete("summary", "s", "u") == "backup/model:free"      # the one that answered goes first
    m.queue.remove("backup/model:free")
    assert m.complete("summary", "s", "u") == "busy/model:free"


def test_a_paid_model_needs_an_explicit_opt_in(monkeypatch):
    with pytest.raises(ValueError, match="NAVINIB_ALLOW_PAID"):
        OpenRouterLLM("key", "meta-llama/llama-3.3-70b-instruct")
    assert OpenRouterLLM("key", "meta-llama/llama-3.3-70b-instruct", allow_paid=True).model == \
        "meta-llama/llama-3.3-70b-instruct"
    monkeypatch.setattr(llm_mod, "_dotenv", lambda: {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("NAVINIB_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setenv("NAVINIB_ALLOW_PAID", "1")
    assert get_llm().name == "openrouter:meta-llama/llama-3.3-70b-instruct"


def test_answers_are_cached_so_reruns_spend_no_quota(tmp_path):
    class Counting:
        name, n = "fake:model", 0

        def complete(self, task, system, user, data=None):
            Counting.n += 1
            return f"answer {Counting.n}"
    c = llm_mod.CachedLLM(Counting(), tmp_path)
    assert c.complete("brief", "s", "u") == "answer 1"
    assert llm_mod.CachedLLM(Counting(), tmp_path).complete("brief", "s", "u") == "answer 1"
    assert Counting.n == 1 and c.complete("brief", "s", "other") == "answer 2"


def test_when_every_listed_model_fails_it_asks_openrouter_for_free_ones(monkeypatch):
    m = OpenRouterLLM("key", "only/one:free")
    monkeypatch.setattr(llm_mod, "free_models", lambda key="": ["only/one:free", "fresh/model:free"])
    monkeypatch.setattr(m, "_call", lambda model, s, u: "ok" if model == "fresh/model:free"
                        else (_ for _ in ()).throw(llm_mod._Skip("gone")))
    assert m.complete("summary", "s", "u") == "ok" and m.model == "fresh/model:free"


def test_empty_environment_variable_does_not_hide_dotenv(monkeypatch):
    """A blank NAVINIB_MODEL in the shell must not hide the paid model named in .env."""
    monkeypatch.setattr(llm_mod, "_dotenv", lambda: {"NAVINIB_MODEL": "meta-llama/llama-3.3-70b-instruct",
                                                     "OPENROUTER_API_KEY": "sk-env-file"})
    monkeypatch.setenv("NAVINIB_MODEL", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-shell")
    s = llm_mod.settings()
    assert s["NAVINIB_MODEL"] == "meta-llama/llama-3.3-70b-instruct" and s["OPENROUTER_API_KEY"] == "sk-shell"


def test_citation_format_is_normalized_before_the_check():
    """[Label 5.1] is the same source as [Label §5.1]: the claim stays, its numbers still checked."""
    text, removed = guardrails.claim_check("Neutropenia occurred in 31% of patients [Label 5.1].", SOURCES)
    assert not removed and "[Label §5.1]" in text
    _, removed = guardrails.claim_check("Neutropenia occurred in 8% of patients [Label 5.1].", SOURCES)
    assert removed and removed[0]["why"].startswith("numbers not in the cited source")
