"""Retrieval tests. The first run downloads the free MiniLM embedding model (~80 MB)."""
import pytest

retrieval = pytest.importorskip("navinib.rag.retrieval")


@pytest.fixture(scope="module")
def col(tmp_path_factory):
    """One Chroma index shared by the tests in this file (skipped if the model can't download)."""
    try:
        return retrieval.build_index(tmp_path_factory.mktemp("chroma"))
    except Exception as e:  # no network for the model download
        pytest.skip(f"embedding model unavailable: {e}")


def test_call_notes_stay_with_one_provider(col):
    """A call-note search for Dr. Lee only returns Dr. Lee's notes."""
    hits = retrieval.call_note_recall(col, "p-001", "first-line results and neutropenia")  # Dr. Lee
    assert hits and all(h["meta"]["provider_id"] == "p-001" for h in hits)


def test_label_lookup_returns_only_approved(col):
    """Label lookup never returns investigational material."""
    hits = retrieval.label_lookup(col, "first-line therapy response rate")
    assert all(h["meta"]["status"] == "approved" for h in hits)


def test_out_of_scope_returns_nothing(col):
    """A question the label doesn't cover returns nothing rather than a weak match."""
    assert retrieval.label_lookup(col, "reimbursement and copay assistance for Product X") == []


def test_publications_are_per_provider(col):
    """A publication search returns only that provider's papers, best match first."""
    hits = retrieval.publication_search(col, "p-001", "neutropenia in older patients")  # Dr. Lee
    assert hits and all(h["meta"]["provider_id"] == "p-001" for h in hits)
    assert hits[0]["meta"]["ref"] == "PUB-SYN-001"


def test_brief_lists_publications_and_flags_unapproved_use(col, con):
    """Briefs list each provider's papers and contacts, and mark unapproved-use papers do-not-discuss."""
    from navinib import config
    from navinib.tools.brief import build_brief, render_brief

    lee = build_brief(con, col, "p-001", config.DEMO_DATE)
    assert [p["pub_id"] for p in lee["publications"]] == ["PUB-SYN-002", "PUB-SYN-001"]
    assert all(p["may_discuss"] for p in lee["publications"])
    assert {c["full_name"] for c in lee["contacts"]} == {"Kevin Tran", "Maria Gomez"}

    leigh = build_brief(con, col, "p-002", config.DEMO_DATE)
    assert {"ref": "PUB-SYN-003", "why": "their publication covers an unapproved use"} in leigh["do_not_raise"]
    assert "background only, do not discuss" in render_brief(leigh)

    okafor = build_brief(con, col, "p-004", config.DEMO_DATE)
    assert okafor["publications"] == []
    assert all(t["ref"].startswith(("Label", "NCT")) for t in okafor["talking_points"])
