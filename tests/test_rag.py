"""Retrieval tests. The first run downloads the free MiniLM embedding model (~80 MB)."""
import pytest

retrieval = pytest.importorskip("navinib.rag.retrieval")


@pytest.fixture(scope="module")
def col(tmp_path_factory):
    try:
        return retrieval.build_index(tmp_path_factory.mktemp("chroma"))
    except Exception as e:  # no network for the model download
        pytest.skip(f"embedding model unavailable: {e}")


def test_call_notes_stay_with_one_provider(col):
    hits = retrieval.call_note_recall(col, "p-001", "first-line results and neutropenia")  # Dr. Lee
    assert hits and all(h["meta"]["provider_id"] == "p-001" for h in hits)


def test_label_lookup_returns_only_approved(col):
    hits = retrieval.label_lookup(col, "first-line therapy response rate")
    assert all(h["meta"]["status"] == "approved" for h in hits)


def test_out_of_scope_returns_nothing(col):
    assert retrieval.label_lookup(col, "reimbursement and copay assistance for Product X") == []
