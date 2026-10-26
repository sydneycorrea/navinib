"""Navinib retrieval layer (from the Checkpoint 3.1 prototype).

Run the 3.1 demo:  python -m navinib.rag.retrieval

Builds a Chroma vector store from the synthetic corpus and exposes the
retrieval tools the ReAct planner calls in Action 4 (build the brief):
  label_lookup        approved label text and approved trial data
  call_note_recall    past call notes for one provider (long-term memory)
  trial_search        trial summaries, tagged approved or investigational
  publication_search  the provider's own publications, tagged by use_status

Embeddings: all-MiniLM-L6-v2 (free, runs locally through Chroma's ONNX build).
"""
import sys

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

from navinib.config import CHROMA_PATH
from navinib.rag.corpus import CALL_NOTES, LABEL, PUBLICATIONS, TRIALS

CHUNK_WORDS, OVERLAP_WORDS = 300, 50   # ~400 tokens with 50-word overlap
MIN_SIM = 0.30                          # below this, treat as "not found"


def chunk(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split on words; short sections stay whole, long ones overlap."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def build_index(path=CHROMA_PATH):
    client = chromadb.PersistentClient(path=str(path))
    ef = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2
    try:
        client.delete_collection("navinib")
    except Exception:
        pass
    col = client.create_collection("navinib", embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for section, text in LABEL:
        for i, piece in enumerate(chunk(text)):
            ids.append(f"label-{section.split()[0]}-{i}")
            docs.append(f"{section}. {piece}")
            metas.append({"source": "label", "status": "approved", "ref": f"Label §{section.split()[0]}"})
    for nct, status, text in TRIALS:
        ids.append(nct)
        docs.append(text)
        metas.append({"source": "trial", "status": status, "ref": nct})
    for nid, pid, name, date, text in CALL_NOTES:
        ids.append(nid)
        docs.append(f"{name}, {date}: {text}")
        metas.append({"source": "call_note", "provider_id": pid, "date": date, "ref": nid})
    for pub_id, authors, title, journal, year, use_status, abstract in PUBLICATIONS:
        for pid in authors:  # one entry per author so the provider_id filter stays exact
            ids.append(f"{pub_id}-{pid}")
            docs.append(f"{title}. {journal}, {year}. {abstract}")
            metas.append({"source": "publication", "provider_id": pid, "status": use_status,
                          "year": year, "ref": pub_id})
    col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def _search(col, query, where, k, fetch_k=8, lam=0.7):
    """Cosine top-fetch_k under a metadata filter, then MMR down to k."""
    r = col.query(query_texts=[query], n_results=fetch_k, where=where,
                  include=["documents", "metadatas", "distances", "embeddings"])
    sims = [1 - d for d in r["distances"][0]]
    cands = [i for i, s in enumerate(sims) if s >= MIN_SIM]
    embs = np.array(r["embeddings"][0])
    chosen = []
    while cands and len(chosen) < k:   # MMR (Lab 3.2)
        def score(i):
            red = max((float(embs[i] @ embs[j]) for j in chosen), default=0.0)
            return lam * sims[i] - (1 - lam) * red
        best = max(cands, key=score)
        chosen.append(best)
        cands.remove(best)
    return [{"id": r["ids"][0][i], "sim": round(sims[i], 3), "meta": r["metadatas"][0][i],
             "text": r["documents"][0][i]} for i in chosen]


def label_lookup(col, query, k=3):
    return _search(col, query, {"status": "approved"}, k)


def call_note_recall(col, provider_id, query, k=2):
    return _search(col, query, {"$and": [{"source": "call_note"}, {"provider_id": provider_id}]}, k)


def recent_call_notes(col, provider_id, k=2):
    """The provider's latest notes by date, with no similarity cut-off (used when there's no topic)."""
    r = col.get(where={"$and": [{"source": "call_note"}, {"provider_id": provider_id}]},
                include=["documents", "metadatas"])
    hits = [{"id": i, "sim": None, "meta": m, "text": t}
            for i, m, t in zip(r["ids"], r["metadatas"], r["documents"])]
    return sorted(hits, key=lambda h: h["meta"]["date"], reverse=True)[:k]


def trial_search(col, query, k=2):
    return _search(col, query, {"source": "trial"}, k)


def publication_search(col, provider_id, query, k=2):
    return _search(col, query, {"$and": [{"source": "publication"}, {"provider_id": provider_id}]}, k)


def naive_search(col, query, k=4):
    """Baseline: plain top-k over everything, no filters, no threshold, no MMR."""
    r = col.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    return [{"id": i, "sim": round(1 - d, 3), "meta": m, "text": t}
            for i, d, m, t in zip(r["ids"][0], r["distances"][0], r["metadatas"][0], r["documents"][0])]


def show(title, hits):
    print(f"  {title}")
    for h in hits:
        tag = h["meta"].get("status", h["meta"].get("provider_id", ""))
        print(f"    {h['id']:<16} sim={h['sim']:.3f}  [{h['meta']['source']}/{tag}]  {h['text'][:90]}...")
    if not hits:
        print("    (nothing above the similarity threshold: report 'not found')")


if __name__ == "__main__":
    col = build_index(sys.argv[1] if len(sys.argv) > 1 else CHROMA_PATH)
    print(f"Indexed {col.count()} chunks\n")

    print("=== Action 4: build the brief for the substitute, Dr. Alicia Lee (provider_id=p-001) ===")
    notes = call_note_recall(col, "p-001", "neutropenia concerns, dose reduction and lab monitoring")
    show("Tool: Call-note recall (filter provider_id=p-001, k=2)", notes)
    q = "neutropenia in older patients, dose reduction and CBC monitoring"
    label = label_lookup(col, q)
    show(f"Tool: Label lookup (filter status=approved, k=3, MMR) query='{q}'", label)
    trials = trial_search(col, "neutropenia in patients over 65")
    show("Tool: Literature and trial search (k=2)", trials)

    print("\n  Brief (talking points may cite approved label text only):")
    for h in notes:
        print(f"    Last visit [{h['id']}, {h['meta']['date']}]: {h['text'].split(': ', 1)[1][:150]}...")
    for h in label:
        print(f"    Talking point [{h['meta']['ref']}]: {h['text'][:150]}...")
    for h in trials:
        if h["meta"]["status"] != "approved":
            print(f"    Do not raise [{h['id']}]: investigational; if asked, refer to Medical Information.")

    print("\n=== Baseline: naive top-4 over the whole store, no filters ===")
    show("query='Dr. Lee wants data on Product X neutropenia and first-line results'",
         naive_search(col, "Dr. Lee wants data on Product X neutropenia and first-line results"))

    print("\n=== Same request through the filtered tools ===")
    q2 = "Dr. Lee wants data on Product X neutropenia and first-line results"
    show("Call-note recall (provider_id=p-001)", call_note_recall(col, "p-001", q2))
    show("Label lookup (status=approved)", label_lookup(col, q2))

    print("\n=== Out-of-scope check: question the store cannot answer ===")
    show("Label lookup query='reimbursement and copay assistance for Product X'",
         label_lookup(col, "reimbursement and copay assistance for Product X"))
