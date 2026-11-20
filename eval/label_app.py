"""Label claim sentences for the judge-agreement metric: python -m streamlit run eval/label_app.py

One sentence at a time with the source text it cites. Answer only from that text:
is every fact and number in the sentence there? The claim check's own verdict is hidden.
Answers save as you go to eval/labels/answers.json, so you can stop and come back.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.label_sheet import ANSWERS, SHEET, score  # noqa: E402

st.set_page_config(page_title="Navinib labels", page_icon="🏷️")
st.title("🏷️ Label claim sentences")
if not SHEET.exists():
    st.error("No sheet yet. Run `python -m eval.label_sheet` first.")
    st.stop()
rows = json.loads(SHEET.read_text(encoding="utf-8"))["rows"]
answers = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.exists() else {}
left = [r for r in rows if r["id"] not in answers]
st.progress(1 - len(left) / len(rows), text=f"{len(rows) - len(left)} of {len(rows)} labeled")


def save(rid: str, value: str) -> None:
    answers[rid] = value
    ANSWERS.write_text(json.dumps(answers, indent=1), encoding="utf-8")


if left:
    r = left[0]
    st.caption(f"{r['id']} · provider {r['provider']}")
    st.markdown("**Sentence from the brief**")
    st.info(r["sentence"])
    st.markdown("**What it cites**")
    if r["sources"]:
        for ref, text in r["sources"].items():
            st.markdown(f"`{ref}`")
            st.text(text)
    else:
        st.text("(nothing: the sentence has no citation)")
    st.markdown("Is every fact and number in the sentence supported by the cited text?")
    a, b, c = st.columns(3)
    if a.button("✅ Supported", use_container_width=True):
        save(r["id"], "supported")
        st.rerun()
    if b.button("❌ Not supported", use_container_width=True):
        save(r["id"], "unsupported")
        st.rerun()
    if c.button("Skip (unsure)", use_container_width=True):
        save(r["id"], "skip")
        st.rerun()
    st.caption("Advice to the rep (\"be prepared to...\", \"refer them to Medical Information\") counts as "
               "supported. A product claim with no citation counts as not supported. A sentence that drops a "
               "limitation changing who the product is for (for example, leaves out \"after platinum-based "
               "chemotherapy\" from the indication) counts as not supported.")
else:
    st.success("All done. Thank you!")
    s = score()
    if s:
        st.metric("Agreement with the claim check (Cohen's kappa)", s["kappa"])
        st.write(s)
    st.markdown("Now commit `eval/labels/` (or send the file `eval/labels/answers.json`).")
if answers and st.button("Undo last answer"):
    answers.pop(list(answers)[-1])
    ANSWERS.write_text(json.dumps(answers, indent=1), encoding="utf-8")
    st.rerun()
