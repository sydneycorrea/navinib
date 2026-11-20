"""Navinib app: the rep's day and the regional manager's view (Streamlit).

    python -m streamlit run app/main.py

Rep view: a calendar-style day, the actions a rep takes (plan the morning, report a
cancellation, brief the next call, ask a question), the plan waiting for approval with
its alternatives, and the lane each output went to. Manager view: read-only, team level:
goal coverage, the weekly sample of briefs, escalations and the rep's plan decisions.

Each browser session works on its own copy of the synthetic demo day, so nothing here
touches data/navinib.db. Everything is synthetic.
"""
import html
import sys
import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navinib import audit, config  # noqa: E402
from navinib.agents.coordinator import approve_plan  # noqa: E402
from navinib.agents.graph import brief_next_call, run  # noqa: E402
from navinib.agents.llm import get_llm  # noqa: E402
from navinib.data.generate import generate  # noqa: E402
from navinib.db import connect  # noqa: E402
from navinib.tools.drive import DriveTimes  # noqa: E402
from navinib.tools.providers import alert_display  # noqa: E402
from navinib.tools.schedule import hhmm, schedule_lookup, to_min  # noqa: E402

DAY, REP = config.DEMO_DATE, "rep-01"
DAY_START, DAY_END, PX_PER_HOUR = 8 * 60, 18 * 60, 64
LANE_STYLE = {"auto": ("#e8f1fb", "#1f5fa8"), "rep_approves": ("#e7f6ec", "#1e7a3c"),
              "flag": ("#fff4dc", "#8a5a00"), "escalate": ("#fde8e8", "#a12020")}

st.set_page_config(page_title="Navinib", page_icon="🧭", layout="wide")


# ---------- session data ----------

@st.cache_resource(show_spinner="Building the retrieval index (first run downloads a small embedding model)...")
def retrieval_index():
    """One Chroma index for the app; it only holds the synthetic corpus, so sessions can share it."""
    try:
        from navinib.rag.retrieval import build_index
        return build_index(Path(tempfile.gettempdir()) / "navinib_app_chroma")
    except Exception as e:   # no embedding model: briefs fall back to partial ones
        st.warning(f"Retrieval unavailable ({e}); briefs will be partial.")
        return None


def new_session() -> None:
    """A fresh copy of the demo day for this browser session."""
    folder = Path(tempfile.gettempdir()) / "navinib_app"
    folder.mkdir(exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.db"
    generate(path)
    st.session_state.update(db=str(path), result=None, kind=None, log=folder / f"{path.stem}.jsonl",
                            history=[])


if "db" not in st.session_state:
    new_session()
con = connect(st.session_state["db"])
col = retrieval_index()
drive = DriveTimes()


def record_decision(decision: str, detail: str) -> None:
    """Long-term memory of what the rep did with a plan: the override log the manager samples."""
    con.execute("INSERT INTO plan_decision VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex[:10], REP, DAY, decision, detail))
    con.commit()


# ---------- pieces of the page ----------

def lane_chips(lanes: list) -> str:
    """Each output with the lane it went to, as coloured chips."""
    out = []
    for l in lanes:
        bg, fg = LANE_STYLE[l["lane"]]
        out.append(f"<div style='margin:4px 0;padding:6px 10px;border-radius:8px;background:{bg};color:{fg};"
                   f"font-size:13px'><b>{html.escape(l['lane_name'])}</b> · {html.escape(l['item'])}: "
                   f"{html.escape(l['why'])}</div>")
    return "".join(out)


MODE = {"face_to_face": ("🚶", "Face to face"), "phone": ("📞", "Phone call")}


def office(location_id: str | None) -> dict:
    """The office a call is at: practice name and address (empty for tasks with no place)."""
    row = con.execute("SELECT affiliation, address FROM location WHERE location_id = ?",
                      (location_id,)).fetchone() if location_id else None
    return {"name": row["affiliation"], "address": row["address"]} if row else {"name": "", "address": ""}


def call_summary(start: str, end: str, call_type: str, location_id: str | None, status: str = "") -> dict:
    """The quick summary of one call: when, how (face to face or phone) and where."""
    icon, mode = MODE.get(call_type, MODE["face_to_face"])
    where = office(location_id)
    return {"when": f"{start}–{end}", "icon": icon, "mode": mode, "office": where["name"],
            "address": where["address"], "status": status}


def calendar(events: list, proposed: list, briefed: set, now_min: int) -> str:
    """A one-day calendar: booked calls, fixed tasks, cancelled calls, proposed calls and the current time.
    Each call shows how it happens and where; hovering over it shows the full summary."""
    height = (DAY_END - DAY_START) * PX_PER_HOUR // 60
    y = lambda m: (m - DAY_START) * PX_PER_HOUR / 60   # noqa: E731
    rows = []
    for h in range(DAY_START // 60, DAY_END // 60 + 1):
        top = y(h * 60)
        rows.append(f"<div style='position:absolute;top:{top}px;left:0;right:0;border-top:1px solid #e5e7eb'></div>"
                    f"<div style='position:absolute;top:{top - 8}px;left:0;width:44px;font-size:11px;"
                    f"color:#6b7280'>{h:02d}:00</div>")

    def block(start, end, title, sub, bg, border, extra="", tip=""):
        top, h = y(start), max(y(end) - y(start), 18)
        return (f"<div title='{html.escape(tip or title, quote=True)}' style='position:absolute;top:{top}px;"
                f"height:{h - 2}px;left:52px;right:4px;background:{bg};cursor:default;"
                f"border-left:4px solid {border};border-radius:6px;padding:2px 8px;font-size:12px;overflow:hidden;"
                f"white-space:nowrap;text-overflow:ellipsis;"
                f"{extra}'><b>{html.escape(title)}</b> <span style='color:#4b5563'>{html.escape(sub)}</span></div>")

    def tip(title, c):
        return "\n".join(x for x in (title, c["when"], f"{c['mode']}", c["office"], c["address"], c["status"]) if x)

    for e in events:
        sub = f"{e['start']}–{e['end']}"
        if e["status"] == "cancelled":
            rows.append(block(e["start_min"], e["end_min"], e["title"], sub + " · cancelled", "#f3f4f6", "#9ca3af",
                              "text-decoration:line-through;color:#9ca3af"))
        elif e["kind"] == "call":
            c = call_summary(e["start"], e["end"], e["call_type"], e["location_id"],
                             "Brief ready" if e["event_id"] in briefed else "")
            tag = " · brief ready" if e["event_id"] in briefed else ""
            rows.append(block(e["start_min"], e["end_min"], e["title"], f"{sub} · {c['icon']} {c['office']}{tag}",
                              "#e8f1fb", "#1f5fa8", tip=tip(e["title"], c)))
        else:
            rows.append(block(e["start_min"], e["end_min"], e["title"], sub, "#f5f3ff", "#7c3aed"))
    for v in proposed:
        s, en = to_min(v["start"]), to_min(v["end"])
        c = call_summary(v["start"], v["end"], v["call_type"], v.get("location_id"), "Proposed, waiting for you")
        title = f"Proposed: {v['full_name']}"
        rows.append(block(s, en, title, f"{c['when']} · {c['icon']} {c['office']}",
                          "#e7f6ec", "#1e7a3c", "border:1px dashed #1e7a3c;border-left:4px solid #1e7a3c",
                          tip=tip(title, c)))
    if DAY_START <= now_min <= DAY_END:
        rows.append(f"<div style='position:absolute;top:{y(now_min)}px;left:44px;right:0;border-top:2px solid "
                    f"#dc2626'></div><div style='position:absolute;top:{y(now_min) - 7}px;left:0;width:44px;font-size:11px;"
                    f"font-weight:600;color:#dc2626;background:white'>{hhmm(now_min)}</div>")
    return (f"<div style='position:relative;height:{height + 10}px;margin-top:10px;font-family:sans-serif'>"
            + "".join(rows) + "</div>")


def show_result(result: dict) -> None:
    """The latest answer from Navinib: message, proposed plan or brief, and the lanes."""
    status = result["status"]
    st.markdown(f"**Navinib** · `{status}` · trace `{result.get('run_id', '')}`")
    st.info(result.get("message") or "")
    plan = result.get("plan")
    if plan and status == "awaiting_approval":
        if plan.get("reason"):
            st.caption(f"Why this plan: {plan['reason']}")
        for a in plan.get("alternatives", []):
            st.caption(f"Alternative ({a['value']:.2f}): {a['summary']}")
        c1, c2 = st.columns(2)
        if c1.button("Approve and book", type="primary", use_container_width=True):
            ids = approve_plan(con, result)
            record_decision("accepted", f"{result['request']['flow']}: {', '.join(ids)}")
            st.session_state["result"] = None
            st.session_state["history"].append(f"Approved: booked {len(ids)} call(s)")
            st.session_state["notice"] = f"Booked {len(ids)} call(s). They are on your calendar now."
            st.rerun()
        if c2.button("Reject", use_container_width=True):
            record_decision("rejected", f"{result['request']['flow']}: {len(plan['visits'])} call(s) proposed")
            st.session_state["result"] = None
            st.session_state["history"].append("Rejected the proposed plan")
            st.session_state["notice"] = "Plan rejected: nothing was booked, and your day is unchanged."
            st.rerun()
    for b in (result.get("briefs") or {}).values():
        if status == "brief_ready":
            st.markdown(f"**Brief** ({b['status']})")
            st.code(b["text"], language=None, wrap_lines=True)
    if result.get("lanes"):
        st.markdown("**Who acts on what**")
        st.markdown(lane_chips(result["lanes"]), unsafe_allow_html=True)
    with st.expander("Agent trace"):
        for line in result.get("trace", []):
            st.text(line)


# ---------- sidebar ----------

with st.sidebar:
    st.title("🧭 Navinib")
    st.caption("An AI co-pilot for oncology field reps. Synthetic data only.")
    view = st.radio("View", ["Rep: my day", "Regional manager"], label_visibility="collapsed")
    times = [hhmm(m) for m in range(7 * 60, 18 * 60 + 1, 5)]
    now_min = to_min(st.selectbox("Time now (simulated)", times, index=times.index("08:20")))
    st.caption(f"Model: {get_llm().name}")
    if st.button("Reset the demo day"):
        new_session()
        st.rerun()


# ---------- rep view ----------

rep = con.execute("SELECT * FROM rep WHERE rep_id = ?", (REP,)).fetchone()
if view == "Rep: my day":
    st.subheader(f"{date.fromisoformat(DAY):%A %d %B %Y} · {rep['name']}, {rep['territory']}")
    schedule = schedule_lookup(con, REP, DAY)
    briefed = {r[0] for r in con.execute("SELECT event_id FROM call_brief")}
    result = st.session_state.get("result")
    proposed = result["plan"]["visits"] if result and result.get("status") == "awaiting_approval" else []

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("**Today**")
        st.markdown(calendar(schedule["events"], proposed, briefed, now_min), unsafe_allow_html=True)
        st.caption("Hover over a call for its summary, or open it below.")
        st.markdown("**Calls today**")
        items = [(e["start_min"], e["title"], e["start"], e["end"], e["call_type"], e["location_id"],
                  "Brief ready" if e["event_id"] in briefed else "Booked")
                 for e in schedule["events"] if e["kind"] == "call" and e["status"] != "cancelled"]
        items += [(to_min(v["start"]), f"Proposed: {v['full_name']}", v["start"], v["end"], v["call_type"],
                   v.get("location_id"), "Proposed, waiting for your approval") for v in proposed]
        for _, title, s0, e0, ctype, loc_id, status in sorted(items):
            c = call_summary(s0, e0, ctype, loc_id, status)
            with st.expander(f"{c['when']} · {c['icon']} {title}"):
                st.markdown(f"**How:** {c['mode']}  \n**Where:** {c['office'] or 'n/a'}"
                            + (f", {c['address']}" if c["address"] else "")
                            + (" (the rep calls from wherever they are)" if ctype == "phone" else "")
                            + f"  \n**Status:** {c['status']}")

    with right:
        points = []
        for e in schedule["events"]:
            if e["location_id"] and e["status"] != "cancelled" and e["call_type"] != "phone":
                loc = con.execute("SELECT lat, lon FROM location WHERE location_id = ?", (e["location_id"],)).fetchone()
                points.append({"lat": loc["lat"], "lon": loc["lon"], "color": "#1f5fa8"})
        for v in proposed:
            if v["call_type"] == "phone":
                continue
            loc = con.execute("SELECT lat, lon FROM location WHERE location_id = ?", (v["location_id"],)).fetchone()
            if loc:
                points.append({"lat": loc["lat"], "lon": loc["lon"], "color": "#1e7a3c"})
        if points:
            st.markdown("**Where** (blue booked, green proposed visits)")
            st.map(pd.DataFrame(points), color="color", size=60, height=240)
        st.markdown("**What do you need?**")
        a, b, c = st.columns(3)
        if a.button("Plan my day", use_container_width=True):
            st.session_state["result"] = run(con, {"flow": "morning", "rep_id": REP, "day": DAY, "now_min": now_min},
                                             col=col, drive=drive, log_path=st.session_state["log"])
            st.rerun()
        if b.button("Brief my next call", use_container_width=True):
            st.session_state["result"] = brief_next_call(con, REP, DAY, now_min, col=col, drive=drive,
                                                         log_path=st.session_state["log"])
            st.rerun()
        calls = [e for e in schedule["events"] if e["kind"] == "call" and e["status"] != "cancelled"
                 and e["end_min"] > now_min]
        with c.popover("A call cancelled", use_container_width=True):
            pick = st.selectbox("Which call?", calls, format_func=lambda e: f"{e['start']} {e['title']}")
            if st.button("Replan around it", disabled=not calls):
                st.session_state["result"] = run(con, {"flow": "cancellation", "rep_id": REP, "day": DAY,
                                                       "event_id": pick["event_id"], "now_min": now_min},
                                                 col=col, drive=drive, log_path=st.session_state["log"])
                st.rerun()
        ask = st.chat_input("Tell Navinib something, e.g. a question from a doctor")
        if ask:
            st.session_state["result"] = run(con, {"flow": "question", "rep_id": REP, "day": DAY,
                                                   "now_min": now_min, "message": ask},
                                             col=col, drive=drive, log_path=st.session_state["log"])
            st.rerun()
        notice = st.session_state.pop("notice", None)
        if notice:
            st.success(notice)
        if result:
            show_result(result)
        for h in st.session_state["history"][-3:]:
            st.caption(h)

# ---------- regional manager view ----------

else:
    mgr = con.execute("SELECT * FROM manager WHERE manager_id = ?", (rep["manager_id"],)).fetchone()
    st.subheader(f"{mgr['name']} · {mgr['region']} region")
    st.caption("Read-only and team level: goal coverage and a weekly sample of briefs, escalations and plan "
               "decisions. Not a live view of each rep.")
    month_start = date.fromisoformat(DAY).replace(day=1).isoformat()
    rows = con.execute(
        "SELECT p.provider_id, p.full_name, p.tier, la.open_count,"
        " (SELECT SUM(CASE call_type WHEN 'phone' THEN 0.5 ELSE 1 END) FROM visit v WHERE v.provider_id = p.provider_id"
        "   AND outcome = 'completed' AND date >= ? AND date < ?) AS done,"
        " (SELECT MAX(date) FROM visit v WHERE v.provider_id = p.provider_id AND outcome = 'completed'"
        "   AND call_type = 'face_to_face') AS last_visit"
        " FROM provider p LEFT JOIN lab_alert la USING (provider_id) WHERE p.rep_id = ?",
        (month_start, DAY, REP)).fetchall()
    df = pd.DataFrame([{
        "Provider": r["full_name"], "Tier": r["tier"], "Calls this month": r["done"] or 0,
        "Goal": config.CALL_GOALS[r["tier"]],
        "Days since visit": (date.fromisoformat(DAY) - date.fromisoformat(r["last_visit"])).days
        if r["last_visit"] else None,
        "Lab alerts": alert_display(r["open_count"] or 0)} for r in rows])
    df["On track"] = df["Calls this month"] >= df["Goal"]
    tier_a = df[df["Tier"] == "A"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tier A on goal", f"{int(tier_a['On track'].sum())} of {len(tier_a)}")
    k2.metric("Providers overdue (60+ days)", int((df["Days since visit"].fillna(999) >= 60).sum()))
    decisions = pd.read_sql_query("SELECT decision, detail FROM plan_decision", con)
    k3.metric("Plans approved / rejected", f"{(decisions['decision'] == 'accepted').sum()} / "
                                           f"{(decisions['decision'] == 'rejected').sum()}")
    try:
        esc = pd.read_sql_query("SELECT kind, source_ref, routed_to, created_at FROM escalation", con)
    except Exception:
        esc = pd.DataFrame(columns=["kind", "source_ref", "routed_to", "created_at"])
    k4.metric("Escalations", len(esc))

    st.markdown(f"**Goal coverage, {rep['territory']}** ({rep['name']})")
    st.dataframe(df.sort_values(["Tier", "On track", "Days since visit"], ascending=[True, True, False]),
                 hide_index=True, use_container_width=True, height=320)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weekly sample of briefs**")
        briefs = pd.read_sql_query("SELECT event_id, provider_id, status, written_min FROM call_brief", con)
        briefs["written"] = briefs["written_min"].map(hhmm)
        st.dataframe(briefs.drop(columns="written_min"), hide_index=True, use_container_width=True)
        st.markdown("**Rep plan decisions (override log)**")
        st.dataframe(decisions, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Escalations** (sent out of Navinib)")
        st.dataframe(esc, hide_index=True, use_container_width=True)
        runs = audit.read(st.session_state["log"])
        st.markdown(f"**Runs logged this session:** {len(runs)}")
        if runs:
            st.dataframe(pd.DataFrame([{"run": r["run_id"], "status": r["status"], "model": r["model"],
                                        "seconds": r["seconds"]} for r in runs]),
                         hide_index=True, use_container_width=True)
    st.caption(f"Policy version {config.POLICY['version']} · generated "
               f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
