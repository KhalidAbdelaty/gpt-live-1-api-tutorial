"""Streamlit demo for the DataCamp Voice Learning Assistant.

This is the reader-facing app for the GPT-Live-1 tutorial. It embeds the real
WebRTC call widget (call_widget.html) so the whole session runs inside one
page, then polls the same application server (server.py) this article's
code samples use, so the dashboard below reflects the actual session and
task state, not a replay.

Run:
    uvicorn server:app --host 127.0.0.1 --port 8000 --reload
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

SERVER = "http://127.0.0.1:8000"
REPO_URL = "https://github.com/KhalidAbdelaty/gpt-live-1-api-tutorial"
ASSETS = Path(__file__).parent / "assets"
DATACAMP_LOGO = ASSETS / "datacamp-logo.png"
OPENAI_LOGO = ASSETS / "openai-logo.png"
WIDGET_HTML = (Path(__file__).parent / "static" / "call_widget.html").read_text(encoding="utf-8")

FEATURES = [
    ("\U0001F3A4", "Full-duplex voice", "Real WebRTC audio to gpt-live-1. Interruptions and pauses handled by the model, not a VAD hack."),
    ("\U0001F9E9", "Backend delegation", "gpt-5.6-sol researches DataCamp resources with web_search while the conversation keeps going."),
    ("\U0001F501", "Live task versioning", "A new spoken constraint bumps the task version. Stale backend results are logged and discarded."),
    ("\u2705", "Confirmed save", "save_learning_plan only writes to disk after an explicit, app-level confirmation click."),
]

st.set_page_config(page_title="DataCamp Voice Learning Assistant", page_icon="\U0001F399\uFE0F", layout="wide")


@st.cache_data
def b64(path: Path) -> str | None:
    return base64.b64encode(path.read_bytes()).decode() if path.is_file() else None


dc_logo = b64(DATACAMP_LOGO)
oai_logo = b64(OPENAI_LOGO)

st.markdown(
    """
    <style>
      :root {
        --brand: #02904A;
        --brand-soft: #E7F7EE;
        --ink: #05192D;
        --muted: #5B6B7A;
        --surface: #FFFFFF;
        --line: #E4E7EA;
        --radius: 14px;
        --shadow: 0 10px 28px rgba(5, 25, 45, 0.06);
      }

      .stApp { background: var(--surface); }
      .block-container { padding-top: 1.6rem; max-width: 1180px; }
      [data-testid="stSidebar"] { display: none; }

      /* ---- top bar ---- */
      .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: .9rem; border-bottom: 1px solid var(--line); margin-bottom: 1.4rem;
      }
      .topbar-brand { display: flex; align-items: center; gap: .7rem; }
      .topbar-divider { width: 1px; height: 22px; background: var(--line); }
      .topbar-links { display: flex; align-items: center; gap: .6rem; }
      .pill-link {
        display: inline-flex; align-items: center; white-space: nowrap;
        font-size: .82rem; font-weight: 600; color: var(--ink);
        border: 1px solid var(--line); border-radius: 999px; padding: .4rem .9rem;
        text-decoration: none !important; background: var(--surface);
      }
      .pill-link:hover { border-color: var(--brand); color: var(--brand); }

      /* ---- hero ---- */
      .hero-badge {
        display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem; font-weight: 700;
        color: var(--brand); background: var(--brand-soft); border-radius: 999px;
        padding: .32rem .8rem; margin-bottom: .8rem;
      }
      .hero-title { font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 .5rem; line-height: 1.15; }
      .hero-sub { color: var(--muted); font-size: 1.02rem; line-height: 1.6; max-width: 760px; margin: 0 0 1.4rem; }

      /* ---- feature cards ---- */
      .feature-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
        padding: 1rem 1.05rem; height: 100%; box-shadow: var(--shadow);
      }
      .feature-icon { font-size: 1.3rem; }
      .feature-name { font-weight: 700; color: var(--ink); margin: .35rem 0 .25rem; font-size: .95rem; }
      .feature-desc { color: var(--muted); font-size: .84rem; line-height: 1.45; margin: 0; }

      /* ---- section cards ---- */
      .section-card {
        border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface);
        box-shadow: var(--shadow); padding: 1.1rem 1.2rem 1.3rem; height: 100%;
      }
      .section-title { font-weight: 700; color: var(--ink); font-size: 1.05rem; margin: 0 0 .2rem; }
      .section-sub { color: var(--muted); font-size: .84rem; margin: 0 0 .9rem; }

      /* ---- widget frame ---- */
      .widget-frame iframe {
        border: 1px solid var(--line) !important; border-radius: 12px !important;
        box-shadow: var(--shadow);
      }

      /* ---- status + metrics ---- */
      .status-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: 1rem; }
      .status-chip {
        border-radius: 12px; padding: .55rem .8rem; background: var(--brand-soft);
        border: 1px solid rgba(2, 144, 74, 0.18); min-width: 130px;
      }
      .status-chip .label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
      .status-chip .value { font-size: .95rem; font-weight: 700; color: var(--ink); margin-top: .1rem; word-break: break-all; }
      .status-chip.idle { background: #F7F7F5; border-color: var(--line); }

      /* ---- plan items ---- */
      .plan-item {
        border: 1px solid var(--line); border-radius: 10px; padding: .7rem .85rem;
        margin-bottom: .55rem; background: #FBFCFB;
      }
      .plan-item a { font-weight: 700; color: var(--ink); text-decoration: none; }
      .plan-item a:hover { color: var(--brand); }
      .plan-item .reason { color: var(--muted); font-size: .84rem; margin-top: .2rem; }

      /* ---- log ---- */
      .log-line {
        font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .78rem;
        color: var(--ink); border-left: 3px solid var(--brand);
        padding: .28rem .6rem; margin-bottom: .28rem; background: #F7F7F5; border-radius: 6px;
      }
      .empty-state { color: var(--muted); font-size: .88rem; padding: .6rem 0; }

      .stButton > button {
        border-radius: 10px; font-weight: 700; border: 1px solid var(--line);
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: .4rem; background: #F7F7F5; padding: .35rem; border-radius: 12px; border: 1px solid var(--line);
      }
      .stTabs [data-baseweb="tab"] { border-radius: 9px; font-weight: 700; color: var(--muted); }
      .stTabs [aria-selected="true"] { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ top bar
dc_img = f"<img src='data:image/png;base64,{dc_logo}' height='24'/>" if dc_logo else "DataCamp"
oai_img = f"<img src='data:image/png;base64,{oai_logo}' height='22' style='border-radius:50%;'/>" if oai_logo else ""

st.markdown(
    f"""
    <div class="topbar">
      <div class="topbar-brand">
        <a href="https://www.datacamp.com/blog" target="_blank" rel="noopener">{dc_img}</a>
        <div class="topbar-divider"></div>
        {oai_img}
      </div>
      <div class="topbar-links">
        <a class="pill-link" href="{REPO_URL}" target="_blank" rel="noopener">\U0001F4C2 Source on GitHub</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ hero
st.markdown(
    """
    <div class="hero-badge">\U0001F399\uFE0F Voice Learning Assistant &middot; tutorial prototype</div>
    <div class="hero-title">Talk through a learning goal, out loud</div>
    <p class="hero-sub">Speak a goal to GPT-Live-1, let a backend model search real DataCamp resources
    while the conversation keeps going, change your mind mid-search, and confirm before anything gets saved.
    Not DataCamp's DataLab AI Assistant product.</p>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
for col, (icon, name, desc) in zip(cols, FEATURES):
    with col:
        st.markdown(
            f"""<div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-name">{name}</div>
                <p class="feature-desc">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")

# ------------------------------------------------------------------ main area
left, right = st.columns([1, 1], gap="medium")

with left:
    st.markdown(
        """<div class="section-card">
            <div class="section-title">\U0001F3A7 Live conversation</div>
            <p class="section-sub">Real WebRTC over the local server.py, using the OpenAI API key from .env.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="widget-frame">', unsafe_allow_html=True)
    components.html(WIDGET_HTML, height=470, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    header_col, refresh_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            """<div class="section-title">\U0001F4CA Session dashboard</div>
            <p class="section-sub">Polls GET /api/state on the application server.</p>""",
            unsafe_allow_html=True,
        )
    with refresh_col:
        refresh = st.button("Refresh", use_container_width=True)

    try:
        state = requests.get(f"{SERVER}/api/state", timeout=2).json()
    except Exception:
        state = None

    if state is None:
        st.warning("Application server is not reachable. Start it with `uvicorn server:app --port 8000`.")
    else:
        session_id = state.get("session_id") or "not started"
        chip_class = "status-chip" if state.get("session_id") else "status-chip idle"
        st.markdown(
            f"""
            <div class="status-row">
              <div class="{chip_class}"><div class="label">Session</div><div class="value">{session_id}</div></div>
              <div class="status-chip"><div class="label">Task version</div><div class="value">{state.get("task_version", 0)}</div></div>
              <div class="status-chip"><div class="label">Constraint</div><div class="value">{state.get("active_constraint") or "none yet"}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_plan, tab_log = st.tabs(["Learning plan", "Event log"])

        with tab_plan:
            plan = state.get("saved_plan", {}).get("plan") if state.get("saved_plan") else None
            pending = state.get("pending_save")
            draft = state.get("plan_draft")
            shown = plan or pending or draft

            if state.get("saved_plan"):
                st.success(f"Saved as {state['saved_plan']['id']} at task version {state['saved_plan']['task_version']}")
            elif pending:
                st.info("Assistant proposed a save. Waiting for confirmation in the widget.")

            if shown:
                st.caption(f"Goal: {shown.get('goal', '')} &middot; {shown.get('weekly_hours', '?')} hours/week")
                for item in shown.get("items", []):
                    st.markdown(
                        f"""<div class="plan-item">
                            <a href="{item.get('url', '#')}" target="_blank" rel="noopener">{item.get('title', 'Untitled')}</a>
                            <div class="reason">{item.get('reason', '')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<p class="empty-state">No plan yet. Start a conversation and state a learning goal.</p>', unsafe_allow_html=True)

        with tab_log:
            log = state.get("log", [])
            if not log:
                st.markdown('<p class="empty-state">No events yet.</p>', unsafe_allow_html=True)
            for entry in log[-14:][::-1]:
                ts = time.strftime("%H:%M:%S", time.localtime(entry["t"]))
                st.markdown(f'<div class="log-line">[{ts}] {entry["message"]}</div>', unsafe_allow_html=True)

st.write("")
st.markdown(
    f"""<p style="color:#5B6B7A;font-size:.85rem;">
    Server code, widget, and this dashboard share the same task-version and save-confirmation logic
    described in the article. Full source on <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>.
    Built with <a href="https://www.datacamp.com/tutorial/streamlit" target="_blank" rel="noopener">Streamlit</a>.
    </p>""",
    unsafe_allow_html=True,
)
