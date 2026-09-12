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
ASSETS = Path(__file__).parent / "assets"
DATACAMP_LOGO = ASSETS / "datacamp-logo.png"
OPENAI_LOGO = ASSETS / "openai-logo.png"
WIDGET_HTML = (Path(__file__).parent / "static" / "call_widget.html").read_text(encoding="utf-8")


@st.cache_data
def b64(path: Path) -> str | None:
    return base64.b64encode(path.read_bytes()).decode() if path.is_file() else None


st.set_page_config(page_title="DataCamp Voice Learning Assistant", page_icon="\U0001F399\uFE0F", layout="wide")

dc_logo = b64(DATACAMP_LOGO)
oai_logo = b64(OPENAI_LOGO)

st.markdown(
    """
    <style>
      :root {
        --dc-green: #03EF62;
        --dc-navy: #05192D;
        --card-bg: #F7F7F5;
        --card-border: #E6E4DD;
      }
      .stApp { background: #FFFFFF; }
      .block-container { padding-top: 2rem; max-width: 1150px; }
      p, .app-sub, .stCaption, .stMarkdown { color: #05192D; }
      .header-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: .5rem; }
      .app-title { font-size: 1.9rem; font-weight: 800; color: var(--dc-navy); margin: .4rem 0 0; }
      .app-sub { color: #4b5563; max-width: 720px; }
      .oai-badge {
        display: inline-flex; align-items: center; gap: .4rem;
        background: var(--dc-navy); color: #fff; border-radius: 999px;
        padding: .3rem .8rem; font-size: .78rem; font-weight: 700;
      }
      .state-card {
        background: var(--card-bg); border: 1px solid var(--card-border);
        border-radius: 14px; padding: .9rem 1rem; margin-bottom: .8rem;
      }
      .state-card h4 { margin: 0 0 .4rem; color: var(--dc-navy); }
      .log-line {
        font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .8rem;
        color: var(--dc-navy); border-left: 3px solid var(--dc-green);
        padding: .2rem .5rem; margin: .15rem 0; background: #fff; border-radius: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="header-row">
      <a href="https://www.datacamp.com/blog" target="_blank" rel="noopener">
        {f"<img src='data:image/png;base64,{dc_logo}' height='28'/>" if dc_logo else "DataCamp"}
      </a>
      <span class="oai-badge">
        {f"<img src='data:image/png;base64,{oai_logo}' height='16' style='vertical-align:-3px;border-radius:50%;'/>" if oai_logo else ""}
        &nbsp;Built on GPT-Live-1
      </span>
    </div>
    <div class="app-title">DataCamp Voice Learning Assistant (tutorial prototype)</div>
    <p class="app-sub">Talk about a learning goal, hear the assistant delegate research to a backend model,
    change a constraint while it works, and confirm before it saves the plan. This is a tutorial sample,
    not DataCamp's DataLab AI Assistant product.</p>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 1])

with left:
    st.markdown("#### Talk to the assistant")
    st.caption("Runs the real call_widget.html over WebRTC against the local server.py, using the OpenAI API key from .env.")
    components.html(WIDGET_HTML, height=520, scrolling=True)

with right:
    st.markdown("#### Session dashboard")
    st.caption("Polls GET /api/state on the application server. Refresh to pull the latest snapshot.")
    if st.button("Refresh state"):
        st.rerun()

    try:
        state = requests.get(f"{SERVER}/api/state", timeout=2).json()
    except Exception:
        state = None

    if state is None:
        st.warning("Application server is not reachable. Start it with `uvicorn server:app --port 8000`.")
    else:
        st.markdown(
            f"""
            <div class="state-card">
              <h4>Task state</h4>
              Session: <code>{state.get("session_id") or "not started"}</code><br/>
              Task version: <b>{state.get("task_version", 0)}</b><br/>
              Active constraint: {state.get("active_constraint") or "none yet"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        plan_draft = state.get("plan_draft")
        if plan_draft:
            st.markdown("<div class='state-card'><h4>Current plan draft</h4></div>", unsafe_allow_html=True)
            st.json(plan_draft)

        pending = state.get("pending_save")
        if pending:
            st.markdown("<div class='state-card'><h4>Awaiting confirmation</h4></div>", unsafe_allow_html=True)
            st.json(pending)

        saved = state.get("saved_plan")
        if saved:
            st.success(f"Saved plan {saved['id']} at task version {saved['task_version']}")
            st.json(saved["plan"])

        st.markdown("<div class='state-card'><h4>Event log</h4></div>", unsafe_allow_html=True)
        for entry in state.get("log", [])[-12:][::-1]:
            st.markdown(f"<div class='log-line'>[{time.strftime('%H:%M:%S', time.localtime(entry['t']))}] {entry['message']}</div>", unsafe_allow_html=True)

st.caption(
    "Server code, widget, and this Streamlit app share the same task-version and save-confirmation logic "
    "described in the article. See the [Streamlit tutorial](https://www.datacamp.com/tutorial/streamlit) "
    "for the framework this demo is built on."
)
