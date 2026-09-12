"""Application server for the DataCamp Voice Learning Assistant.

A real FastAPI server, adapted from OpenAI's Python WebRTC quickstart
(gpt-live/00-open-first/04-webrtc.md), plus the application-owned pieces the
GPT-Live-1 docs explicitly leave to the app: task-version state, a save
confirmation gate, and a saved-plan store. Run with:

    uvicorn server:app --host 127.0.0.1 --port 8000 --reload

Load OPENAI_API_KEY from .env. Never expose it to the browser.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError, OpenAI

# .env lives at the project root (one level above this app/ folder), not
# inside app/, so the key is never duplicated next to the sample code.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="DataCamp Voice Learning Assistant server")

# Local demo only: Streamlit and the embedded widget run on localhost during
# development. Restrict this before deploying anywhere reachable by others.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

client = OpenAI(max_retries=0)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PLANS_FILE = DATA_DIR / "saved_plans.json"

BACKEND_MODEL = "gpt-5.6-sol"

LIVE_INSTRUCTIONS = """You are Sage, a calm, encouraging voice learning coach for DataCamp learners.
Speak warmly and naturally, at an unhurried pace. Be clear and direct, not overly cheerful.
If the learner is unsure or hesitates, give them time to think instead of filling the silence.

Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with the main response.

Interruption policy: Stop speaking when the learner interrupts. Listen to what they say.

Delegation policy:
Backend tools:
- learning_plan_research: search DataCamp resources and assemble a personalized learning plan.
- save_learning_plan: save the current plan after the learner confirms.

Delegate to the backend when:
- The learner states a learning goal, current skill level, or weekly time budget.
- A correction changes the plan already requested, such as a new priority or a skill to skip.
- The learner asks to save the plan.

Do not delegate to the backend when:
- The learner greets you, asks a clarifying question, or wants a result already given.
- You need a brief clarification to understand the request.

Delegate before giving an answer that depends on backend work.
Do not guess the plan while waiting. Do not say the plan is saved until the backend confirms it."""

BACKEND_INSTRUCTIONS = """You are the reasoning backend for a DataCamp Voice Learning Assistant.

## Voice conversation context
You are helping an assistant in a live voice conversation. Transcripts can contain
mistakes, unfinished phrases, and later corrections. Use the latest stated goal,
skill level, weekly time budget, and preferences. If a needed detail is still
unclear, say so instead of guessing.

## Task instructions
Search DataCamp resources with the web_search tool, preferring datacamp.com URLs.
Assemble a short, ordered learning plan (4 to 7 items) that fits the learner's
stated weekly time budget and skill level. Each item needs a title, a DataCamp
URL, and one short reason it belongs at that point in the sequence. Apply the
learner's latest constraints, including anything that tells you to skip or
reprioritize a topic.

When the learner asks to save the plan, call save_learning_plan with the full
current plan. Only call it after the learner has clearly asked to save.

## Return the result
Return the plan as a short spoken-friendly summary plus the structured list.
State whether the save succeeded once save_learning_plan returns a result.
Do not invent a successful save."""

SAVE_LEARNING_PLAN_TOOL = {
    "type": "function",
    "name": "save_learning_plan",
    "description": "Save the finalized personalized learning plan after the learner has confirmed it aloud.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "The learner's stated goal, e.g. 'become a data engineer'."},
            "weekly_hours": {"type": "number", "description": "Learner's stated weekly time budget in hours."},
            "items": {
                "type": "array",
                "description": "Ordered plan items.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["title", "url", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["goal", "weekly_hours", "items"],
        "additionalProperties": False,
    },
    "strict": True,
}


# In-memory task state for this local, single-user demo. A production app
# would key this by session/user and persist it, per the docs' guidance to
# keep authoritative state outside the model.
STATE: dict[str, Any] = {
    "session_id": None,
    "task_version": 0,
    "active_constraint": None,
    "transcript": [],
    "plan_draft": None,
    "pending_save": None,
    "saved_plan": None,
    "log": [],
    "steps": [],
}


def _add_step(kind: str, detail: str) -> None:
    """Record one backend activity step for the dashboard's live trace."""
    STATE["steps"].append({"t": time.time(), "kind": kind, "detail": detail})
    STATE["steps"] = STATE["steps"][-40:]


def _log(message: str) -> None:
    STATE["log"].append({"t": round(time.time(), 3), "message": message})
    STATE["log"] = STATE["log"][-200:]


@app.post("/api/session")
async def create_session(request: Request) -> dict:
    """Create a GPT-Live-1 session for the browser's WebRTC offer.

    Mirrors gpt-live/00-open-first/04-webrtc.md: the browser sends its SDP
    offer here, the server holds the project API key, and the resolved
    session id plus SDP answer go back to the browser unchanged.
    """
    body = await request.json()
    sdp = body.get("sdp")
    if not isinstance(sdp, str) or not sdp.strip():
        raise HTTPException(status_code=400, detail="An SDP offer is required")

    try:
        result = client.live.create(
            session={
                "model": "gpt-live-1",
                "instructions": LIVE_INSTRUCTIONS,
                "delegation": {
                    "type": "responses",
                    "responses": {
                        "model": BACKEND_MODEL,
                        "instructions": BACKEND_INSTRUCTIONS,
                        "tools": [
                            {
                                "type": "web_search",
                                "filters": {"allowed_domains": ["datacamp.com", "www.datacamp.com"]},
                            },
                            SAVE_LEARNING_PLAN_TOOL,
                        ],
                        "tool_choice": "auto",
                    },
                },
            },
            transport={"type": "webrtc", "sdp": sdp},
        )
    except APIError as error:
        raise HTTPException(status_code=error.status_code or 502, detail="Live session creation failed") from error

    STATE["session_id"] = result.session.id
    STATE["task_version"] = 0
    STATE["active_constraint"] = None
    STATE["transcript"] = []
    STATE["plan_draft"] = None
    STATE["pending_save"] = None
    STATE["saved_plan"] = None
    STATE["steps"] = []
    _log(f"session created: {result.session.id}")
    return result.model_dump()


@app.post("/api/state/event")
async def record_event(request: Request) -> dict:
    """Receive a lightweight event summary from the browser widget.

    The widget owns the live event stream (it holds the data channel).
    It forwards summarized state here so the server stays the source of
    truth for the task version, the plan draft, and the log this article's
    Streamlit demo reads.
    """
    event = await request.json()
    kind = event.get("kind")

    if kind == "transcript":
        STATE["transcript"].append({"speaker": event.get("speaker"), "text": event.get("text", "")})
        STATE["transcript"] = STATE["transcript"][-40:]
    elif kind == "new_delegation":
        STATE["task_version"] += 1
        STATE["active_constraint"] = event.get("constraint")
        _log(f"task version {STATE['task_version']}: {event.get('constraint')}")
        _add_step("delegate", event.get("constraint") or "Delegated to the backend")
    elif kind == "stale_result":
        _log(f"discarded a result from an older task version ({event.get('version')})")
        _add_step("stale", f"Discarded a result from task version {event.get('version')}")
    elif kind == "step":
        _add_step(event.get("step_kind", "info"), event.get("detail", ""))
    elif kind == "plan_draft":
        STATE["plan_draft"] = event.get("plan")
        _log("plan draft updated")
        _add_step("plan_ready", "Backend returned a learning plan")
    elif kind == "pending_save":
        STATE["pending_save"] = event.get("plan")
        _log("assistant proposed a save, awaiting confirmation")
        _add_step("propose_save", "Proposed saving the plan, awaiting confirmation")
    elif kind == "log":
        _log(str(event.get("message", "")))

    return {"ok": True}


@app.post("/api/save-plan")
async def save_plan(request: Request) -> dict:
    """Application-owned save. Executes only with an explicit confirmation flag.

    This endpoint, not the model, decides whether the side effect happens.
    Voice or a function call proposing a save is not authorization by itself.
    """
    body = await request.json()
    if not body.get("confirmed"):
        raise HTTPException(status_code=403, detail="Save requires explicit confirmation")

    plan = body.get("plan")
    if not isinstance(plan, dict):
        raise HTTPException(status_code=400, detail="A plan object is required")

    record = {
        "id": f"plan_{uuid.uuid4().hex[:12]}",
        "saved_at": time.time(),
        "task_version": STATE["task_version"],
        "plan": plan,
    }

    existing = []
    if PLANS_FILE.exists():
        existing = json.loads(PLANS_FILE.read_text(encoding="utf-8"))
    existing.append(record)
    PLANS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    STATE["saved_plan"] = record
    STATE["pending_save"] = None
    _log(f"plan saved: {record['id']}")
    _add_step("saved", f"Saved as {record['id']}")
    return {"status": "confirmed", "plan_id": record["id"]}


@app.get("/api/state")
async def get_state() -> dict:
    """Snapshot the Streamlit demo polls to render the session dashboard."""
    return STATE
