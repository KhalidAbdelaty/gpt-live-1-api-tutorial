"""Application server for the DataCamp Voice Learning Assistant.

A real FastAPI server, adapted from OpenAI's Python WebRTC quickstart
(gpt-live/00-open-first/04-webrtc.md), plus the application-owned pieces the
GPT-Live-1 docs explicitly leave to the app: task-version state, a save
confirmation gate, and a saved-plan store. Run with:

    uvicorn server:app --host 127.0.0.1 --port 8000 --reload

Load OPENAI_API_KEY from .env. Never expose it to the browser.

## Application sessions vs. live sessions

A GPT-Live session (the `live_session_id`) is created fresh on every WebRTC
connect and cannot be reopened. So this server keeps an application-owned
**app session** instead: a stable id the browser stores in localStorage and
sends with every request. All plan, task-version, and log state is keyed by
that app session and persisted to disk, so plans from different sessions never
mix, and closing and reopening the same app session restores its state.
"""

from __future__ import annotations

import json
import re
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
SESSIONS_DIR = DATA_DIR / "sessions"
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
# One append-only file with every confirmed save across all app sessions, plus
# one file per app session holding that session's full state.
PLANS_FILE = DATA_DIR / "saved_plans.json"

BACKEND_MODEL = "gpt-5.6-sol"

LIVE_INSTRUCTIONS = """You are Sage, a warm, encouraging voice learning coach for DataCamp learners.
Speak naturally at an unhurried pace. Be clear and direct, not overly cheerful.
If the learner hesitates, give them time to think instead of filling every silence.

Do not speak first. Wait for the learner to speak. On a brand new conversation, when they
greet you, reply naturally and in that first reply briefly introduce yourself: you are
Sage, a learning coach for DataCamp who helps turn a goal into a short, personalized plan
of DataCamp courses, projects, tracks, and articles, and saves it once they confirm.

If your context already includes an earlier conversation with this learner, you are
resuming. Do NOT introduce yourself again and do NOT repeat who you are or what you do.
Just welcome them back warmly, for example "Welcome back," and briefly recap where you
left off.

Backchannel policy: use moderate backchannels; acknowledge without competing with the response.
Interruption policy: stop speaking when the learner interrupts, and listen.

Understanding the learner: before finalizing a plan, make sure you know their goal,
their level, their weekly time, and how they like to learn. Ask briefly whether they
prefer hands-on projects, reading articles, video-style courses, or a guided track,
and use that preference.

Delegation policy:
Backend tools:
- learning_plan_research: search DataCamp resources and assemble a personalized plan.
- save_learning_plan: save the current plan after the learner confirms.
- end_conversation: hang up the call when the learner asks to end it.

Delegate to the backend when:
- The learner states or changes a goal, skill level, weekly time, or format preference.
- A correction changes the plan already requested.
- The learner asks to save the plan.
- The learner asks to end, stop, or hang up the conversation.

Do not delegate for greetings, small clarifications, or a result already given.

When you delegate, first say one short, natural line so there is no dead air, such as
"Let me pull together a few good options." Do not guess the plan while waiting.

Saving: proposing a save only asks the app to confirm; it does not save anything. Do
not say the plan is saved until a save tool result says it was saved. If a tool result
says it is awaiting confirmation, tell the learner the Confirm and save button is in the
app next to this conversation, then wait. The learner saying "okay", "makes sense", or
"sounds good" is NOT confirmation. Only an actual saved result from the app counts, so
never claim the plan is saved based on the conversation alone.

Ending: you cannot hang up by yourself, but the app can. When the learner clearly asks
to end, stop, or say goodbye, give a short goodbye and let the backend end the call.

After a save, keep the conversation open: acknowledge briefly and ask what they want next."""

BACKEND_INSTRUCTIONS = """You are the reasoning backend for a DataCamp Voice Learning Assistant.

## Voice conversation context
You are helping an assistant in a live voice conversation. Transcripts can contain
mistakes, unfinished phrases, and later corrections. Use the latest stated goal,
skill level, weekly time budget, and format preference. If a needed detail is still
unclear, say so instead of guessing.

## Task instructions
Search DataCamp resources with the web_search tool, preferring datacamp.com URLs.
Assemble a short, ordered plan (4 to 7 items) that fits the learner's weekly time and
level. Include a MIX of resource types, not only courses: use courses, hands-on
projects, guided tracks, and articles or tutorials, weighted toward the learner's
stated preference (more projects for a hands-on learner, more articles for a reader).
Each item needs a title, a DataCamp URL, one short reason, and a type that is exactly
one of course, project, track, or article. Apply the learner's latest constraints,
including anything that tells you to skip or reprioritize a topic.

## Saving
When the learner asks to save, call save_learning_plan once with the full current plan.
If the tool result says it is awaiting confirmation, do not call it again; ask the
learner to confirm in the app and wait. Only state that the plan is saved after a tool
result says it was saved.

## Ending
When the learner clearly asks to end, stop, hang up, or say goodbye, call end_conversation.
Do not call it otherwise.

## Return the result
Return a short spoken-friendly summary plus the structured list. Use confirmed values.
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
                        "type": {
                            "type": "string",
                            "enum": ["course", "project", "track", "article"],
                            "description": "The kind of DataCamp resource this item is.",
                        },
                    },
                    "required": ["title", "url", "reason", "type"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["goal", "weekly_hours", "items"],
        "additionalProperties": False,
    },
    "strict": True,
}

# A client-actionable function the learner can trigger by voice. The backend calls it
# when the learner asks to end; the browser widget executes the actual hang-up, because
# the model cannot close a live session by itself.
END_CONVERSATION_TOOL = {
    "type": "function",
    "name": "end_conversation",
    "description": "End and hang up the voice session. Call only when the learner clearly asks to end, stop, hang up, or say goodbye.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}

# App session ids come from the browser (localStorage). Validate them strictly:
# they become file names, so reject anything that is not a short, safe token.
APP_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# In-memory cache of every app session, keyed by app_session_id. Each value is
# the authoritative per-session state; it is also written to disk so it survives
# a reconnect or a server restart. The docs push authoritative state outside the
# model, so this is where task version, plans, and logs live.
SESSIONS: dict[str, dict[str, Any]] = {}
# The app session the dashboard should show by default: the one most recently
# connected or active. Persisted so the dashboard can find it after a restart.
ACTIVE_SESSION_ID: str | None = None


def _new_session_state(app_id: str) -> dict[str, Any]:
    now = time.time()
    return {
        "app_session_id": app_id,
        "live_session_id": None,
        "session_id": None,  # alias kept for the dashboard's status chip
        "created_at": now,
        "last_active": now,
        "task_version": 0,
        "active_constraint": None,
        "transcript": [],
        "plan_draft": None,
        "pending_save": None,
        "saved_plan": None,
        "saved_plans": [],
        "log": [],
        "steps": [],
    }


def _session_path(app_id: str) -> Path:
    return SESSIONS_DIR / f"{app_id}.json"


def _persist(app_id: str) -> None:
    state = SESSIONS.get(app_id)
    if state is None:
        return
    try:
        _session_path(app_id).write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        # Persistence is best-effort for this local demo; a failed write must
        # not break the live conversation.
        pass


def _load_from_disk(app_id: str) -> dict[str, Any] | None:
    path = _session_path(app_id)
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Backfill any keys added since this file was written.
    base = _new_session_state(app_id)
    base.update(loaded)
    base["app_session_id"] = app_id
    return base


def _get_or_create(app_id: str) -> dict[str, Any]:
    if app_id in SESSIONS:
        return SESSIONS[app_id]
    SESSIONS[app_id] = _load_from_disk(app_id) or _new_session_state(app_id)
    return SESSIONS[app_id]


def _set_active(app_id: str) -> None:
    # Active session is in-memory only, so a server restart clears it and the dashboard
    # starts clean. It becomes set again as soon as a conversation connects.
    global ACTIVE_SESSION_ID
    ACTIVE_SESSION_ID = app_id


def _validate_app_id(app_id: Any) -> str:
    if not isinstance(app_id, str) or not APP_SESSION_RE.match(app_id):
        raise HTTPException(status_code=400, detail="A valid app_session_id is required")
    return app_id


def _add_step(app_id: str, kind: str, detail: str) -> None:
    """Record one backend activity step for the dashboard's live trace."""
    state = _get_or_create(app_id)
    state["steps"].append({"t": time.time(), "kind": kind, "detail": detail})
    state["steps"] = state["steps"][-40:]


def _log(app_id: str, message: str) -> None:
    state = _get_or_create(app_id)
    state["log"].append({"t": round(time.time(), 3), "message": message})
    state["log"] = state["log"][-200:]


def _current_plan(state: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The best available plan for this session and a human status for it.

    A plan may be saved, only proposed (awaiting confirmation), or an earlier draft.
    We recall whichever exists so memory works even if the learner never clicked save.
    """
    record = state.get("saved_plan")
    if record and isinstance(record.get("plan"), dict):
        return record["plan"], "already saved"
    if isinstance(state.get("pending_save"), dict):
        return state["pending_save"], "proposed but not yet saved"
    if isinstance(state.get("plan_draft"), dict):
        return state["plan_draft"], "drafted, not saved"
    return None, ""


def _plan_summary_text(state: dict[str, Any]) -> str | None:
    """A compact recap of the learner's current plan, for seeding a new session."""
    plan, status = _current_plan(state)
    if not plan or not plan.get("items"):
        return None
    lines = []
    for index, item in enumerate(plan.get("items", []), start=1):
        kind = item.get("type", "resource")
        lines.append(f"{index}. {item.get('title', '')} [{kind}] - {item.get('reason', '')}")
    return (
        f"The plan so far ({status}). Goal: {plan.get('goal', '(unknown)')}. "
        f"Weekly time: {plan.get('weekly_hours', '?')} hours.\n"
        f"Items ({len(plan.get('items', []))}):\n" + "\n".join(lines)
    )


def _reconstruct_turns(transcript: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Merge choppy transcript deltas back into readable per-speaker turns.

    Transcript events are fragments, not full turns, so consecutive fragments from the
    same speaker are concatenated (as received, per the docs) into one turn.
    """
    turns: list[dict[str, str]] = []
    for entry in transcript:
        role = "assistant" if entry.get("speaker") == "assistant" else "user"
        text = entry.get("text", "")
        if turns and turns[-1]["role"] == role:
            turns[-1]["text"] += text
        else:
            turns.append({"role": role, "text": text})
    cleaned = []
    for turn in turns:
        collapsed = " ".join(turn["text"].split()).strip()
        if collapsed:
            cleaned.append({"role": turn["role"], "text": collapsed})
    return cleaned


def _seed_input(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Build the session.input history that restores memory on reconnect.

    GPT-Live cannot reopen a closed live session, so its model has no memory of a prior
    connection on its own. Seeding the new session's `input` with a recap plus the recent
    turns is what actually makes the assistant remember, instead of saying it cannot
    recall past sessions.
    """
    summary = _plan_summary_text(state)
    turns = _reconstruct_turns(state.get("transcript", []))
    if not summary and not turns:
        return None

    dev_text = (
        "You are resuming an earlier voice session with this same learner. You DO have "
        "this context, so never say you cannot remember earlier sessions or only see this "
        "chat. You already met and introduced yourself before, so do NOT introduce yourself "
        "again and do NOT repeat who you are or what you do. Open with a brief 'Welcome back' "
        "and a short recap, then continue where you left off. Use the recap and the recent "
        "turns below to recall specifics you discussed, such as any article you read "
        "together, its author or date, and decisions you made, not only the plan."
    )
    if summary:
        dev_text += "\n\n" + summary

    messages: list[dict[str, Any]] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": dev_text}]}
    ]
    # Include as many recent turns as fit in a character budget (well under the
    # session.input token limit), newest kept, so short replies and details still
    # make sense on resume.
    selected: list[dict[str, str]] = []
    budget = 7000
    for turn in reversed(turns):
        if selected and budget - len(turn["text"]) < 0:
            break
        selected.append(turn)
        budget -= len(turn["text"])
    selected.reverse()
    for turn in selected:
        if turn["role"] == "assistant":
            messages.append(
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": turn["text"]}]}
            )
        else:
            messages.append(
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": turn["text"]}]}
            )
    return messages


# A fresh server run starts with a clean dashboard on purpose: we do NOT preload old
# sessions into memory or restore the active pointer, so the UI shows nothing stale until
# a conversation actually starts. The per-session files stay on disk, so reconnecting with
# the same app_session_id still restores that session's memory on demand via _get_or_create.


@app.post("/api/session")
async def create_session(request: Request) -> dict:
    """Create a GPT-Live-1 session for the browser's WebRTC offer.

    Mirrors gpt-live/00-open-first/04-webrtc.md: the browser sends its SDP
    offer here, the server holds the project API key, and the resolved
    session id plus SDP answer go back to the browser unchanged.

    The browser also sends its stable app_session_id. Reconnecting an existing
    app session restores its plans and log (memory) instead of wiping them; only
    a stale pending-save proposal from the previous connection is cleared.
    """
    body = await request.json()
    sdp = body.get("sdp")
    if not isinstance(sdp, str) or not sdp.strip():
        raise HTTPException(status_code=400, detail="An SDP offer is required")
    app_id = _validate_app_id(body.get("app_session_id"))

    # Read state before creating the session so a reconnect can seed the new live
    # session with the previous conversation and plan (real memory for the model).
    state = _get_or_create(app_id)
    seed = _seed_input(state)
    reconnected = bool(seed or state.get("saved_plans") or state.get("task_version"))

    session_config: dict[str, Any] = {
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
                    END_CONVERSATION_TOOL,
                ],
                "tool_choice": "auto",
            },
        },
    }
    if seed:
        session_config["input"] = seed

    try:
        result = client.live.create(
            session=session_config,
            transport={"type": "webrtc", "sdp": sdp},
        )
    except APIError as error:
        raise HTTPException(status_code=error.status_code or 502, detail="Live session creation failed") from error

    state["live_session_id"] = result.session.id
    state["session_id"] = result.session.id
    state["last_active"] = time.time()
    # Clear only the transient proposal tied to the previous connection.
    state["pending_save"] = None
    if reconnected:
        _log(app_id, f"reconnected app session {app_id} to live session {result.session.id}")
        detail = "Reconnected and restored the plan into the assistant's memory" if seed else "Reconnected; restored history"
        _add_step(app_id, "responded", detail)
    else:
        _log(app_id, f"session created: {result.session.id}")
    _set_active(app_id)
    _persist(app_id)
    return result.model_dump()


@app.post("/api/state/event")
async def record_event(request: Request) -> dict:
    """Receive a lightweight event summary from the browser widget.

    The widget owns the live event stream (it holds the data channel).
    It forwards summarized state here so the server stays the source of
    truth for the task version, the plan draft, and the log this article's
    Streamlit demo reads. Every event carries the app_session_id so state
    lands in the right session and never mixes with another one.
    """
    event = await request.json()
    app_id = _validate_app_id(event.get("app_session_id"))
    state = _get_or_create(app_id)
    kind = event.get("kind")

    if kind == "transcript":
        state["transcript"].append({"speaker": event.get("speaker"), "text": event.get("text", "")})
        # Keep a long tail of fragments so a reconnect can recall the details of what
        # actually happened, not just the final plan. Fragments are tiny (often one
        # word), so this stays small on disk.
        state["transcript"] = state["transcript"][-600:]
    elif kind == "new_delegation":
        state["task_version"] += 1
        state["active_constraint"] = event.get("constraint")
        _log(app_id, f"task version {state['task_version']}: {event.get('constraint')}")
        _add_step(app_id, "delegate", event.get("constraint") or "Delegated to the backend")
    elif kind == "stale_result":
        _log(app_id, f"discarded a result from an older task version ({event.get('version')})")
        _add_step(app_id, "stale", f"Discarded a result from task version {event.get('version')}")
    elif kind == "step":
        _add_step(app_id, event.get("step_kind", "info"), event.get("detail", ""))
    elif kind == "plan_draft":
        state["plan_draft"] = event.get("plan")
        _log(app_id, "plan draft updated")
        _add_step(app_id, "plan_ready", "Backend returned a learning plan")
    elif kind == "pending_save":
        state["pending_save"] = event.get("plan")
        _log(app_id, "assistant proposed a save, awaiting confirmation")
        _add_step(app_id, "propose_save", "Proposed saving the plan, awaiting confirmation")
    elif kind == "log":
        _log(app_id, str(event.get("message", "")))

    state["last_active"] = time.time()
    _set_active(app_id)
    _persist(app_id)
    return {"ok": True}


@app.post("/api/save-plan")
async def save_plan(request: Request) -> dict:
    """Application-owned save. Executes only with an explicit confirmation flag.

    This endpoint, not the model, decides whether the side effect happens.
    Voice or a function call proposing a save is not authorization by itself.
    The save is written into the calling app session's own plan history, so
    saves from different sessions stay separate.
    """
    body = await request.json()
    if not body.get("confirmed"):
        raise HTTPException(status_code=403, detail="Save requires explicit confirmation")
    app_id = _validate_app_id(body.get("app_session_id"))

    plan = body.get("plan")
    if not isinstance(plan, dict):
        raise HTTPException(status_code=400, detail="A plan object is required")

    state = _get_or_create(app_id)
    record = {
        "id": f"plan_{uuid.uuid4().hex[:12]}",
        "saved_at": time.time(),
        "app_session_id": app_id,
        "live_session_id": state.get("live_session_id"),
        "task_version": state["task_version"],
        "plan": plan,
    }

    # Append to this session's own history and to the global cross-session file.
    state["saved_plans"].append(record)
    state["saved_plan"] = record
    state["pending_save"] = None

    existing = []
    if PLANS_FILE.exists():
        try:
            existing = json.loads(PLANS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = []
    existing.append(record)
    PLANS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    state["last_active"] = time.time()
    _log(app_id, f"plan saved: {record['id']}")
    _add_step(app_id, "saved", f"Saved as {record['id']}")
    _set_active(app_id)
    _persist(app_id)
    return {"status": "confirmed", "plan_id": record["id"]}


def _placeholder_state() -> dict:
    """State returned when no app session exists yet, so the dashboard renders."""
    empty = _new_session_state("none")
    empty["app_session_id"] = None
    empty["created_at"] = None
    empty["last_active"] = None
    return empty


@app.get("/api/state")
async def get_state(session: str | None = None) -> dict:
    """Snapshot the Streamlit demo polls to render the session dashboard.

    Pass ?session=<app_session_id> to view a specific session; otherwise the
    most recently active session is returned.
    """
    app_id = session or ACTIVE_SESSION_ID
    if not app_id or not APP_SESSION_RE.match(app_id):
        return _placeholder_state()
    if app_id not in SESSIONS and _load_from_disk(app_id) is None:
        return _placeholder_state()
    return _get_or_create(app_id)


@app.get("/api/sessions")
async def list_sessions() -> dict:
    """List app sessions (in memory and persisted on disk) for the session pickers.

    Disk sessions are included so a past session can be chosen and resumed even after a
    server restart, while the dashboard's default active view still starts clean (the
    active pointer is not restored on startup).
    """
    ids = set(SESSIONS.keys())
    for path in SESSIONS_DIR.glob("*.json"):
        if APP_SESSION_RE.match(path.stem):
            ids.add(path.stem)

    summaries = []
    for app_id in ids:
        state = SESSIONS.get(app_id) or _load_from_disk(app_id)
        if state is None:
            continue
        summaries.append(
            {
                "app_session_id": app_id,
                "live_session_id": state.get("live_session_id"),
                "created_at": state.get("created_at"),
                "last_active": state.get("last_active"),
                "task_version": state.get("task_version", 0),
                "saved_count": len(state.get("saved_plans", [])),
                "has_draft": bool(state.get("plan_draft")),
                "is_active": app_id == ACTIVE_SESSION_ID,
            }
        )
    summaries.sort(key=lambda s: s.get("last_active") or 0, reverse=True)
    return {"active": ACTIVE_SESSION_ID, "sessions": summaries}
