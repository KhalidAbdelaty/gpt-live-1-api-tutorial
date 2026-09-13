# DataCamp Voice Learning Assistant

Companion code for the DataCamp tutorial [GPT-Live-1 API Tutorial: Build a Full-Duplex
Voice Assistant](https://www.datacamp.com/tutorial/gpt-live-1-api). A tutorial prototype
built with OpenAI's GPT-Live-1 API, not DataCamp's DataLab AI Assistant product.

The learner talks about a goal over WebRTC, GPT-Live-1 delegates research to a backend
model (`gpt-5.6-sol`) that searches real DataCamp resources with the hosted `web_search`
tool, and the learner can change a constraint mid-search before confirming a save through
a custom `save_learning_plan` function. Every piece here runs against the real API; there
is no mocked or replayed output.

## Features

- **Full-duplex voice** over WebRTC with Responses delegation to a backend model.
- **Mixed learning plans** of DataCamp courses, projects, tracks, and articles, matched to
  the learner's stated format preference.
- **Confirmed save**: a spoken save request only proposes a save; nothing is written until
  the learner clicks Confirm in the app. The backend is answered immediately so the
  conversation never blocks while waiting for that click.
- **Sessions and memory**: each conversation is an app-owned session, persisted to disk.
  Start fresh by default, or resume a past session from the picker to restore its plan and
  have the assistant continue where you left off ("welcome back").
- **End by voice**: say "end the conversation" and the call actually hangs up.
- **Chat-style transcript** with per-turn bubbles, live typing indicator, and inline chips
  for backend activity (searching, reading, saving) and system events.
- **Live dashboard** that follows the active session and updates on its own loop without
  interrupting the call.

## Project layout

- `server.py`: FastAPI application server. Creates the GPT-Live-1 session, configures
  Responses delegation, owns the save-confirmation gate and task-version state, and keeps
  per-session state (keyed by an app session id) persisted under `data/sessions/`. On
  reconnect it seeds the new live session with the prior plan and transcript for memory.
- `static/call_widget.html`: the real WebRTC client (microphone, `RTCPeerConnection`, the
  `oai-events` data channel), embedded inside the Streamlit app rather than shipped as a
  standalone page. Renders the chat transcript, the session picker, the Confirm-and-save
  box, and detects an end-of-call request from the transcript.
- `streamlit_app.py`: the reader-facing app. Hosts the call widget and a live session
  dashboard (learning plan, backend activity, event log) that polls the same server.
- `example_plan_request.py`: a standalone script that exercises the backend model and
  tools directly through the Responses API, without holding a microphone conversation.

## Running it

```bash
pip install -r requirements.txt
```

Add `OPENAI_API_KEY` to a `.env` file one level above this folder (see `.env.example`),
then start both processes:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
streamlit run streamlit_app.py
```

Open the Streamlit URL it prints, then in the call widget click Start conversation and
grant microphone access. Each page load starts a new session; to continue an earlier one,
choose it from the session picker before clicking Start. GPT-Live-1 access requires a paid
OpenAI usage tier; the free tier is not supported.

## License

Provided as-is for the DataCamp tutorial above. No warranty.
