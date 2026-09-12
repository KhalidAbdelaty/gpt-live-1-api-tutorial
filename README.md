# DataCamp Voice Learning Assistant

Companion code for the DataCamp tutorial [GPT-Live-1 API Tutorial: Build a Full-Duplex
Voice Assistant](https://www.datacamp.com/tutorial/gpt-live-1-api). A tutorial prototype
built with OpenAI's GPT-Live-1 API, not DataCamp's DataLab AI Assistant product.

The learner talks about a goal over WebRTC, GPT-Live-1 delegates research to a backend
model (`gpt-5.6-sol`) that searches real DataCamp resources with the hosted `web_search`
tool, and the learner can change a constraint mid-search before confirming a save through
a custom `save_learning_plan` function. Every piece here runs against the real API; there
is no mocked or replayed output.

## Project layout

- `server.py`: FastAPI application server. Creates the GPT-Live-1 session, configures
  Responses delegation, and owns the save-confirmation gate and task-version state.
- `static/call_widget.html`: the real WebRTC client (microphone, `RTCPeerConnection`, the
  `oai-events` data channel), embedded inside the Streamlit app rather than shipped as a
  standalone page.
- `streamlit_app.py`: the reader-facing app. Hosts the call widget and a live session
  dashboard that polls the same server.
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

Open the Streamlit URL it prints, click Start conversation, and grant microphone access.
GPT-Live-1 access requires a paid OpenAI usage tier; the free tier is not supported.

## License

Provided as-is for the DataCamp tutorial above. No warranty.
