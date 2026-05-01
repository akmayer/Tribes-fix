Python FastAPI bridge for Tribes

Quick start:

1. Using `uv` (preferred if installed):

```bash
uv pip install -r requirements.txt
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

2. Fallback (if you don't have `uv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_py_api.sh
```

Endpoints:
- `GET /hello` — returns a simple JSON hello message
- `POST /query` — accepts JSON with `tick` and `n_actions` and returns a dummy policy

The Java `RandomAgent` calls this endpoint at runtime as a proof-of-concept; if the server is not available the agent falls back to random actions.
