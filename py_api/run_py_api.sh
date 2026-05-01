#!/usr/bin/env bash
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
	echo "Detected 'uv' — running server with uv"
	if [ -f requirements.txt ]; then
		echo "Installing dependencies inside uv environment..."
		uv pip install -r requirements.txt
	fi
	uv run uvicorn app:app --host 127.0.0.1 --port 8000
else
	echo "'uv' not found — falling back to venv. Starting server on http://127.0.0.1:8000"
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	uvicorn app:app --host 127.0.0.1 --port 8000
fi
