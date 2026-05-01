import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request

app = FastAPI()

CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/hello")
async def hello():
    return {"message": "hello from python"}


@app.post("/query")
async def query(req: Request):
    payload = await req.json()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    tick = payload.get("tick", "unknown")
    action_count = payload.get("available_action_count", "na")
    filename = f"capture_tick{tick}_actions{action_count}_{timestamp}.json"
    output_path = CAPTURE_DIR / filename

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)

    return {
        "status": "saved",
        "path": str(output_path),
        "policy": "uniform",
        "received_tick": tick,
        "available_action_count": action_count,
    }
