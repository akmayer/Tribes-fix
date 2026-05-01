from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()


@app.get("/hello")
async def hello():
    return {"message": "hello from python"}


class QueryRequest(BaseModel):
    tick: int
    n_actions: int
    info: dict = {}


@app.post("/query")
async def query(req: QueryRequest):
    # Simple echo + dummy policy for a hello-world proof of concept
    return {"policy": "uniform", "received": {"tick": req.tick, "n_actions": req.n_actions, "info": req.info}}
