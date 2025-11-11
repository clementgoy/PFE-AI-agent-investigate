from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os, requests

ES = os.getenv("ES_URL", "http://localhost:9200")
AUTH = (os.getenv("ES_USER","sirenadmin"), os.getenv("ES_PASS","password"))
API_TOKEN = os.getenv("GRAPH_AGENT_TOKEN","devtoken")

app = FastAPI()

class Query(BaseModel):
    op: str
    parent_index: str | None = None
    child_index: str | None = None
    on: list[str] | None = None
    es_query: dict | None = None
    size: int | None = 50

def guard(h):
    if h != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Unauthorized")

@app.post("/graph/query")
def graph_query(body: Query, authorization: str = Header(None)):
    guard(authorization)
    if body.op == "lookup":
        q = body.es_query or {"match_all": {}}
        r = requests.post(f"{ES}/{body.parent_index}/_search",
                          json={"size": body.size, "query": q}, auth=AUTH, timeout=30)
        return r.json()
    if body.op == "join":
        if not (body.parent_index and body.child_index and body.on):
            raise HTTPException(400, "join needs parent_index, child_index, on")
        join = {"indices":[body.child_index], "on": body.on}
        if body.es_query:
            join["request"] = {"query": body.es_query}
        payload = {"query":{"join":join}}
        r = requests.post(f"{ES}/siren/{body.parent_index}/_search",
                          json=payload, auth=AUTH, timeout=60)
        return r.json()
    raise HTTPException(400, f"unsupported op {body.op}")
