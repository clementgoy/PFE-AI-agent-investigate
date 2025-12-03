
# FastAPI + endpoints bas niveau + /chat orchestré par LLM + délégation au SpecialistAgent.

import os, json, requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi import Query as Q
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    pass

from agents.specialist import SpecialistAgent

ES = os.getenv("ES_URL", "http://localhost:9200")
AUTH = (os.getenv("ES_USER", "sirenadmin"), os.getenv("ES_PASS", "password"))
API_TOKEN = os.getenv("GRAPH_AGENT_TOKEN", "devtoken")
VERIFY_TLS = os.getenv("ES_VERIFY", "false").lower() == "true"
CHAT_MODE = os.getenv("CHAT_MODE", "llm").lower()

app = FastAPI()

class Query(BaseModel):
    op: str
    parent_index: str | None = None
    child_index: str | None = None
    on: list[str] | None = None
    es_query: dict | None = None
    size: int | None = 50
    join_type: str | None = None

def guard(h: str | None):
    if h != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Unauthorized")

def es_get(path: str, **kwargs):
    try:
        r = requests.get(f"{ES}{path}", auth=AUTH, verify=VERIFY_TLS,
                         timeout=kwargs.pop("timeout", 30), **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"ES GET {path} failed: {e}")

def es_post(path: str, json=None, **kwargs):
    try:
        r = requests.post(f"{ES}{path}", auth=AUTH, json=json, verify=VERIFY_TLS,
                          timeout=kwargs.pop("timeout", 60), **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"ES POST {path} failed: {e}")

@app.get("/health")
def health(authorization: str = Header(None)):
    guard(authorization)
    try:
        info = es_get("/", timeout=5)
    except HTTPException as e:
        info = {"error": e.detail}
    return {"mode": CHAT_MODE, "es_url": ES, "verify_tls": VERIFY_TLS,
            "es": info, "openai_available": OPENAI_AVAILABLE}

@app.get("/graph/indices")
def list_indices(authorization: str = Header(None)):
    guard(authorization)
    return es_get("/_cat/indices?format=json", timeout=15)

@app.get("/graph/mapping")
def get_mapping(index: str = Q(..., min_length=1), authorization: str = Header(None)):
    guard(authorization)
    return es_get(f"/{index}/_mapping?pretty", timeout=30)

@app.post("/graph/query")
def graph_query(body: Query, authorization: str = Header(None)):
    guard(authorization)
    if body.op == "lookup":
        if not body.parent_index:
            raise HTTPException(400, "lookup needs parent_index")
        return es_post(f"/{body.parent_index}/_search",
                       json={"size": body.size or 50, "query": body.es_query or {"match_all": {}}}, timeout=30)
    if body.op == "join":
        if not (body.parent_index and body.child_index and body.on and len(body.on) == 2):
            raise HTTPException(400, "join needs parent_index, child_index, on=[child_key,parent_key]")
        join = {"indices": [body.child_index], "on": body.on}
        if body.join_type: join["type"] = body.join_type
        if body.es_query:  join["request"] = {"query": body.es_query}
        return es_post(f"/siren/{body.parent_index}/_search",
                       json={"size": body.size or 50, "query": {"join": join}}, timeout=60)
    raise HTTPException(400, f"unsupported op {body.op}")

def local_plan_summary() -> str:
    try:
        es_get("/_cat/indices?format=json", timeout=10)
    except HTTPException:
        return "Elasticsearch hors service."
    res = es_post("/siren/company/_search",
                  json={"size": 10, "query": {"join": {
                      "indices": ["investment"], "on": ["companies", "id"], "request": {"query": {"match_all": {}}}
                  }}}, timeout=60)
    hits = res.get("hits", {}).get("hits", []) or []
    if not hits:
        return "Aucun résultat via investment→company (on=['companies','id'])."
    items = []
    for h in hits:
        s = h.get("_source", {})
        items.append(f"- {s.get('label') or s.get('id')}")
    return "Top résultats :\n" + "\n".join(items)

@app.post("/chat")
async def chat(request: Request, authorization: str = Header(None)):
    guard(authorization)

    # Récupération du prompt (json ou txt)
    prompt = None
    try:
        data = await request.json()
        if isinstance(data, dict):
            prompt = data.get("prompt")
    except Exception:
        pass
    if not prompt:
        raw = (await request.body() or b"").decode("utf-8", "ignore").strip()
        if raw and not raw.startswith("{"):
            prompt = raw
    if not prompt:
        prompt = request.query_params.get("prompt") or request.query_params.get("q")
    if not prompt:
        raise HTTPException(400, 'No prompt provided. Send JSON {"prompt":"..."}, text/plain, or ?prompt=...')

    if CHAT_MODE != "llm":
        return {"answer": local_plan_summary(), "mode": "local"}

    if not OPENAI_AVAILABLE:
        raise HTTPException(503, "OpenAI SDK not installed. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not set in environment.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    TOOLS = [
      {"type":"function","function":{
        "name":"graph_indices","description":"Lister les indices",
        "parameters":{"type":"object","properties":{}}
      }},
      {"type":"function","function":{
        "name":"graph_mapping","description":"Mapping d'un index",
        "parameters":{"type":"object","properties":{"index":{"type":"string"}},"required":["index"]}
      }},
      {"type":"function","function":{
        "name":"graph_query","description":"lookup / join Federate",
        "parameters":{"type":"object","properties":{
          "op":{"type":"string","enum":["lookup","join"]},
          "parent_index":{"type":"string"},
          "child_index":{"type":"string"},
          "on":{"type":"array","items":{"type":"string"}},
          "es_query":{"type":"object"},
          "size":{"type":"integer"}
        },"required":["op","parent_index","es_query"]}
      }},
      {"type":"function","function":{
        "name":"call_specialist","description":"Déléguer un sous-objectif à l'agent spécialiste",
        "parameters":{"type":"object","properties":{
          "task":{"type":"string","enum":[
            "company_investors","investments_by_amount","top_investments_for_company","investments_in_period_currency",
            "common_investors_between_companies","co_invested_companies_for_company",
            "geo_near_companies","temporal_overlap_for_companies"
          ]},
          "params":{"type":"object"}
        },"required":["task"]}
      }}
    ]

    SYSTEM = (
      "Tu planifies façon HTN. Utilise lookup(size<=50) et join quand la paire est claire (on=['companies','id'] "
      "ou ['investors','id']). Pour des requêtes multi-étapes (co-invest, géo, temporalité), appelle call_specialist "
      "avec le task adapté. Rends un résumé clair (#résultats, éléments saillants) + pistes d’affinage."
    )

    messages = [{"role":"system","content": SYSTEM},
                {"role":"user","content": prompt}]

    try:
        for _ in range(8):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2
            )
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                return {"answer": msg.content, "mode": "llm"}

            # On ajoute d'abord le message assistant qui porte les tool_calls
            tool_calls_payload = [{
                "id": tc.id, "type":"function",
                "function":{"name": tc.function.name, "arguments": tc.function.arguments or "{}"}
            } for tc in msg.tool_calls]
            messages.append({"role":"assistant","content": msg.content or "", "tool_calls": tool_calls_payload})

            # Execute chaque outil puis répondre avec un message 'tool'
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

                if name == "graph_indices":
                    result = es_get("/_cat/indices?format=json", timeout=15)

                elif name == "graph_mapping":
                    idx = args.get("index")
                    result = es_get(f"/{idx}/_mapping?pretty", timeout=30) if idx else {"error":"index is required"}

                elif name == "graph_query":
                    op = args.get("op")
                    parent_index = args.get("parent_index")
                    child_index  = args.get("child_index")
                    on           = args.get("on")
                    es_q         = args.get("es_query") or {"match_all":{}}
                    size         = int(args.get("size", 50))
                    if op == "lookup":
                        result = es_post(f"/{parent_index}/_search",
                                         json={"size": size, "query": es_q}, timeout=30)
                    elif op == "join":
                        if not (parent_index and child_index and on and len(on)==2):
                            result = {"error":"join needs parent_index, child_index, on=[child_key,parent_key]"}
                        else:
                            join = {"indices":[child_index], "on": on, "request":{"query": es_q}}
                            result = es_post(f"/siren/{parent_index}/_search",
                                             json={"size": size, "query":{"join":join}}, timeout=60)
                    else:
                        result = {"error": f"unsupported op {op}"}

                elif name == "call_specialist":
                    task   = args.get("task")
                    params = args.get("params") or {}
                    specialist = SpecialistAgent(es_get, es_post)
                    result = specialist.run(task, params)

                else:
                    result = {"error": f"unknown tool {name}"}

                messages.append({"role":"tool","tool_call_id": tc.id,
                                 "name": name, "content": json.dumps(result)[:15000]})

        raise HTTPException(500, "LLM did not produce a final answer in 8 steps.")
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
