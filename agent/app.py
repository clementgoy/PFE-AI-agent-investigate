
# on peut lancer ce service avec : uvicorn agent.app:app --reload --port 8000
# /chat fonctionne en 2 modes: "llm" (OpenAI) ou "local" (plan minimal)

import os, json, requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi import Query as Q
from pydantic import BaseModel

# on pourrait charger un .env avec la clé api à penser pour la suite quelle pratique est mieux 
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

ES = os.getenv("ES_URL", "http://localhost:9200")
AUTH = (os.getenv("ES_USER", "sirenadmin"), os.getenv("ES_PASS", "password"))
API_TOKEN = os.getenv("GRAPH_AGENT_TOKEN", "devtoken")
VERIFY_TLS = os.getenv("ES_VERIFY", "false").lower() == "true"
CHAT_MODE = os.getenv("CHAT_MODE", "llm").lower() # mode llm ou local

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
        r = requests.get(
            f"{ES}{path}",
            auth=AUTH,
            verify=VERIFY_TLS,
            timeout=kwargs.pop("timeout", 30),
            **kwargs
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"ES GET {path} failed: {e}")

def es_post(path: str, json=None, **kwargs):
    try:
        r = requests.post(
            f"{ES}{path}",
            auth=AUTH,
            json=json,
            verify=VERIFY_TLS,
            timeout=kwargs.pop("timeout", 60),
            **kwargs
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"ES POST {path} failed: {e}")


@app.get("/health")
def health(authorization: str = Header(None)):
    guard(authorization)
    info = {}
    try:
        info = es_get("/", timeout=5)
    except HTTPException as e:
        info = {"error": e.detail}
    return {
        "mode": CHAT_MODE,
        "es_url": ES,
        "verify_tls": VERIFY_TLS,
        "es": info,
        "openai_available": OPENAI_AVAILABLE
    }

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
        q = body.es_query or {"match_all": {}}
        size = body.size or 50
        return es_post(
            f"/{body.parent_index}/_search",
            json={"size": size, "query": q},
            timeout=30
        )

    if body.op == "join":
        if not (body.parent_index and body.child_index and body.on and len(body.on) == 2):
            raise HTTPException(400, "join needs parent_index, child_index, on=[child_key,parent_key]")
        join = {"indices": [body.child_index], "on": body.on}
        if body.join_type:
            join["type"] = body.join_type
        if body.es_query:
            join["request"] = {"query": body.es_query}
        size = body.size or 50
        payload = {"size": size, "query": {"join": join}}
        return es_post(f"/siren/{body.parent_index}/_search", json=payload, timeout=60)

    raise HTTPException(400, f"unsupported op {body.op}")


def local_plan_summary() -> str:
    # join investment -> company sur ["companies","id"]
    try:
        indices = es_get("/_cat/indices?format=json", timeout=15)
    except HTTPException:
        return "Elasticsearch hors service."

    have_company = any(i.get("index") == "company" for i in indices)
    have_invest  = any(i.get("index") == "investment" for i in indices)
    if not (have_company and have_invest):
        return "Indices requis absents (company, investment)."

    join = {"indices": ["investment"], "on": ["companies", "id"], "request": {"query": {"match_all": {}}}}
    payload = {"size": 10, "query": {"join": join}}
    res = es_post("/siren/company/_search", json=payload, timeout=60)

    hits = res.get("hits", {}).get("hits", []) or []
    total = res.get("hits", {}).get("total")
    total_value = total.get("value", 0) if isinstance(total, dict) else (0 if total is None else total)

    out = []
    for h in hits:
        src = h.get("_source", {})
        label = src.get("label") or src.get("permalink") or src.get("id")
        city  = src.get("city")
        cat   = src.get("category_code")
        out.append(f"- {label} (cat: {cat or 'n/a'}, city: {city or 'n/a'})")

    if not out:
        return "Aucun résultat via le JOIN (investment→company on=['companies','id']). Essaie d'affiner (année/montant/investisseur)."

    head = f"Résultats (≈{total_value} au total, top {len(out)} affichés) :"
    return head + "\n" + "\n".join(out)


@app.post("/chat")
async def chat(request: Request, authorization: str = Header(None)):
    guard(authorization)

    # Recup le prompt
    prompt: str | None = None
    try:
        data = await request.json()
        if isinstance(data, dict):
            prompt = data.get("prompt")
    except Exception:
        pass
    if not prompt:
        raw = await request.body()
        raw_text = (raw or b"").decode("utf-8", "ignore").strip()
        if raw_text and not raw_text.startswith("{"):
            prompt = raw_text
    if not prompt:
        qp = request.query_params
        prompt = qp.get("prompt") or qp.get("q")
    if not prompt:
        raise HTTPException(400, 'No prompt provided. Send JSON {"prompt":"..."}, text/plain body, or ?prompt=...')

    # local
    if CHAT_MODE != "llm":
        summary = local_plan_summary()
        return {"answer": summary, "mode": "local"}

    # LLM
    if not OPENAI_AVAILABLE:
        raise HTTPException(503, "OpenAI SDK not installed. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not set in environment.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Tools fournis au LLM
    TOOLS = [
      {"type":"function","function":{
        "name":"graph_indices",
        "description":"Liste les indices Elasticsearch disponibles.",
        "parameters":{"type":"object","properties":{}}
      }},
      {"type":"function","function":{
        "name":"graph_mapping",
        "description":"Récupère le mapping d'un index.",
        "parameters":{"type":"object","properties":{"index":{"type":"string"}},"required":["index"]}
      }},
      {"type":"function","function":{
        "name":"graph_query",
        "description":"lookup / join via Siren Federate",
        "parameters":{"type":"object","properties":{
          "op":{"type":"string","enum":["lookup","join"]},
          "parent_index":{"type":"string"},
          "child_index":{"type":"string"},
          "on":{"type":"array","items":{"type":"string"}},
          "es_query":{"type":"object"},
          "size":{"type":"integer"}
        },"required":["op","parent_index","es_query"]}
      }}
    ]

    SYSTEM = (
      "Tu es un planificateur HTN d'investigation. Étapes: 1.indices 2.mapping 3.lookup(size<=50) 4.join si paire claire "
      "(on=[clé_child,clé_parent]). Utilise investment company via on=['companies','id'] si pertinent. "
      "Résume (#hits, champs utiles) et propose [affiner]/[conclure]. Réponds de manière concise."
    )

    messages: list[dict] = [
        {"role":"system","content": SYSTEM},
        {"role":"user","content": prompt}
    ]

    try:
        # limite du nb d'itérations pour éviter boucles infinies
        for _ in range(6):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            # Si aucun tool_calls -> réponse finale en clair
            if not getattr(msg, "tool_calls", None):
                return {"answer": msg.content, "mode": "llm"}

            tool_calls_payload = []
            for tc in msg.tool_calls:
                tool_calls_payload.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}"
                    }
                })
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls_payload
            })

            # exécute chaque tool_call et un message 'tool' par appel
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
                        result = es_post(
                            f"/{parent_index}/_search",
                            json={"size": size, "query": es_q},
                            timeout=30
                        )
                    elif op == "join":
                        if not (parent_index and child_index and on and len(on)==2):
                            result = {"error":"join needs parent_index, child_index, on=[child_key,parent_key]"}
                        else:
                            join = {"indices":[child_index], "on": on, "request":{"query": es_q}}
                            payload = {"size": size, "query":{"join":join}}
                            result = es_post(f"/siren/{parent_index}/_search", json=payload, timeout=60)
                    else:
                        result = {"error": f"unsupported op {op}"}
                else:
                    result = {"error": f"unknown tool {name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result)[:15000]
                })

        # Si on sort de la boucle sans réponse
        raise HTTPException(500, "LLM did not produce a final answer in 6 steps.")
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
