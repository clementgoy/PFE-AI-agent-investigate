from fastapi import FastAPI, Header, HTTPException
from fastapi import Query as Q
from pydantic import BaseModel
import os, json, requests

# Si on active /chat, on aura besoin d'OpenAI
# (on gère l'import dynamiquement pour que l'agent tourne même sans clé)
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    pass

# ===== Config de base (env) =====
ES = os.getenv("ES_URL", "http://localhost:9200")
AUTH = (os.getenv("ES_USER", "sirenadmin"), os.getenv("ES_PASS", "password"))
API_TOKEN = os.getenv("GRAPH_AGENT_TOKEN", "devtoken")
VERIFY_TLS = os.getenv("ES_VERIFY", "false").lower() == "true"  # pack Siren => false par défaut (cert auto-signé)

app = FastAPI()

# ===== Modèles =====
class Query(BaseModel):
    op: str                               # "lookup" | "join"
    parent_index: str | None = None
    child_index: str | None = None
    on: list[str] | None = None           # ordre attendu: [clé_dans_child, clé_dans_parent]
    es_query: dict | None = None
    size: int | None = 50
    join_type: str | None = None          # ex: HASH_JOIN, BROADCAST_JOIN (optionnel)

class ChatIn(BaseModel):
    prompt: str

# ===== Helpers HTTP vers ES (on garde ça simple) =====
def guard(h: str | None):
    if h != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "Unauthorized")

def es_get(path: str, **kwargs):
    # évite la duplication verify/auth/timeout
    try:
        r = requests.get(f"{ES}{path}", auth=AUTH, verify=VERIFY_TLS,
                         timeout=kwargs.pop("timeout", 30), **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ES GET {path} failed: {e}")

def es_post(path: str, json=None, **kwargs):
    try:
        r = requests.post(f"{ES}{path}", auth=AUTH, json=json, verify=VERIFY_TLS,
                          timeout=kwargs.pop("timeout", 60), **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ES POST {path} failed: {e}")

# ===== Endpoints “outils” =====
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
        return es_post(f"/{body.parent_index}/_search",
                       json={"size": size, "query": q},
                       timeout=30)

    if body.op == "join":
        if not (body.parent_index and body.child_index and body.on and len(body.on) == 2):
            raise HTTPException(400, "join needs parent_index, child_index, on=[child_key,parent_key]")
        join = {"indices": [body.child_index], "on": body.on}
        if body.join_type:
            join["type"] = body.join_type  # tu peux tester HASH_JOIN/BROADCAST_JOIN si tu veux
        if body.es_query:
            join["request"] = {"query": body.es_query}
        size = body.size or 50
        payload = {"size": size, "query": {"join": join}}
        return es_post(f"/siren/{body.parent_index}/_search", json=payload, timeout=60)

    raise HTTPException(400, f"unsupported op {body.op}")

# ===== /chat : prompt général -> tool-calls -> réponse finale =====
# But : l’utilisateur tape en langage naturel; ici l’agent orchestre un LLM qui choisit
# quand appeler /graph/indices, /graph/mapping et /graph/query (lookup/join).
@app.post("/chat")
def chat(body: ChatIn, authorization: str = Header(None)):
    guard(authorization)

    if not OPENAI_AVAILABLE:
        raise HTTPException(503, "OpenAI SDK not installed. Add 'openai' to requirements.txt and pip install.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not set in environment.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # à ajuster si besoin
    client = OpenAI(api_key=api_key)

    # Déclaration des outils (schéma minimal)
    TOOLS = [
      {
        "type":"function",
        "function":{
          "name":"graph_indices",
          "description":"Liste les indices Elasticsearch disponibles.",
          "parameters":{"type":"object","properties":{}}
        }
      },
      {
        "type":"function",
        "function":{
          "name":"graph_mapping",
          "description":"Récupère le mapping d'un index pour identifier des champs et clés.",
          "parameters":{"type":"object","properties":{
            "index":{"type":"string"}
          },"required":["index"]}
        }
      },
      {
        "type":"function",
        "function":{
          "name":"graph_query",
          "description":"Exécute lookup ou join via l'agent (Federate).",
          "parameters":{"type":"object","properties":{
            "op":{"type":"string","enum":["lookup","join"]},
            "parent_index":{"type":"string"},
            "child_index":{"type":"string"},
            "on":{"type":"array","items":{"type":"string"}},
            "es_query":{"type":"object"},
            "size":{"type":"integer"}
          },"required":["op","parent_index","es_query"]}
        }
      }
    ]

    SYSTEM = (
      "Tu es un planificateur HTN d'investigation. Règles:\n"
      "- Commence par graph_indices puis, si utile, graph_mapping(index) pour comprendre les champs.\n"
      "- Fais un lookup ciblé (size<=50). Si une paire de clés claire existe, tente un join (on=[clé_dans_child,clé_dans_parent]).\n"
      "- Résume: #hits, champs utiles, éventuels liens. Propose [affiner] ou [conclure]. Réponds concis."
    )

    messages = [
        {"role":"system","content": SYSTEM},
        {"role":"user","content": body.prompt}
    ]

    # Petite boucle tool-calls (max 5 itérations)
    for _ in range(5):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Réponse du modèle
            return {"answer": msg.content}

        # Exécuter chaque tool call et renvoyer le résultat au LLM
        for tc in msg.tool_calls:
            name = tc.function.name
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                pass

            # appelle directement nos helpers plutôt que de faire des requêtes HTTP à soi-même
            if name == "graph_indices":
                result = es_get("/_cat/indices?format=json", timeout=15)

            elif name == "graph_mapping":
                idx = args.get("index")
                if not idx:
                    result = {"error":"index is required"}
                else:
                    result = es_get(f"/{idx}/_mapping?pretty", timeout=30)

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
                        payload = {"size": size, "query":{"join":join}}
                        result = es_post(f"/siren/{parent_index}/_search", json=payload, timeout=60)
                else:
                    result = {"error": f"unsupported op {op}"}
            else:
                result = {"error": f"unknown tool {name}"}

            messages.append({
                "role":"tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result)[:15000] 
            })

    raise HTTPException(500, "LLM did not produce a final answer in 5 steps.")
