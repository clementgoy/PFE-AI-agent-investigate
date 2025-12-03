
# Agent “métier” : encode les bons enchaînements (lookup/join) pour des questions d’enquête

from typing import Any, Dict, List, Set
from dateutil import parser as dateparser

class SpecialistAgent:
    SUPPORTED_TASKS = {
        "company_investors": "Investisseurs d'une entreprise (par label/id).",
        "investments_by_amount": "Filtrer des investissements (montant/devise/période) + join company.",
        "top_investments_for_company": "Investissements d'une entreprise donnée.",
        "investments_in_period_currency": "Investissements par période + devise (+ join company).",
        "common_investors_between_companies": "Investisseurs communs entre 2 entreprises.",
        "co_invested_companies_for_company": "Entreprises partageant au moins 1 investisseur avec une cible.",
        "geo_near_companies": "Entreprises proches d’un point (km).",
        "temporal_overlap_for_companies": "Investissements proches dans le temps entre 2 entreprises."
    }

    def __init__(self, es_get_func, es_post_func):
        self.es_get = es_get_func
        self.es_post = es_post_func

    # helpers
    def _find_company_id_by_label(self, label: str) -> str | None:

        q = {"term": {"label.raw": label}}
        res = self.es_post("/company/_search", json={"size": 1, "query": q})
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            q = {"term": {"label": label}}
            res = self.es_post("/company/_search", json={"size": 1, "query": q})
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                return None
        return hits[0].get("_source", {}).get("id")

    def _investors_for_company_id(self, company_id: str, size: int = 200) -> Set[str]:
        # Récupère un échantillon d’investissements de la company 
        q = {"terms": {"companies": [company_id]}}
        res = self.es_post("/investment/_search", json={"size": size, "query": q})
        inv_ids: Set[str] = set()
        for h in res.get("hits", {}).get("hits", []) or []:
            inv_ids.update(h.get("_source", {}).get("investors", []) or [])
        return inv_ids

    # tasks existants 
    def company_investors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        size = int(params.get("size", 5))
        company_id = params.get("company_id")
        if not company_id:
            label = params.get("company_label")
            if not label:
                return {"error": "company_investors needs company_id or company_label"}
            company_id = self._find_company_id_by_label(label)
            if not company_id:
                return {"error": f"Company '{label}' not found."}

        es_query = {"terms": {"companies": [company_id]}}
        join = {"indices": ["investment"], "on": ["investors", "id"], "request": {"query": es_query}}
        res = self.es_post("/siren/investor/_search", json={"size": size, "query": {"join": join}})

        out = []
        for h in res.get("hits", {}).get("hits", []) or []:
            s = h.get("_source", {})
            out.append({"investor_label": s.get("label"), "investor_id": s.get("id")})

        total = res.get("hits", {}).get("total")
        total_val = total.get("value", 0) if isinstance(total, dict) else 0
        return {"summary": f"{total_val} investisseurs pour {company_id} (top {len(out)}).",
                "company_id": company_id, "investors": out}

    def investments_by_amount(self, params: Dict[str, Any]) -> Dict[str, Any]:
        size = int(params.get("size", 10))
        q = {"bool": {"filter": []}}
        if "min_amount" in params:
            q["bool"]["filter"].append({"range": {"raised_amount": {"gte": float(params["min_amount"])}}})
        if "currency_code" in params:
            q["bool"]["filter"].append({"term": {"raised_currency_code": str(params["currency_code"])}})
        yr = {}
        if "year_min" in params: yr["gte"] = int(params["year_min"])
        if "year_max" in params: yr["lte"] = int(params["year_max"])
        if yr:
            q["bool"]["filter"].append({"range": {"funded_year": yr}})

        inv = self.es_post("/investment/_search", json={"size": size, "query": q})
        hits = inv.get("hits", {}).get("hits", []) or []

        if params.get("join_company", True):
            join = {"indices": ["investment"], "on": ["companies", "id"], "request": {"query": q}}
            res = self.es_post("/siren/company/_search", json={"size": size, "query": {"join": join}})
            chits = res.get("hits", {}).get("hits", []) or []
            companies = [{"company_label": c.get("_source", {}).get("label"),
                          "company_id": c.get("_source", {}).get("id")} for c in chits]
            total = res.get("hits", {}).get("total")
            total_val = total.get("value", 0) if isinstance(total, dict) else 0
            return {"summary": f"{total_val} entreprises liées (top {len(companies)}).",
                    "filters": params, "companies": companies,
                    "sample_investments": [x.get("_source", {}) for x in hits[:min(3, len(hits))]]}
        else:
            total = inv.get("hits", {}).get("total")
            total_val = total.get("value", 0) if isinstance(total, dict) else 0
            return {"summary": f"{total_val} investissements (top {len(hits)}).",
                    "filters": params, "investments": [x.get("_source", {}) for x in hits]}

    def top_investments_for_company(self, params: Dict[str, Any]) -> Dict[str, Any]:
        size = int(params.get("size", 5))
        company_id = params.get("company_id")
        if not company_id:
            label = params.get("company_label")
            if not label:
                return {"error": "top_investments_for_company needs company_id or company_label"}
            company_id = self._find_company_id_by_label(label)
            if not company_id:
                return {"error": f"Company '{label}' not found."}

        q = {"terms": {"companies": [company_id]}}
        inv = self.es_post("/investment/_search", json={"size": size, "query": q})
        hits = inv.get("hits", {}).get("hits", []) or []
        out = []
        for h in hits:
            s = h.get("_source", {})
            out.append({"label": s.get("label"),
                        "funded_year": s.get("funded_year"),
                        "raised_amount": s.get("raised_amount"),
                        "raised_currency_code": s.get("raised_currency_code")})
        total = inv.get("hits", {}).get("total")
        total_val = total.get("value", 0) if isinstance(total, dict) else 0
        return {"summary": f"{total_val} investissements pour {company_id} (top {len(out)}).",
                "company_id": company_id, "investments": out}

    def investments_in_period_currency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params or {})
        params.setdefault("min_amount", 0)
        params.setdefault("join_company", True)
        return self.investments_by_amount(params)

    # tasks “enquête”
    def common_investors_between_companies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Entrée: company_id_a / company_label_a, company_id_b / company_label_b
        def get_id(pfx: str) -> str | None:
            cid = params.get(f"company_id_{pfx}")
            if cid: return cid
            lab = params.get(f"company_label_{pfx}")
            return self._find_company_id_by_label(lab) if lab else None

        a = get_id("a"); b = get_id("b")
        if not (a and b):
            return {"error": "needs company_a and company_b (id_* or label_*)"}
        inv_a = self._investors_for_company_id(a)
        inv_b = self._investors_for_company_id(b)
        common = list(inv_a.intersection(inv_b))[:200] or []

        if not common:
            return {"summary": f"Aucun investisseur commun entre {a} et {b}.", "common_investors": []}

        # Résoudre les labels d’investors
        res = self.es_post("/investor/_search", json={"size": len(common), "query": {"terms": {"id": common}}})
        out = [{"investor_id": h.get("_source", {}).get("id"),
                "investor_label": h.get("_source", {}).get("label")} for h in res.get("hits", {}).get("hits", [])]
        return {"summary": f"{len(common)} investisseurs communs.", "company_a": a, "company_b": b, "common_investors": out}

    def co_invested_companies_for_company(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # entreprises partageant au moins 1 investisseur
        size = int(params.get("size", 10))
        company_id = params.get("company_id")
        if not company_id:
            label = params.get("company_label")
            if not label:
                return {"error": "needs company_id or company_label"}
            company_id = self._find_company_id_by_label(label)
            if not company_id:
                return {"error": f"Company '{label}' not found."}

        inv_ids = list(self._investors_for_company_id(company_id))[:200]
        if not inv_ids:
            return {"summary": f"Aucun investisseur trouvé pour {company_id}.", "companies": []}

        # Investissements où investors est ds inv_ids pour recup les companies
        q = {"terms": {"investors": inv_ids}}
        res = self.es_post("/investment/_search", json={"size": 500, "query": q})
        freq: Dict[str, int] = {}
        for h in res.get("hits", {}).get("hits", []) or []:
            for c in h.get("_source", {}).get("companies", []) or []:
                if c != company_id:
                    freq[c] = freq.get(c, 0) + 1

        # Top-N par fréquence
        ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:size]
        if not ranked:
            return {"summary": "Aucune entreprise co-investie trouvée.", "companies": []}

        ids = [c for c, _ in ranked]
        res2 = self.es_post("/company/_search", json={"size": len(ids), "query": {"terms": {"id": ids}}})
        labels = {h.get("_source", {}).get("id"): h.get("_source", {}).get("label")
                  for h in res2.get("hits", {}).get("hits", [])}
        out = [{"company_id": cid, "company_label": labels.get(cid), "co_invest_count": count} for cid, count in ranked]
        return {"summary": f"{len(out)} co-investies avec {company_id}.", "companies": out}

    def geo_near_companies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Entrée: lat, lon, distance_km
        try:
            lat = float(params["lat"]); lon = float(params["lon"])
        except Exception:
            return {"error": "geo_near_companies needs lat, lon"}
        dist = float(params.get("distance_km", 50))
        size = int(params.get("size", 10))

        q = {"bool": {"filter": [{"geo_distance": {"distance": f"{dist}km", "location": {"lat": lat, "lon": lon}}}]}}
        res = self.es_post("/company/_search", json={"size": size, "query": q})
        out = [{"company_id": h.get("_source", {}).get("id"),
                "company_label": h.get("_source", {}).get("label"),
                "city": h.get("_source", {}).get("city"),
                "countrycode": h.get("_source", {}).get("countrycode")} for h in res.get("hits", {}).get("hits", [])]
        total = res.get("hits", {}).get("total")
        total_val = total.get("value", 0) if isinstance(total, dict) else 0
        return {"summary": f"{total_val} entreprises à ~{dist}km (top {len(out)}).", "companies": out}

    def temporal_overlap_for_companies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Entrée: company_a (id/label), company_b (id/label), window_days
        def get_id(arg_id: str, arg_label: str) -> str | None:
            return params.get(arg_id) or (self._find_company_id_by_label(params.get(arg_label)) if params.get(arg_label) else None)

        a = get_id("company_id_a", "company_label_a")
        b = get_id("company_id_b", "company_label_b")
        if not (a and b):
            return {"error": "needs company_a and company_b (id_* or label_*)."}
        window = int(params.get("window_days", 90))

        def fetch_dates(cid: str) -> List[str]:
            q = {"terms": {"companies": [cid]}}
            res = self.es_post("/investment/_search", json={"size": 200, "query": q})
            out = []
            for h in res.get("hits", {}).get("hits", []) or []:
                fd = h.get("_source", {}).get("funded_date")
                if fd: out.append(fd)
            return out

        dates_a = [dateparser.parse(d) for d in fetch_dates(a)]
        dates_b = [dateparser.parse(d) for d in fetch_dates(b)]
        matches: List[Dict[str, Any]] = []
        for da in dates_a:
            for db in dates_b:
                if abs((da - db).days) <= window:
                    matches.append({"date_a": da.isoformat(), "date_b": db.isoformat(), "delta_days": abs((da-db).days)})
        return {"summary": f"{len(matches)} paires d’événements dans ±{window} jours.",
                "company_a": a, "company_b": b, "pairs": matches[:50]}
    
    # point d’entree
    def run(self, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if task not in self.SUPPORTED_TASKS:
            return {"error": f"unsupported task '{task}'",
                    "supported": list(self.SUPPORTED_TASKS.keys())}
        return getattr(self, task)(params or {})
