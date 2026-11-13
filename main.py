from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import pulp
import uvicorn

app = FastAPI(title="FPL Genius API")

# Allow your future website to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = "https://fantasy.premierleague.com/api/"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "FPL-Genius-API/1.0"})

# ==========================================
# MODELS
# ==========================================
class Transfer(BaseModel):
    out_player: str
    in_player: str

class OptimizeRequest(BaseModel):
    team_id: int
    free_transfers: int
    manual_transfers: list[Transfer] = []

# ==========================================
# HELPER FUNCTIONS (The Math Brain)
# ==========================================
def api_get(path):
    res = HTTP.get(BASE + path)
    res.raise_for_status()
    return res.json()

def safe_float(val, default=0.0):
    try:
        return float(val) if val not in (None, "", "-", "null") else default
    except:
        return default

def get_gws(events):
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    target = nxt or current
    picks_gw = current if current else target
    return picks_gw, target

def calculate_projections(bootstrap, fixtures, owned, target_gw, horizon=5):
    teams_short = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    completed_gws = sum(1 for e in bootstrap["events"] if e.get("finished"))
    
    # Map fixtures
    fmap = {}
    for f in fixtures:
        if not f.get("event"): continue
        gw = f["event"]
        fmap.setdefault(gw, {}).setdefault(f["team_h"], []).append({
            "opp": f["team_a"], "diff": safe_float(f.get("team_h_difficulty"), 3), "home": True
        })
        fmap.setdefault(gw, {}).setdefault(f["team_a"], []).append({
            "opp": f["team_h"], "diff": safe_float(f.get("team_a_difficulty"), 3), "home": False
        })

    rows = []
    for p in bootstrap["elements"]:
        is_owned = p["id"] in owned
        if not is_owned and p.get("status") in ("i", "s", "u", "n"): continue

        pos = POSITIONS[p["element_type"]]
        ppg = safe_float(p.get("points_per_game"))
        form = safe_float(p.get("form"))
        mins = max(0, safe_float(p.get("minutes")))
        
        if ppg <= 0: ppg = safe_float(p.get("total_points")) / max(1, safe_float(p.get("starts")))
        if form <= 0: form = ppg

        xg = max(0, safe_float(p.get("expected_goals")))
        xa = max(0, safe_float(p.get("expected_assists")))
        xgi_90 = ((xg + xa) / mins * 90) if mins > 0 else 0
        xgi_pts = xgi_90 * (GOAL_POINTS[pos] * 0.65 + 3 * 0.35)
        base_ppg = (0.55 * ppg) + (0.25 * form) + (0.20 * min(4.0, xgi_pts))

        exp_mins = min(90, mins / completed_gws) if completed_gws > 0 and mins > 0 else (25 if p.get("status") == "a" else 0)
        avail = 1.0 if p.get("chance_of_playing_next_round") is None else max(0.0, min(1.0, float(p["chance_of_playing_next_round"]) / 100.0))
        if p.get("status") in ("i", "s", "u", "n"): avail = 0

        min_factor = (min(1.0, exp_mins / 80.0)) * avail

        projections = {}
        fixtures_text = {}
        for gw in range(target_gw, target_gw + horizon):
            fx_list = fmap.get(gw, {}).get(p["team"], [])
            if not fx_list:
                projections[gw] = 0
                fixtures_text[gw] = "Blank"
                continue
            
            fx_factor = sum(max(0.78, min(1.22, 1.0 + (3.0 - fx["diff"]) * 0.075)) * (1.025 if fx["home"] else 1) for fx in fx_list) / len(fx_list)
            projections[gw] = max(0, base_ppg * min_factor * fx_factor)
            
            labels = [("H " if fx["home"] else "A ") + teams_short.get(fx["opp"], str(fx["opp"])) for fx in fx_list]
            fixtures_text[gw] = " / ".join(labels)

        rows.append({
            "id": int(p["id"]), "name": p["web_name"], "team": next((t["name"] for t in bootstrap["teams"] if t["id"] == p["team"]), ""),
            "team_id": int(p["team"]), "position": pos, "price": p["now_cost"] / 10.0,
            "points": int(p.get("total_points", 0)), "status": p.get("status", "a"),
            "owned": int(is_owned), "sell_price": owned.get(p["id"], {}).get("selling_price", p["now_cost"]) / 10.0,
            "proj": projections, "fixtures": fixtures_text
        })

    return pd.DataFrame(rows)

def run_optimizer(df, budget, target_gw, fts, current_ids, manual_transfers):
    # Apply manual transfers
    for t in manual_transfers:
        out_p = df[df["name"] == t.out_player].iloc[0]
        in_p = df[df["name"] == t.in_player].iloc[0]
        if out_p["id"] in current_ids:
            current_ids.remove(out_p["id"])
            current_ids.add(in_p["id"])
            # Note: Budget adjustments would happen here in a full production app

    prob = pulp.LpProblem("FPL_Opt", pulp.LpMaximize)
    x = prob.add_variable_dicts("pick", df.index, cat=pulp.LpBinary)

    # Objective: Max points over horizon
    prob += pulp.lpSum(x[i] * sum(df.loc[i, "proj"].values()) for i in df.index)

    for pos, cnt in SQUAD_COUNTS.items():
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == pos) == cnt
    for t_id in df["team_id"].unique():
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "team_id"] == t_id) <= 3

    prob += pulp.lpSum(x[i] * df.loc[i, "price"] for i in df.index) <= budget
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "id"] not in current_ids) <= fts

    prob.solve(pulp.HiGHS(msg=False))
    chosen = [i for i in df.index if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]
    return df.loc[chosen].to_dict(orient="records")

# ==========================================
# ROUTES
# ==========================================
@app.get("/api/team/{team_id}")
def get_team(team_id: int):
    try:
        bootstrap = api_get("bootstrap-static/")
        picks_gw, target_gw = get_gws(bootstrap["events"])
        picks = api_get(f"entry/{team_id}/event/{picks_gw}/picks/")
        history = api_get(f"entry/{team_id}/history/")
        transfers = api_get(f"entry/{team_id}/transfers/")
        
        return {
            "status": "success",
            "current_gw": target_gw,
            "bank": picks['entry_history']['bank'] / 10.0,
            "squad_ids": [p['element'] for p in picks['picks']],
            "history": history,
            "transfers": transfers
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/optimize")
def optimize_team(req: OptimizeRequest):
    try:
        bootstrap = api_get("bootstrap-static/")
        fixtures = api_get("fixtures/")
        picks_gw, target_gw = get_gws(bootstrap["events"])
        
        picks = api_get(f"entry/{req.team_id}/event/{picks_gw}/picks/")
        owned = {p['element']: {"selling_price": 0} for p in picks['picks']}
        
        # In a production app, you'd fetch real selling prices here
        # For now, we use current market price for math
        df = calculate_projections(bootstrap, fixtures, owned, target_gw)
        
        budget = (sum(df[df["id"].isin(owned)]["price"]) + (picks['entry_history']['bank'] / 10.0))
        
        optimal_squad = run_optimizer(df, budget, target_gw, req.free_transfers, list(owned.keys()), req.manual_transfers)
        
        return {"status": "success", "squad": optimal_squad}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)