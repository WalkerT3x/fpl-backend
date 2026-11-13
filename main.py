from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import pulp
import uvicorn
import traceback

app = FastAPI(title="FPL Genius API")

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

class Transfer(BaseModel):
    out_player: str
    in_player: str

class OptimizeRequest(BaseModel):
    team_id: int
    free_transfers: int
    manual_transfers: list[Transfer] = []

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
    teams_code = {t["id"]: t["code"] for t in bootstrap["teams"]}
    completed_gws = sum(1 for e in bootstrap["events"] if e.get("finished"))
    
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

        team_name = next((t["name"] for t in bootstrap["teams"] if t["id"] == p["team"]), "")
        team_code = teams_code.get(p["team"], 0)
        
        projections = {}
        fixtures_text = {}
        for gw in range(target_gw, target_gw + horizon):
            fx_list = fmap.get(gw, {}).get(p["team"], [])
            if not fx_list:
                projections[str(gw)] = 0
                fixtures_text[str(gw)] = "Blank"
                continue
            
            fx_factor = sum(max(0.78, min(1.22, 1.0 + (3.0 - fx["diff"]) * 0.075)) * (1.025 if fx["home"] else 1) for fx in fx_list) / len(fx_list)
            projections[str(gw)] = max(0, base_ppg * min_factor * fx_factor)
            
            labels = [("H " if fx["home"] else "A ") + teams_short.get(fx["opp"], str(fx["opp"])) for fx in fx_list]
            fixtures_text[str(gw)] = " / ".join(labels)

        if pos == "GK":
            shirt_url = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{team_code}_1-110.png"
        else:
            shirt_url = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{team_code}-110.png"

        rows.append({
            "id": int(p["id"]), 
            "code": int(p.get("code", 0)), 
            "shirt_url": shirt_url,
            "name": p["web_name"], 
            "team": team_name,
            "team_id": int(p["team"]), "position": pos, 
            "price10": int(p["now_cost"]),
            "sell_price10": int(owned.get(p["id"], {}).get("selling_price", p["now_cost"])),
            "points": int(p.get("total_points", 0)), "status": p.get("status", "a"),
            "proj": projections, "fixtures": fixtures_text
        })

    return pd.DataFrame(rows)

def solve_squad(df, budget10, target_gw, horizon=5, max_transfers=None, current_ids=None, objective="horizon", force_free_hit=False):
    current_ids = set(current_ids or [])
    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
    indices = list(df.index)
    x = prob.add_variable_dicts("pick", indices, cat=pulp.LpBinary)

    if objective == "target":
        score = {i: float(df.loc[i, "proj"].get(str(target_gw), 0)) for i in indices}
    else:
        score = {i: sum(float(df.loc[i, "proj"].get(str(gw), 0)) for gw in range(target_gw, min(39, target_gw + horizon))) for i in indices}

    prob += pulp.lpSum(x[i] * score[i] for i in indices)

    for pos, count in SQUAD_COUNTS.items():
        prob += pulp.lpSum(x[i] for i in indices if df.loc[i, "position"] == pos) == count

    for team_id in df["team_id"].unique():
        prob += pulp.lpSum(x[i] for i in indices if df.loc[i, "team_id"] == team_id) <= 3

    costs = {}
    for i in indices:
        player_id = int(df.loc[i, "id"])
        costs[i] = int(df.loc[i, "sell_price10"]) if player_id in current_ids else int(df.loc[i, "price10"])

    prob += pulp.lpSum(x[i] * costs[i] for i in indices) <= int(budget10)

    if (max_transfers is not None and not force_free_hit):
        prob += pulp.lpSum(x[i] for i in indices if int(df.loc[i, "id"]) not in current_ids) <= int(max_transfers)

    prob.solve(pulp.HiGHS(msg=False))
    chosen = [i for i in indices if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5]

    if len(chosen) != 15: raise ValueError("No legal squad found. Check your budget and transfers.")
    return df.loc[chosen].copy()

def best_legal_xi(squad, gw):
    best, best_score = None, -1
    for defenders in range(3, 6):
        for mids in range(2, 6):
            forwards = 10 - defenders - mids
            if forwards < 1 or forwards > 3: continue
            selected = []
            formation = [("GK", 1), ("DEF", defenders), ("MID", mids), ("FWD", forwards)]

            for pos, count in formation:
                candidates = squad[squad["position"] == pos].copy()
                candidates["_projection"] = candidates["proj"].apply(lambda d: float(d.get(str(gw), 0)))
                selected.extend(candidates.sort_values("_projection", ascending=False).head(count).index.tolist())

            if len(selected) != 11: continue
            score = sum(float(squad.loc[i, "proj"].get(str(gw), 0)) for i in selected)
            if score > best_score:
                best_score = score
                best = selected

    if best is None: raise ValueError("Could not construct a legal starting XI.")

    xi = squad.loc[best].copy()
    xi["GWProjection"] = xi["proj"].apply(lambda d: float(d.get(str(gw), 0)))
    xi["Pos_Order"] = xi["position"].map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4})
    xi = xi.sort_values(["Pos_Order", "GWProjection"], ascending=[True, False])

    bench = squad.drop(index=best).copy()
    bench["GWProjection"] = bench["proj"].apply(lambda d: float(d.get(str(gw), 0)))
    bench = pd.concat([bench[bench["position"] == "GK"], bench[bench["position"] != "GK"].sort_values("GWProjection", ascending=False)])

    return xi, bench, best_score

def captain_choice(xi):
    ranked = xi.sort_values("GWProjection", ascending=False)
    return (ranked.iloc[0]["name"], ranked.iloc[1]["name"], float(ranked.iloc[0]["GWProjection"]))

def apply_manual_transfers(df, owned, bank10, manual_transfers):
    player_db = {p["name"].lower(): p for _, p in df.iterrows()}
    
    for t in manual_transfers:
        out_p = player_db.get(t.out_player.lower())
        in_p = player_db.get(t.in_player.lower())
        
        if out_p is not None and in_p is not None and out_p["id"] in owned:
            out_id, in_id = out_p["id"], in_p["id"]
            bank10 += owned[out_id]["selling_price"]
            del owned[out_id]
            cost = in_p["price10"]
            bank10 -= cost
            owned[in_id] = {"selling_price": cost}
            
    return owned, bank10

@app.post("/api/optimize")
def optimize_team(req: OptimizeRequest):
    try:
        bootstrap = api_get("bootstrap-static/")
        fixtures = api_get("fixtures/")
        picks_gw, target_gw = get_gws(bootstrap["events"])
        
        picks = api_get(f"entry/{req.team_id}/event/{picks_gw}/picks/")
        
        player_costs = {p["id"]: p["now_cost"] for p in bootstrap["elements"]}
        owned = {}
        for p in picks["picks"]:
            pid = p["element"]
            sp = p.get("selling_price")
            if not sp: sp = player_costs.get(pid, 0)
            owned[pid] = {"selling_price": sp}

        df = calculate_projections(bootstrap, fixtures, owned, target_gw)
        bank10 = int(picks.get("entry_history", {}).get("bank", 0))
        
        owned, bank10 = apply_manual_transfers(df, owned, bank10, req.manual_transfers)
        current_ids = list(owned.keys())
        
        selling_value10 = sum(owned[player_id]["selling_price"] for player_id in current_ids)
        budget10 = selling_value10 + bank10
        
        current_squad = df[df['id'].isin(current_ids)].copy()
        if len(current_squad) != 15: raise ValueError("Could not retrieve all 15 players from your existing squad.")

        new_squad = solve_squad(df, budget10, target_gw, horizon=5, max_transfers=req.free_transfers, current_ids=current_ids, objective="horizon")
        
        proposed_xi, proposed_bench, proposed_points = best_legal_xi(new_squad, target_gw)
        captain, vice, _ = captain_choice(proposed_xi)
        
        return {
            "status": "success", 
            "current_gw": target_gw, 
            "starting_11": proposed_xi.to_dict(orient="records"), 
            "bench": proposed_bench.to_dict(orient="records"),
            "captain": captain,
            "vice": vice
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/api/chips")
def get_chips(req: OptimizeRequest):
    try:
        bootstrap = api_get("bootstrap-static/")
        fixtures = api_get("fixtures/")
        history = api_get(f"entry/{req.team_id}/history/")
        picks_gw, target_gw = get_gws(bootstrap["events"])
        
        used_chips = []
        if "chips" in history:
            used_chips = [c["name"].upper() for c in history["chips"]]
            used_chips = ["WC" if c == "WILDCARD" else "FH" if c == "FREEHIT" else "BB" if c == "BBOOST" else "TC" if c == "3XC" else c for c in used_chips]

        picks = api_get(f"entry/{req.team_id}/event/{picks_gw}/picks/")
        
        player_costs = {p["id"]: p["now_cost"] for p in bootstrap["elements"]}
        owned = {}
        for p in picks["picks"]:
            pid = p["element"]
            sp = p.get("selling_price")
            if not sp: sp = player_costs.get(pid, 0)
            owned[pid] = {"selling_price": sp}

        df = calculate_projections(bootstrap, fixtures, owned, target_gw, horizon=9)
        bank10 = int(picks.get("entry_history", {}).get("bank", 0))
        
        owned, bank10 = apply_manual_transfers(df, owned, bank10, req.manual_transfers)
        current_ids = list(owned.keys())
        
        selling_value10 = sum(owned[player_id]["selling_price"] for player_id in current_ids)
        budget10 = selling_value10 + bank10
        current_squad = df[df['id'].isin(current_ids)].copy()
        
        weeks = list(range(target_gw, 20))
        if not weeks: raise ValueError("No remaining gameweeks.")

        weekly = {}
        for gw in weeks:
            xi, bench, xi_points = best_legal_xi(current_squad, gw)
            bench_points = float(bench["GWProjection"].sum())
            cap_name, vice_name, cap_points = captain_choice(xi)
            weekly[gw] = {"xi": xi_points, "bb": bench_points, "tc": cap_points, "captain": cap_name, "bench": bench}

        fh_values = {}
        for gw in weeks:
            try:
                fh_squad = solve_squad(df, budget10=budget10, target_gw=gw, horizon=1, objective="target", current_ids=set(), force_free_hit=True)
                fh_xi, _, fh_points = best_legal_xi(fh_squad, gw)
                fh_values[gw] = (fh_points - weekly[gw]["xi"])
            except: fh_values[gw] = 0.0

        def best_week(values): return max(values, key=values.get) if values else None

        best_bb_gw = best_week({g: weekly[g]["bb"] for g in weeks})
        best_tc_gw = best_week({g: weekly[g]["tc"] for g in weeks})
        best_fh_gw = best_week(fh_values)
        
        fh_squad = solve_squad(df, budget10=budget10, target_gw=best_fh_gw, horizon=1, objective="target", current_ids=set(), force_free_hit=True)
        fh_xi, fh_bench, _ = best_legal_xi(fh_squad, best_fh_gw)

        return {
            "status": "success",
            "current_gw": target_gw,
            "chip_advice": {
                "bench_boost": {"gw": best_bb_gw, "reason": f"Highest projected bench points ({weekly[best_bb_gw]['bb']:.1f} pts).", "used": "BB" in used_chips},
                "triple_captain": {"gw": best_tc_gw, "reason": f"Highest projected captain points: {weekly[best_tc_gw]['captain']} ({weekly[best_tc_gw]['tc']:.1f} pts).", "used": "TC" in used_chips},
                "free_hit": {"gw": best_fh_gw, "reason": f"Biggest upgrade to your starting 11 (+{fh_values[best_fh_gw]:.1f} pts).", "used": "FH" in used_chips},
                "wildcard": {"gw": best_fh_gw, "reason": "Major fixture swing to restructure.", "used": "WC" in used_chips}
            },
            "free_hit_team": {"starting_11": fh_xi.to_dict(orient="records"), "bench": fh_bench.to_dict(orient="records")}
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/teams")
def get_teams():
    try:
        bootstrap = api_get("bootstrap-static/")
        teams = [{"id": t["id"], "name": t["name"], "code": t["code"]} for t in bootstrap["teams"]]
        return {"status": "success", "teams": teams}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/players")
def get_players():
    try:
        bootstrap = api_get("bootstrap-static/")
        players = sorted(list(set([p["web_name"] for p in bootstrap["elements"]])))
        return {"status": "success", "players": players}
    except Exception as e:
        return {"error": str(e)}