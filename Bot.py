import time
import json
import os
import requests
import numpy as np
from scipy.stats import norm, poisson
from datetime import datetime, timezone
import pytz

# ==========================================
# 1. CREDENTIALS & 4.0% MINIMUM EV FILTER
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1565139751810629673/OP5KjHw2InKG1EBpdrjjYlpF0XLnM92UqokmY-x_8_d97QVKHScQZp2ynKnOOTPPWhcE"
)
THE_ODDS_API_KEY = "9ecb23956a7654b951782fd28ec3a8b5"

# Market Toggles
SCAN_FULL_GAME = True         # Full-Game Spreads & Totals
SCAN_HALVES_QUARTERS = True   # 1st Half & 1st Quarter
SCAN_PLAYER_PROPS = True      # Player Points, Rebounds, Assists

MAX_DEEP_SCAN_GAMES_PER_RUN = 3
MIN_EDGE_TO_POST = 0.040      # STRICT 4.0% OR BETTER +EV FILTER

TRACKER_FILE = "tracked_all_markets.json"
SCANNED_EVENTS_FILE = "scanned_prop_events.json"

SCORE_CHECK_SECONDS = 300     # Checks ESPN scores every 5 mins (FREE)
FANDUEL_SCAN_SECONDS = 7200   # Scans FanDuel every 2 hours

NBA_STD_DEV_TOTAL = 16.0
NBA_STD_DEV_SPREAD = 12.5
NBA_STD_DEV_H1_TOTAL = 11.5
NBA_STD_DEV_H1_SPREAD = 8.8
NBA_STD_DEV_Q1_TOTAL = 8.2
NBA_STD_DEV_Q1_SPREAD = 6.2
HOME_COURT_ADVANTAGE = 2.2

def load_json(filename, default_val):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except:
            return default_val
    return default_val

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def american_to_implied_prob(odds_val):
    try:
        odds = float(odds_val)
        return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)
    except:
        return 0.5238

def get_team_efficiency_data():
    return {
        "San Antonio Spurs": {"off": 119.4, "def": 107.5, "pace": 100.8},
        "Houston Rockets": {"off": 119.2, "def": 109.7, "pace": 99.5},
        "Golden State Warriors": {"off": 121.3, "def": 111.7, "pace": 101.2},
        "Minnesota Timberwolves": {"off": 117.2, "def": 106.0, "pace": 98.4},
        "Oklahoma City Thunder": {"off": 113.4, "def": 105.6, "pace": 100.5},
        "Los Angeles Lakers": {"off": 122.3, "def": 114.7, "pace": 101.4},
        "Portland Trail Blazers": {"off": 119.6, "def": 113.8, "pace": 99.8},
        "Memphis Grizzlies": {"off": 117.3, "def": 110.2, "pace": 100.9},
        "Dallas Mavericks": {"off": 115.8, "def": 111.0, "pace": 99.2},
        "Denver Nuggets": {"off": 116.0, "def": 115.6, "pace": 98.1},
        "Detroit Pistons": {"off": 114.3, "def": 106.2, "pace": 99.0},
        "Boston Celtics": {"off": 119.6, "def": 108.6, "pace": 98.7},
        "Los Angeles Clippers": {"off": 111.9, "def": 116.0, "pace": 97.8},
        "LA Clippers": {"off": 111.9, "def": 116.0, "pace": 97.8},
        "Phoenix Suns": {"off": 112.6, "def": 121.5, "pace": 99.1},
        "Toronto Raptors": {"off": 116.6, "def": 114.0, "pace": 100.2},
        "Atlanta Hawks": {"off": 115.5, "def": 113.3, "pace": 101.8},
        "New York Knicks": {"off": 114.6, "def": 108.5, "pace": 96.9},
        "Cleveland Cavaliers": {"off": 115.9, "def": 114.5, "pace": 98.5},
        "New Orleans Pelicans": {"off": 113.2, "def": 118.6, "pace": 99.4},
        "Miami Heat": {"off": 111.8, "def": 116.9, "pace": 97.2},
        "Utah Jazz": {"off": 104.4, "def": 116.0, "pace": 99.6},
        "Philadelphia 76ers": {"off": 111.3, "def": 108.3, "pace": 98.0},
        "Sacramento Kings": {"off": 108.5, "def": 121.6, "pace": 100.4},
        "Indiana Pacers": {"off": 110.4, "def": 114.3, "pace": 102.1},
        "Orlando Magic": {"off": 108.2, "def": 117.4, "pace": 98.3},
        "Washington Wizards": {"off": 109.0, "def": 114.2, "pace": 101.5},
        "Brooklyn Nets": {"off": 106.3, "def": 115.3, "pace": 97.6},
        "Milwaukee Bucks": {"off": 108.6, "def": 121.9, "pace": 99.3},
        "Charlotte Hornets": {"off": 103.5, "def": 115.6, "pace": 98.8},
        "Chicago Bulls": {"off": 108.6, "def": 124.6, "pace": 100.7}
    }

def get_player_baselines():
    return {
        "Jayson Tatum": {"PTS": 28.4, "REB": 8.5, "AST": 5.4},
        "Jaylen Brown": {"PTS": 25.2, "REB": 6.1, "AST": 4.2},
        "Jalen Brunson": {"PTS": 27.9, "REB": 3.6, "AST": 6.8},
        "Karl-Anthony Towns": {"PTS": 24.5, "REB": 11.8, "AST": 3.1},
        "Tyrese Maxey": {"PTS": 27.6, "REB": 3.8, "AST": 6.4},
        "Nikola Jokić": {"PTS": 29.1, "REB": 13.2, "AST": 10.4},
        "Jamal Murray": {"PTS": 21.8, "REB": 4.1, "AST": 6.5},
        "Shai Gilgeous-Alexander": {"PTS": 31.2, "REB": 5.6, "AST": 6.4},
        "Luka Dončić": {"PTS": 34.5, "REB": 9.1, "AST": 9.6},
        "Kyrie Irving": {"PTS": 25.4, "REB": 4.9, "AST": 5.3},
        "Anthony Davis": {"PTS": 26.8, "REB": 12.4, "AST": 3.6},
        "LeBron James": {"PTS": 24.2, "REB": 7.4, "AST": 8.9},
        "Stephen Curry": {"PTS": 26.4, "REB": 4.5, "AST": 5.2},
        "Anthony Edwards": {"PTS": 28.1, "REB": 5.8, "AST": 5.1},
        "Victor Wembanyama": {"PTS": 26.8, "REB": 11.2, "AST": 4.1},
        "Kevin Durant": {"PTS": 27.4, "REB": 6.6, "AST": 4.8},
        "Devin Booker": {"PTS": 27.8, "REB": 4.6, "AST": 6.9},
        "James Harden": {"PTS": 21.4, "REB": 5.8, "AST": 8.9},
        "Giannis Antetokounmpo": {"PTS": 30.8, "REB": 11.6, "AST": 6.1},
        "Damian Lillard": {"PTS": 20.4, "REB": 3.8, "AST": 4.9},
        "Trae Young": {"PTS": 26.2, "REB": 3.1, "AST": 11.2},
        "Cade Cunningham": {"PTS": 24.6, "REB": 6.4, "AST": 9.1},
        "Donovan Mitchell": {"PTS": 27.2, "REB": 4.4, "AST": 5.2},
        "Domantas Sabonis": {"PTS": 19.8, "REB": 14.1, "AST": 7.8},
        "De'Aaron Fox": {"PTS": 26.9, "REB": 4.6, "AST": 6.1},
        "Bam Adebayo": {"PTS": 20.2, "REB": 10.4, "AST": 4.1},
        "Tyler Herro": {"PTS": 23.4, "REB": 5.2, "AST": 4.8},
        "Lauri Markkanen": {"PTS": 23.6, "REB": 8.2, "AST": 2.1},
        "Paolo Banchero": {"PTS": 25.8, "REB": 7.6, "AST": 5.4},
        "Franz Wagner": {"PTS": 24.8, "REB": 5.7, "AST": 4.6},
        "Alperen Şengün": {"PTS": 21.6, "REB": 11.2, "AST": 5.1},
        "Ja Morant": {"PTS": 25.4, "REB": 5.1, "AST": 8.4},
        "Tyrese Haliburton": {"PTS": 21.4, "REB": 3.9, "AST": 10.4},
        "LaMelo Ball": {"PTS": 26.9, "REB": 5.4, "AST": 8.2},
        "Zach LaVine": {"PTS": 23.4, "REB": 4.8, "AST": 4.1},
        "Nikola Vučević": {"PTS": 18.2, "REB": 10.8, "AST": 3.4}
    }

def fetch_injured_players():
    """Pulls live injured/out player names from ESPN's public scoreboard & team injury feeds."""
    injured_set = set()
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        res = requests.get(url, timeout=10).json()
        for event in res.get("events", []):
            for team in event["competitions"][0]["competitors"]:
                for inj in team.get("injuries", []):
                    p_name = inj.get("athlete", {}).get("displayName")
                    status = inj.get("status", "").lower()
                    if p_name and any(term in status for term in ["out", "doubtful", "questionable", "injured"]):
                        injured_set.add(p_name)
    except:
        pass
    return injured_set

def post_alert_to_discord(base_text):
    res = requests.post(f"{DISCORD_WEBHOOK_URL}?wait=true", json={"content": f"⏳ {base_text}"})
    if res.status_code in [200, 204]:
        return res.json()["id"]
    return None

def scan_all_fanduel_markets():
    tracked_bets = load_json(TRACKER_FILE, [])
    already_posted = {b["bet_key"] for b in tracked_bets}
    scanned_events = set(load_json(SCANNED_EVENTS_FILE, []))
    injured_players = fetch_injured_players()
    
    stats_db = get_team_efficiency_data()
    player_db = get_player_baselines()
    avg_off = np.mean([v["off"] for v in stats_db.values()])
    avg_pace = np.mean([v["pace"] for v in stats_db.values()])
    
    central_tz = pytz.timezone("America/Chicago")
    now_utc = datetime.now(timezone.utc)
    today_ct_str = now_utc.astimezone(central_tz).strftime("%Y-%m-%d")
    
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY.strip(),
        "regions": "us",
        "markets": "totals,spreads",
        "bookmakers": "fanduel",
        "oddsFormat": "american"
    }
    
    res = requests.get(url, params=params, timeout=10)
    print(f"📊 FanDuel Sync for {today_ct_str}! (Credits Remaining: {res.headers.get('x-requests-remaining')})")
    if res.status_code != 200:
        return
        
    events = res.json()
    deep_scans_done = 0
    
    for g in events:
        if not g.get("bookmakers"):
            continue
        dt_utc = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        if dt_utc <= now_utc:
            continue
            
        date_str = dt_utc.astimezone(central_tz).strftime("%Y-%m-%d")
        
        # STRICT TODAY-ONLY FILTER (Ignores tomorrow's games)
        if date_str != today_ct_str:
            continue
            
        event_id = g["id"]
        home, away = g["home_team"], g["away_team"]
        time_ct = dt_utc.astimezone(central_tz).strftime("%-I:%M %p CT")
        match_key = f"{date_str}_{away}_vs_{home}"
        
        h_stats = stats_db.get(home, {"off": avg_off, "def": avg_off, "pace": avg_pace})
        a_stats = stats_db.get(away, {"off": avg_off, "def": avg_off, "pace": avg_pace})
        
        exp_pace = (h_stats["pace"] * a_stats["pace"]) / avg_pace
        proj_h_ortg = (h_stats["off"] * a_stats["def"]) / avg_off
        proj_a_ortg = (a_stats["off"] * h_stats["def"]) / avg_off
        
        home_pts = (proj_h_ortg * (exp_pace / 100.0)) + (HOME_COURT_ADVANTAGE / 2.0)
        away_pts = (proj_a_ortg * (exp_pace / 100.0)) - (HOME_COURT_ADVANTAGE / 2.0)
        
        model_total = home_pts + away_pts
        model_margin = home_pts - away_pts
        
        # --- A. FULL-GAME TOTALS & SPREADS ---
        if SCAN_FULL_GAME:
            for m in g["bookmakers"][0].get("markets", []):
                if m["key"] == "totals":
                    bet_key = f"{match_key}_FG_TOTAL"
                    if bet_key not in already_posted:
                        over_o = next((o for o in m["outcomes"] if o["name"] == "Over"), None)
                        under_o = next((o for o in m["outcomes"] if o["name"] == "Under"), None)
                        if over_o and under_o:
                            line = float(over_o["point"])
                            diff = model_total - line
                            pick_dir = "Over" if diff >= 0 else "Under"
                            juice = int(over_o["price"]) if diff >= 0 else int(under_o["price"])
                            win_prob = norm.cdf(abs(diff) / NBA_STD_DEV_TOTAL)
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                base_text = f"🏀 NBA: {away} vs {home}\n{time_ct} - Total: {line} {pick_dir} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV"
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "FG_TOTAL", "line": line, "pick_dir": pick_dir, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

                elif m["key"] == "spreads":
                    bet_key = f"{match_key}_FG_SPREAD"
                    if bet_key not in already_posted:
                        home_o = next((o for o in m["outcomes"] if o["name"] == home), None)
                        away_o = next((o for o in m["outcomes"] if o["name"] == away), None)
                        if home_o and away_o:
                            h_line = float(home_o["point"])
                            sp_diff = model_margin - (-h_line)
                            pick_team = home if sp_diff >= 0 else away
                            pick_line = h_line if sp_diff >= 0 else float(away_o["point"])
                            juice = int(home_o["price"]) if sp_diff >= 0 else int(away_o["price"])
                            win_prob = norm.cdf(abs(sp_diff) / NBA_STD_DEV_SPREAD)
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                base_text = f"🏀 NBA: {away} vs {home}\n{time_ct} - Spread: {pick_team} {pick_line:+.1f} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV"
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "FG_SPREAD", "pick_team": pick_team, "line": pick_line, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

        # --- B. DEEP SCAN: 1H / 1Q & PLAYER PROPS ---
        if (SCAN_HALVES_QUARTERS or SCAN_PLAYER_PROPS) and (event_id not in scanned_events) and (deep_scans_done < MAX_DEEP_SCAN_GAMES_PER_RUN):
            extra_markets = []
            if SCAN_HALVES_QUARTERS:
                extra_markets.extend(["totals_h1", "spreads_h1", "totals_q1", "spreads_q1"])
            if SCAN_PLAYER_PROPS:
                extra_markets.extend(["player_points", "player_rebounds", "player_assists"])
                
            ev_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            ev_params = {
                "apiKey": THE_ODDS_API_KEY.strip(),
                "regions": "us",
                "markets": ",".join(extra_markets),
                "bookmakers": "fanduel",
                "oddsFormat": "american"
            }
            ev_res = requests.get(ev_url, params=ev_params, timeout=10)
            if ev_res.status_code == 200:
                deep_scans_done += 1
                scanned_events.add(event_id)
                ev_data = ev_res.json()
                
                if ev_data.get("bookmakers"):
                    for m in ev_data["bookmakers"][0].get("markets", []):
                        m_key = m["key"]
                        
                        if m_key in ["totals_h1", "totals_q1"]:
                            period_label = "1H Total" if m_key == "totals_h1" else "1Q Total"
                            scale = 0.505 if m_key == "totals_h1" else 0.254
                            std_dev = NBA_STD_DEV_H1_TOTAL if m_key == "totals_h1" else NBA_STD_DEV_Q1_TOTAL
                            bet_key = f"{match_key}_{m_key}"
                            
                            over_o = next((o for o in m["outcomes"] if o["name"] == "Over"), None)
                            under_o = next((o for o in m["outcomes"] if o["name"] == "Under"), None)
                            if over_o and under_o and bet_key not in already_posted:
                                line = float(over_o["point"])
                                diff = (model_total * scale) - line
                                pick_dir = "Over" if diff >= 0 else "Under"
                                juice = int(over_o["price"]) if diff >= 0 else int(under_o["price"])
                                win_prob = norm.cdf(abs(diff) / std_dev)
                                edge = win_prob - american_to_implied_prob(juice)
                                if edge >= MIN_EDGE_TO_POST:
                                    ev_pct = edge * 100.0
                                    base_text = f"🏀 NBA: {away} vs {home}\n{time_ct} - {period_label}: {line} {pick_dir} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV"
                                    msg_id = post_alert_to_discord(base_text)
                                    if msg_id:
                                        tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": m_key.upper(), "line": line, "pick_dir": pick_dir, "base_text": base_text, "status": "PENDING"})
                                        already_posted.add(bet_key)

                        elif m_key in ["player_points", "player_rebounds", "player_assists"]:
                            stat_code = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST"}[m_key]
                            players_in_market = {}
                            for o in m["outcomes"]:
                                p_name = o.get("description", "")
                                if p_name not in players_in_market:
                                    players_in_market[p_name] = {}
                                players_in_market[p_name][o["name"]] = o
                                
                            for p_name, sides in players_in_market.items():
                                # SKIP IF PLAYER IS ON INJURY REPORT!
                                if p_name in injured_players or p_name not in player_db or "Over" not in sides or "Under" not in sides:
                                    continue
                                bet_key = f"{match_key}_{p_name}_{stat_code}"
                                if bet_key in already_posted:
                                    continue
                                    
                                line = float(sides["Over"]["point"])
                                base_avg = player_db[p_name][stat_code]
                                def_mult = ((h_stats["def"] + a_stats["def"]) / 2.0) / avg_off
                                lambda_proj = base_avg * def_mult
                                
                                prob_under = poisson.cdf(int(np.floor(line)), lambda_proj)
                                prob_over = 1.0 - prob_under
                                
                                if prob_over >= prob_under:
                                    pick_dir, win_prob, juice = "Over", prob_over, int(sides["Over"]["price"])
                                else:
                                    pick_dir, win_prob, juice = "Under", prob_under, int(sides["Under"]["price"])
                                    
                                edge = win_prob - american_to_implied_prob(juice)
                                if edge >= MIN_EDGE_TO_POST:
                                    ev_pct = edge * 100.0
                                    base_text = f"🏀 NBA Prop: {p_name} ({away} vs {home})\n{time_ct} - {stat_code}: {line} {pick_dir} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV"
                                    msg_id = post_alert_to_discord(base_text)
                                    if msg_id:
                                        tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "PLAYER_PROP", "player": p_name, "stat": stat_code, "line": line, "pick_dir": pick_dir, "base_text": base_text, "status": "PENDING"})
                                        already_posted.add(bet_key)

    save_json(TRACKER_FILE, tracked_bets)
    save_json(SCANNED_EVENTS_FILE, list(scanned_events))

def check_scores_and_grade_all_markets():
    tracked_bets = load_json(TRACKER_FILE, [])
    pending = [b for b in tracked_bets if b["status"] == "PENDING"]
    if not pending:
        return

    espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        res = requests.get(espn_url, timeout=10).json()
    except:
        return

    updated = False
    for event in res.get("events", []):
        is_final = event["status"]["type"]["completed"]
        period = event["status"].get("period", 0)
        comp = event["competitions"][0]
        competitors = comp["competitors"]
        
        home_c = next(c for c in competitors if c["homeAway"] == "home")
        away_c = next(c for c in competitors if c["homeAway"] == "away")
        
        h_name = "Los Angeles Clippers" if home_c["team"]["displayName"] == "LA Clippers" else home_c["team"]["displayName"]
        a_name = "Los Angeles Clippers" if away_c["team"]["displayName"] == "LA Clippers" else away_c["team"]["displayName"]
        
        h_pts = float(home_c.get("score", 0))
        a_pts = float(away_c.get("score", 0))
        
        h_lines = [float(x.get("value", 0)) for x in home_c.get("linescores", [])]
        a_lines = [float(x.get("value", 0)) for x in away_c.get("linescores", [])]
        
        player_stats_map = {}
        has_props = any(b["status"] == "PENDING" and b["market"] == "PLAYER_PROP" and b["home"] == h_name for b in tracked_bets)
        if is_final and has_props:
            sum_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event['id']}"
            try:
                sum_data = requests.get(sum_url, timeout=10).json()
                for team_box in sum_data.get("boxscore", {}).get("players", []):
                    stat_Block = team_box["statistics"][0]
                    labels = stat_Block["labels"]
                    pts_idx = labels.index("PTS") if "PTS" in labels else -1
                    reb_idx = labels.index("REB") if "REB" in labels else -1
                    ast_idx = labels.index("AST") if "AST" in labels else -1
                    for ath in stat_Block.get("athletes", []):
                        p_disp = ath["athlete"]["displayName"]
                        s_vals = ath.get("stats", [])
                        if s_vals:
                            player_stats_map[p_disp] = {
                                "PTS": float(s_vals[pts_idx]) if pts_idx != -1 else 0,
                                "REB": float(s_vals[reb_idx]) if reb_idx != -1 else 0,
                                "AST": float(s_vals[ast_idx]) if ast_idx != -1 else 0
                            }
            except:
                pass

        for bet in tracked_bets:
            if bet["status"] != "PENDING" or bet["home"] != h_name or bet["away"] != a_name:
                continue
                
            did_win = None
            mkt = bet["market"]
            
            if mkt == "TOTALS_Q1" and (period > 1 or is_final) and len(h_lines) >= 1 and len(a_lines) >= 1:
                q1_tot = h_lines[0] + a_lines[0]
                did_win = (q1_tot > bet["line"]) if bet["pick_dir"] == "Over" else (q1_tot < bet["line"])
            elif mkt == "TOTALS_H1" and (period > 2 or is_final) and len(h_lines) >= 2 and len(a_lines) >= 2:
                h1_tot = sum(h_lines[:2]) + sum(a_lines[:2])
                did_win = (h1_tot > bet["line"]) if bet["pick_dir"] == "Over" else (h1_tot < bet["line"])
            elif mkt == "FG_TOTAL" and is_final:
                fg_tot = h_pts + a_pts
                did_win = (fg_tot > bet["line"]) if bet["pick_dir"] == "Over" else (fg_tot < bet["line"])
            elif mkt == "FG_SPREAD" and is_final:
                if bet["pick_team"] == h_name:
                    did_win = (h_pts + bet["line"]) > a_pts
                else:
                    did_win = (a_pts + bet["line"]) > h_pts
            elif mkt == "PLAYER_PROP" and is_final and bet["player"] in player_stats_map:
                actual_stat = player_stats_map[bet["player"]][bet["stat"]]
                did_win = (actual_stat > bet["line"]) if bet["pick_dir"] == "Over" else (actual_stat < bet["line"])

            if did_win is not None:
                icon = "✅" if did_win else "❌"
                patch_res = requests.patch(f"{DISCORD_WEBHOOK_URL}/messages/{bet['message_id']}", json={"content": f"{icon} {bet['base_text']}"})
                if patch_res.status_code == 200:
                    bet["status"] = "GRADED"
                    updated = True

    if updated:
        save_json(TRACKER_FILE, tracked_bets)

if __name__ == "__main__":
    print("🤖 TODAY-ONLY 4.0%+ EV FANDUEL BOT IS LIVE!")
    last_scan = 0
    while True:
        now_ts = time.time()
        if (now_ts - last_scan) >= FANDUEL_SCAN_SECONDS:
            scan_all_fanduel_markets()
            last_scan = now_ts
        check_scores_and_grade_all_markets()
        time.sleep(SCORE_CHECK_SECONDS)
