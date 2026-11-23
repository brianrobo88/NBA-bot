import sys
import time
import json
import os
import unicodedata
import requests
import numpy as np
from scipy.stats import norm, poisson
from datetime import datetime, timedelta, timezone
import pytz

# ==========================================
# 1. CREDENTIALS & V16 CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

NEW_PAID_ODDS_KEY = "b7d67c68e4dd79b7bc22fb02f8d82e73"
_env_odds_key = os.getenv("THE_ODDS_API_KEY", "").strip()
THE_ODDS_API_KEY = (
    NEW_PAID_ODDS_KEY
    if (not _env_odds_key or _env_odds_key == "9ecb23956a7654b951782fd28ec3a8b5")
    else _env_odds_key
)

# Market Toggles
SCAN_FULL_GAME = True         # Full-Game Spreads & Totals
SCAN_HALVES_QUARTERS = True   # 1st Half & 1st Quarter Spreads & Totals
SCAN_PLAYER_PROPS = True      # Core Starters Only: PTS, REB, AST, PRA, 3PM (Matchup + Form)
SCAN_PRA_AND_3PM = True       # Scans PRA & 3PM alongside PTS/REB/AST
SCAN_Q1_PLAYER_PROPS = True   # 1st Quarter Starter Props: PTS, REB, AST (7/10 L10 Filter)

# Calibrated EV, Dislocation, Hit-Rate, Juice & Volume Controls
MIN_EDGE_TEAM_MARKETS = 0.110 # 11.0% EV for FG Spreads & 1Q/1H Markets (75/25 blend)
MIN_EDGE_FG_TOTAL = 0.050     # 5.0% Calibrated EV for FG Totals (60/40 Pinnacle blend + 6.5-pt cap)
MAX_FG_TOTAL_DISLOCATION = 6.5 # Blocks FG Totals disagreeing with consensus by > 6.5 pts
MIN_EDGE_PLAYER_PROPS = 0.065 # 6.5% Calibrated EV for FG Props (60/40 Pinnacle blend + 22% gap cap)
MAX_FG_PROP_GAP_PCT = 0.22    # Blocks FG Props where projection deviates > 22% from line
MIN_Q1_L10_HITS = 7           # Strictly for 1Q Starter Props (>= 7 hits in Last 10 games)
MAX_FG_PROP_JUICE = -140      # Blocks heavily juiced FG props (e.g., -146)
MAX_Q1_PROP_JUICE = -165      # Blocks heavily juiced 1Q props (e.g., -210, -550)
MAX_FG_PROPS_PER_DAY = 5      # Caps daily slate to top 5 highest-EV FG props
MAX_FG_PROP_UNITS = 1.0       # Caps FG prop Kelly sizing at 1.0u max

# 10 Valid Bookmakers across US / US2 / EU = exactly 1 region credit in The Odds API
# (Do NOT pass 'regions' param when specifying cross-region 'bookmakers')
BOOKMAKERS_LIST = (
    "fanduel,draftkings,betmgm,williamhill_us,fanatics,"
    "espnbet,betrivers,hardrockbet,bovada,pinnacle"
)
PLAYABLE_US_BOOKS = {
    "fanduel", "draftkings", "betmgm", "williamhill_us",
    "fanatics", "espnbet", "betrivers", "hardrockbet", "bovada"
}
BOOK_LABELS = {
    "fanduel": "FD",
    "draftkings": "DK",
    "betmgm": "MGM",
    "williamhill_us": "CZR",
    "fanatics": "FAN",
    "espnbet": "ESPN",
    "betrivers": "BR",
    "hardrockbet": "HR",
    "bovada": "BOV",
    "pinnacle": "PIN"
}

MAX_DEEP_SCAN_GAMES_PER_RUN = 15 # Scans full slate (up to 15 games) per pass
LOOKBACK_DAYS_FOR_STATS = 30     # 30-day window ensures full 10-game (L10) & split logs

# Minimum Prop Line & Minutes Filters
MIN_PROP_LINE_PTS = 7.5
MIN_PROP_LINE_OTHER = 3.5
MIN_PROP_LINE_3PM = 1.5
MIN_FG_PROP_AVG_MINUTES = 24.0   # Core 24+ MPG starters only for Full-Game props
MIN_STARTER_AVG_MINUTES = 22.0   # Verified starter threshold for 1Q props

TRACKER_FILE = "tracked_all_markets.json"
LIVE_STATS_CACHE_FILE = "live_nba_stats_cache.json"
RECAP_POSTED_FILE = "daily_recaps_posted.json"
CACHE_SCHEMA_VERSION = "v16_live_splits_pace_dvp_fg_props"

SCORE_CHECK_SECONDS = 300     # Checks ESPN scores every 5 mins (FREE)
FANDUEL_SCAN_SECONDS = 2700   # Rescans lines & injury reports every 45 mins until tip-off
MAX_RUNTIME_SECONDS = 19800   # Clean exit at 5h 30m so GitHub Actions commits JSON state before 6h kill

NBA_STD_DEV_TOTAL = 16.0
NBA_STD_DEV_SPREAD = 12.5
NBA_STD_DEV_H1_TOTAL = 11.5
NBA_STD_DEV_H1_SPREAD = 8.8
NBA_STD_DEV_Q1_TOTAL = 8.2
NBA_STD_DEV_Q1_SPREAD = 6.2
DEFAULT_HCA = 2.2
SPREAD_SHRINKAGE = 0.78       # Applied to true Net Rating differential (no double-halving)
PERIOD_TOTAL_SHRINKAGE = 0.75 # Regresses 1Q/1H totals toward league average

# Venue-Specific Home Court Advantage (Altitude & Elite Home Tiers)
TEAM_HCA = {
    "Denver Nuggets": 3.6,
    "Utah Jazz": 3.1,
    "Boston Celtics": 2.6,
    "Oklahoma City Thunder": 2.5,
    "Minnesota Timberwolves": 2.4,
    "Cleveland Cavaliers": 2.4,
    "Golden State Warriors": 2.4,
    "New York Knicks": 2.4
}

DIVIDER = "⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯"

TEAM_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

def get_abbr(team_name):
    return TEAM_ABBR.get(team_name, team_name[:3].upper())

def get_team_hca(home_team):
    return TEAM_HCA.get(home_team, DEFAULT_HCA)

def normalize_name(name):
    if not name:
        return ""
    clean = unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("utf-8")
    clean = clean.replace(".", "").replace("'", "").strip().lower()
    for suffix in [" jr", " sr", " iii", " ii", " iv"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)].strip()
    return clean

def normalize_team_name(name):
    if not name:
        return ""
    raw = str(name).strip()
    if raw == "LA Clippers":
        return "Los Angeles Clippers"
    if raw in TEAM_ABBR and raw != "LA Clippers":
        return raw
    raw_lower = raw.lower()
    abbr_overrides = {
        "gs": "Golden State Warriors",
        "no": "New Orleans Pelicans",
        "ny": "New York Knicks",
        "sa": "San Antonio Spurs",
        "utah": "Utah Jazz",
        "wsh": "Washington Wizards"
    }
    if raw_lower in abbr_overrides:
        return abbr_overrides[raw_lower]
    for full_name, abbr in TEAM_ABBR.items():
        if full_name == "LA Clippers":
            continue
        if raw.upper() == abbr or raw_lower == full_name.lower() or full_name.lower().endswith(" " + raw_lower):
            return full_name
    return raw

def parse_minutes(val):
    if val is None:
        return 0.0
    try:
        s = str(val).strip()
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + (float(parts[1]) / 60.0)
        return float(s)
    except:
        return 0.0

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

def profit_per_unit(odds_val):
    try:
        odds = float(odds_val)
        return (100.0 / abs(odds)) if odds < 0 else (odds / 100.0)
    except:
        return 0.9091

def calc_kelly_units(win_prob, odds_val, max_units=2.0):
    """Calculates Quarter-Kelly unit recommendation bounded between 0.5u and max_units."""
    try:
        b = profit_per_unit(odds_val)
        q = 1.0 - win_prob
        kelly_f = max(0.0, (b * win_prob - q) / b)
        units = round(kelly_f * 0.25 * 20.0, 1)
        return max(0.5, min(max_units, units))
    except:
        return min(1.0, max_units)

def extract_fanduel_and_consensus(bookmakers_list):
    """Separates FanDuel markets from other major books to build de-vigged consensus."""
    fd_markets = []
    other_books = []
    for b in bookmakers_list or []:
        if b.get("key") == "fanduel":
            fd_markets = b.get("markets", [])
        else:
            other_books.append(b)
    return fd_markets, other_books

def get_devigged_total_or_prop_consensus(other_books, market_key, pick_dir, fd_line, player_name=None):
    """
    Returns (weighted_consensus_line, weighted_devigged_win_prob) across comparison books.
    Gives Pinnacle 2.0x sharp weight when available.
    """
    pts = []
    pts_w = []
    devig_probs = []
    probs_w = []
    target_norm = normalize_name(player_name) if player_name else None

    for b in other_books:
        b_weight = 2.0 if b.get("key") == "pinnacle" else 1.0
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            over_o = None
            under_o = None
            for o in m.get("outcomes", []):
                if target_norm and normalize_name(o.get("description", "")) != target_norm:
                    continue
                if o.get("name") == "Over" and "point" in o:
                    over_o = o
                elif o.get("name") == "Under" and "point" in o:
                    under_o = o
            if over_o and under_o:
                b_line = float(over_o["point"])
                pts.append(b_line)
                pts_w.append(b_weight)
                p_over_raw = american_to_implied_prob(over_o["price"])
                p_under_raw = american_to_implied_prob(under_o["price"])
                tot_vig = p_over_raw + p_under_raw
                if tot_vig > 0:
                    fair_over = p_over_raw / tot_vig
                    fair_pick = fair_over if pick_dir == "Over" else (1.0 - fair_over)
                    line_diff = b_line - fd_line
                    if abs(line_diff) <= 1.5:
                        adj = (line_diff * 0.035) if pick_dir == "Over" else (-line_diff * 0.035)
                        devig_probs.append(float(np.clip(fair_pick + adj, 0.05, 0.95)))
                        probs_w.append(b_weight)

    avg_line = round(float(np.average(pts, weights=pts_w)), 1) if pts else None
    avg_prob = float(np.average(devig_probs, weights=probs_w)) if devig_probs else None
    return avg_line, avg_prob

def get_devigged_spread_consensus(other_books, market_key, pick_team, fd_line):
    """
    Returns (weighted_consensus_spread, weighted_devigged_win_prob) across comparison books.
    Gives Pinnacle 2.0x sharp weight when available.
    """
    pts = []
    pts_w = []
    devig_probs = []
    probs_w = []
    for b in other_books:
        b_weight = 2.0 if b.get("key") == "pinnacle" else 1.0
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            pick_o = None
            opp_o = None
            for o in m.get("outcomes", []):
                if "point" not in o:
                    continue
                if normalize_team_name(o.get("name", "")) == pick_team:
                    pick_o = o
                else:
                    opp_o = o
            if pick_o and opp_o:
                b_line = float(pick_o["point"])
                pts.append(b_line)
                pts_w.append(b_weight)
                p_pick = american_to_implied_prob(pick_o["price"])
                p_opp = american_to_implied_prob(opp_o["price"])
                tot_vig = p_pick + p_opp
                if tot_vig > 0:
                    fair_pick = p_pick / tot_vig
                    line_diff = fd_line - b_line
                    if abs(line_diff) <= 1.5:
                        devig_probs.append(float(np.clip(fair_pick + (line_diff * 0.03), 0.05, 0.95)))
                        probs_w.append(b_weight)

    avg_line = round(float(np.average(pts, weights=pts_w)), 1) if pts else None
    avg_prob = float(np.average(devig_probs, weights=probs_w)) if devig_probs else None
    return avg_line, avg_prob

def extract_game_pace_48(sum_data, reg_scale):
    """
    Computes 48-minute-normalized NBA possessions (FGA + 0.44*FTA - OREB + TO)
    from ESPN's team boxscore (falling back to summing player boxscores if needed).
    """
    poss_list = []
    for tb in sum_data.get("boxscore", {}).get("teams", []):
        s_map = {s.get("name", ""): str(s.get("displayValue", "")) for s in tb.get("statistics", [])}
        fg_str = s_map.get("fieldGoalsMade-fieldGoalsAttempted", "")
        ft_str = s_map.get("freeThrowsMade-freeThrowsAttempted", "")
        oreb_str = s_map.get("offensiveRebounds", "")
        to_str = s_map.get("totalTurnovers") or s_map.get("turnovers", "")
        if "-" in fg_str and "-" in ft_str:
            try:
                fga = float(fg_str.split("-")[1])
                fta = float(ft_str.split("-")[1])
                oreb = float(oreb_str) if oreb_str else 0.0
                tov = float(to_str) if to_str else 0.0
                poss = fga + (0.44 * fta) - oreb + tov
                if 65.0 <= poss <= 135.0:
                    poss_list.append(poss)
            except:
                continue

    if len(poss_list) == 2:
        return float(np.clip(np.mean(poss_list) * reg_scale, 88.0, 112.0))

    # Fallback: sum player boxscores per team
    poss_list = []
    for tb in sum_data.get("boxscore", {}).get("players", []):
        stat_blk = (tb.get("statistics") or [{}])[0]
        labels = stat_blk.get("labels", [])
        fg_idx = labels.index("FG") if "FG" in labels else -1
        ft_idx = labels.index("FT") if "FT" in labels else -1
        oreb_idx = labels.index("OREB") if "OREB" in labels else -1
        to_idx = labels.index("TO") if "TO" in labels else -1
        if fg_idx == -1 or ft_idx == -1:
            continue
        fga_sum, fta_sum, oreb_sum, to_sum = 0.0, 0.0, 0.0, 0.0
        for ath in stat_blk.get("athletes", []):
            s_vals = ath.get("stats", [])
            if not s_vals or ath.get("didNotPlay"):
                continue
            try:
                if "-" in str(s_vals[fg_idx]):
                    fga_sum += float(str(s_vals[fg_idx]).split("-")[1])
                if "-" in str(s_vals[ft_idx]):
                    fta_sum += float(str(s_vals[ft_idx]).split("-")[1])
                if oreb_idx != -1:
                    oreb_sum += float(s_vals[oreb_idx])
                if to_idx != -1:
                    to_sum += float(s_vals[to_idx])
            except:
                continue
        poss = fga_sum + (0.44 * fta_sum) - oreb_sum + to_sum
        if 65.0 <= poss <= 135.0:
            poss_list.append(poss)

    if len(poss_list) == 2:
        return float(np.clip(np.mean(poss_list) * reg_scale, 88.0, 112.0))
    return None

def extract_q1_player_stats_from_summary(sum_data):
    """
    Parses ESPN's play-by-play ('plays') array for Period 1 to compute exact
    1st Quarter PTS, REB, and AST for every active player in the game.
    Prioritizes ESPN athlete IDs over play-by-play text to avoid name-abbreviation misses.
    """
    plays = sum_data.get("plays", [])
    p_boxes = sum_data.get("boxscore", {}).get("players", [])
    if not plays or not p_boxes:
        return {}

    id_to_norm = {}
    q1_stats = {}

    for tb in p_boxes:
        stat_blk = (tb.get("statistics") or [{}])[0]
        labels = stat_blk.get("labels", [])
        min_idx = labels.index("MIN") if "MIN" in labels else -1
        for ath in stat_blk.get("athletes", []):
            if ath.get("didNotPlay"):
                continue
            s_vals = ath.get("stats", [])
            if min_idx != -1 and s_vals:
                mins = parse_minutes(s_vals[min_idx])
                if mins <= 0.0:
                    continue
            ath_obj = ath.get("athlete", {})
            p_disp = ath_obj.get("displayName", "")
            ath_id = str(ath_obj.get("id", ""))
            if p_disp:
                norm_p = normalize_name(p_disp)
                q1_stats[norm_p] = {"PTS": 0.0, "REB": 0.0, "AST": 0.0}
                if ath_id:
                    id_to_norm[ath_id] = norm_p

    for play in plays:
        if play.get("period", {}).get("number") != 1:
            continue
        text = str(play.get("text", "")).strip()
        if not text:
            continue
        text_lower = text.lower()
        type_text = str(play.get("type", {}).get("text", "")).lower()
        parts = play.get("participants", [])

        # 1. Scoring Plays (PTS & AST)
        if play.get("scoringPlay"):
            pts_val = float(play.get("scoreValue") or 0)
            if pts_val <= 0:
                if "free throw" in text_lower or "technical" in text_lower:
                    pts_val = 1.0
                elif "three point" in text_lower or "3-pt" in text_lower:
                    pts_val = 3.0
                else:
                    pts_val = 2.0

            scorer_norm = ""
            if parts:
                scorer_id = str(parts[0].get("athlete", {}).get("id", ""))
                scorer_norm = id_to_norm.get(scorer_id, "")
            if not scorer_norm and " makes " in text:
                scorer_norm = normalize_name(text.split(" makes ")[0])
            if scorer_norm in q1_stats:
                q1_stats[scorer_norm]["PTS"] += pts_val

            if "assist" in text_lower:
                assister_norm = ""
                if len(parts) >= 2:
                    ast_id = str(parts[1].get("athlete", {}).get("id", ""))
                    assister_norm = id_to_norm.get(ast_id, "")
                if not assister_norm and "(" in text:
                    inside_paren = text.split("(")[-1].split(")")[0]
                    ast_raw = (
                        inside_paren.replace(" assists", "")
                        .replace(" Assists", "")
                        .replace(" assist", "")
                        .replace(" Assist", "")
                        .strip()
                    )
                    assister_norm = normalize_name(ast_raw)
                if assister_norm in q1_stats:
                    q1_stats[assister_norm]["AST"] += 1.0

        # 2. Rebounds (Offensive & Defensive)
        elif "rebound" in type_text or "rebound" in text_lower:
            if "team rebound" in text_lower or "team" in type_text:
                continue
            reb_norm = ""
            if parts:
                reb_id = str(parts[0].get("athlete", {}).get("id", ""))
                reb_norm = id_to_norm.get(reb_id, "")
            if not reb_norm:
                reb_raw = (
                    text.split(" defensive rebound")[0]
                    .split(" offensive rebound")[0]
                    .split(" Defensive Rebound")[0]
                    .split(" Offensive Rebound")[0]
                    .strip()
                )
                reb_norm = normalize_name(reb_raw)
            if reb_norm in q1_stats:
                q1_stats[reb_norm]["REB"] += 1.0

    return q1_stats

def get_team_efficiency_baselines():
    return {
        "San Antonio Spurs": {"off": 119.4, "def": 107.5, "pace": 100.8, "opp_reb": 42.1, "opp_ast": 24.8, "q1_off": 30.2, "q1_def": 27.5, "h1_off": 60.5, "h1_def": 54.2},
        "Houston Rockets": {"off": 119.2, "def": 109.7, "pace": 99.5, "opp_reb": 41.5, "opp_ast": 24.2, "q1_off": 29.8, "q1_def": 27.1, "h1_off": 59.8, "h1_def": 54.8},
        "Golden State Warriors": {"off": 121.3, "def": 111.7, "pace": 101.2, "opp_reb": 44.2, "opp_ast": 26.1, "q1_off": 31.4, "q1_def": 28.2, "h1_off": 62.1, "h1_def": 56.0},
        "Minnesota Timberwolves": {"off": 117.2, "def": 106.0, "pace": 98.4, "opp_reb": 41.2, "opp_ast": 23.9, "q1_off": 29.1, "q1_def": 26.4, "h1_off": 58.4, "h1_def": 53.0},
        "Oklahoma City Thunder": {"off": 116.8, "def": 105.6, "pace": 100.5, "opp_reb": 45.8, "opp_ast": 24.0, "q1_off": 29.8, "q1_def": 26.2, "h1_off": 59.2, "h1_def": 52.8},
        "Los Angeles Lakers": {"off": 122.3, "def": 114.7, "pace": 101.4, "opp_reb": 44.6, "opp_ast": 27.4, "q1_off": 31.0, "q1_def": 28.9, "h1_off": 61.8, "h1_def": 57.5},
        "Portland Trail Blazers": {"off": 119.6, "def": 113.8, "pace": 99.8, "opp_reb": 44.9, "opp_ast": 26.8, "q1_off": 29.5, "q1_def": 28.4, "h1_off": 59.2, "h1_def": 56.8},
        "Memphis Grizzlies": {"off": 117.3, "def": 110.2, "pace": 100.9, "opp_reb": 43.2, "opp_ast": 25.4, "q1_off": 29.6, "q1_def": 27.8, "h1_off": 59.0, "h1_def": 55.2},
        "Dallas Mavericks": {"off": 115.8, "def": 111.0, "pace": 99.2, "opp_reb": 43.8, "opp_ast": 25.9, "q1_off": 28.9, "q1_def": 27.9, "h1_off": 58.0, "h1_def": 55.6},
        "Denver Nuggets": {"off": 116.0, "def": 115.6, "pace": 98.1, "opp_reb": 42.6, "opp_ast": 26.5, "q1_off": 29.2, "q1_def": 28.8, "h1_off": 58.5, "h1_def": 57.8},
        "Detroit Pistons": {"off": 114.3, "def": 106.2, "pace": 99.0, "opp_reb": 42.0, "opp_ast": 24.5, "q1_off": 28.4, "q1_def": 26.6, "h1_off": 57.0, "h1_def": 53.2},
        "Boston Celtics": {"off": 119.6, "def": 108.6, "pace": 98.7, "opp_reb": 42.4, "opp_ast": 24.1, "q1_off": 30.5, "q1_def": 27.0, "h1_off": 60.8, "h1_def": 54.0},
        "Los Angeles Clippers": {"off": 111.9, "def": 116.0, "pace": 97.8, "opp_reb": 43.5, "opp_ast": 26.2, "q1_off": 27.8, "q1_def": 29.1, "h1_off": 55.8, "h1_def": 58.0},
        "Phoenix Suns": {"off": 112.6, "def": 114.5, "pace": 99.1, "opp_reb": 44.2, "opp_ast": 26.4, "q1_off": 28.1, "q1_def": 28.6, "h1_off": 56.2, "h1_def": 57.5},
        "Toronto Raptors": {"off": 116.6, "def": 114.0, "pace": 100.2, "opp_reb": 44.0, "opp_ast": 26.7, "q1_off": 29.0, "q1_def": 28.5, "h1_off": 58.2, "h1_def": 57.0},
        "Atlanta Hawks": {"off": 115.5, "def": 113.3, "pace": 101.8, "opp_reb": 45.3, "opp_ast": 27.8, "q1_off": 29.4, "q1_def": 28.6, "h1_off": 58.8, "h1_def": 56.6},
        "New York Knicks": {"off": 114.6, "def": 108.5, "pace": 96.9, "opp_reb": 40.8, "opp_ast": 24.3, "q1_off": 28.6, "q1_def": 27.1, "h1_off": 57.2, "h1_def": 54.2},
        "Cleveland Cavaliers": {"off": 115.9, "def": 114.5, "pace": 98.5, "opp_reb": 43.1, "opp_ast": 25.6, "q1_off": 29.0, "q1_def": 28.6, "h1_off": 58.0, "h1_def": 57.2},
        "New Orleans Pelicans": {"off": 113.2, "def": 118.6, "pace": 99.4, "opp_reb": 45.6, "opp_ast": 27.9, "q1_off": 28.2, "q1_def": 29.6, "h1_off": 56.4, "h1_def": 59.2},
        "Miami Heat": {"off": 111.8, "def": 116.9, "pace": 97.2, "opp_reb": 43.4, "opp_ast": 26.0, "q1_off": 27.9, "q1_def": 29.2, "h1_off": 55.6, "h1_def": 58.4},
        "Utah Jazz": {"off": 104.4, "def": 116.0, "pace": 99.6, "opp_reb": 46.2, "opp_ast": 28.4, "q1_off": 26.2, "q1_def": 29.0, "h1_off": 52.2, "h1_def": 58.0},
        "Philadelphia 76ers": {"off": 111.3, "def": 108.3, "pace": 98.0, "opp_reb": 43.0, "opp_ast": 25.0, "q1_off": 27.8, "q1_def": 27.1, "h1_off": 55.5, "h1_def": 54.1},
        "Sacramento Kings": {"off": 108.5, "def": 121.6, "pace": 100.4, "opp_reb": 44.8, "opp_ast": 27.5, "q1_off": 27.2, "q1_def": 30.4, "h1_off": 54.2, "h1_def": 60.8},
        "Indiana Pacers": {"off": 110.4, "def": 114.3, "pace": 102.1, "opp_reb": 45.9, "opp_ast": 27.1, "q1_off": 27.6, "q1_def": 28.6, "h1_off": 55.0, "h1_def": 57.1},
        "Orlando Magic": {"off": 108.2, "def": 114.6, "pace": 98.3, "opp_reb": 41.9, "opp_ast": 24.6, "q1_off": 27.0, "q1_def": 28.6, "h1_off": 54.0, "h1_def": 57.3},
        "Washington Wizards": {"off": 109.0, "def": 118.2, "pace": 101.5, "opp_reb": 47.4, "opp_ast": 28.9, "q1_off": 27.4, "q1_def": 29.6, "h1_off": 54.6, "h1_def": 59.1},
        "Brooklyn Nets": {"off": 106.3, "def": 115.3, "pace": 97.6, "opp_reb": 44.5, "opp_ast": 26.4, "q1_off": 26.6, "q1_def": 28.8, "h1_off": 53.1, "h1_def": 57.6},
        "Milwaukee Bucks": {"off": 108.6, "def": 118.9, "pace": 99.3, "opp_reb": 45.2, "opp_ast": 27.2, "q1_off": 27.2, "q1_def": 29.8, "h1_off": 54.2, "h1_def": 59.5},
        "Charlotte Hornets": {"off": 106.5, "def": 114.6, "pace": 98.8, "opp_reb": 44.7, "opp_ast": 26.9, "q1_off": 26.6, "q1_def": 28.6, "h1_off": 53.2, "h1_def": 57.3},
        "Chicago Bulls": {"off": 108.6, "def": 121.6, "pace": 100.7, "opp_reb": 46.5, "opp_ast": 28.2, "q1_off": 27.2, "q1_def": 30.4, "h1_off": 54.2, "h1_def": 60.8}
    }

def get_fallback_player_baselines():
    raw_baselines = {
        "Jayson Tatum": {"PTS": 26.4, "REB": 8.5, "AST": 6.8, "POS": "F", "TEAM": "Boston Celtics"},
        "Jaylen Brown": {"PTS": 22.6, "REB": 6.6, "AST": 4.0, "POS": "G", "TEAM": "Philadelphia 76ers"},
        "Derrick White": {"PTS": 16.8, "REB": 4.3, "AST": 5.1, "POS": "G", "TEAM": "Boston Celtics"},
        "Jrue Holiday": {"PTS": 12.8, "REB": 4.8, "AST": 4.6, "POS": "G", "TEAM": "Boston Celtics"},
        "Kristaps Porzingis": {"PTS": 19.5, "REB": 7.1, "AST": 2.0, "POS": "C", "TEAM": "Boston Celtics"},
        "Payton Pritchard": {"PTS": 13.4, "REB": 3.2, "AST": 3.8, "POS": "G", "TEAM": "Boston Celtics"},
        "Al Horford": {"PTS": 8.6, "REB": 6.2, "AST": 2.6, "POS": "C", "TEAM": "Boston Celtics"},
        "Jalen Brunson": {"PTS": 26.6, "REB": 3.6, "AST": 6.8, "POS": "G", "TEAM": "New York Knicks"},
        "Karl-Anthony Towns": {"PTS": 24.5, "REB": 13.9, "AST": 3.1, "POS": "C", "TEAM": "New York Knicks"},
        "Mikal Bridges": {"PTS": 17.8, "REB": 3.9, "AST": 3.4, "POS": "F", "TEAM": "New York Knicks"},
        "OG Anunoby": {"PTS": 16.4, "REB": 5.1, "AST": 2.2, "POS": "F", "TEAM": "New York Knicks"},
        "Josh Hart": {"PTS": 6.5, "REB": 8.6, "AST": 5.2, "POS": "G", "TEAM": "New York Knicks"},
        "Shai Gilgeous-Alexander": {"PTS": 31.7, "REB": 5.6, "AST": 6.4, "POS": "G", "TEAM": "Oklahoma City Thunder"},
        "Jalen Williams": {"PTS": 21.4, "REB": 5.4, "AST": 5.1, "POS": "F", "TEAM": "Oklahoma City Thunder"},
        "Chet Holmgren": {"PTS": 18.2, "REB": 8.8, "AST": 2.6, "POS": "C", "TEAM": "Oklahoma City Thunder"},
        "Isaiah Hartenstein": {"PTS": 11.4, "REB": 10.8, "AST": 3.6, "POS": "C", "TEAM": "Oklahoma City Thunder"},
        "Damian Lillard": {"PTS": 20.4, "REB": 3.8, "AST": 4.9, "POS": "G", "TEAM": "Portland Trail Blazers"},
        "Donovan Clingan": {"PTS": 7.8, "REB": 10.7, "AST": 1.2, "POS": "C", "TEAM": "Portland Trail Blazers"},
        "Stephen Curry": {"PTS": 26.4, "REB": 4.5, "AST": 5.2, "POS": "G", "TEAM": "Golden State Warriors"},
        "Jimmy Butler": {"PTS": 18.8, "REB": 5.4, "AST": 4.8, "POS": "F", "TEAM": "Golden State Warriors"},
        "Kevin Durant": {"PTS": 27.4, "REB": 6.6, "AST": 4.8, "POS": "F", "TEAM": "Houston Rockets"},
        "Alperen Sengun": {"PTS": 21.6, "REB": 9.3, "AST": 7.0, "POS": "C", "TEAM": "Houston Rockets"},
        "Franz Wagner": {"PTS": 25.1, "REB": 5.7, "AST": 8.4, "POS": "F", "TEAM": "Orlando Magic"},
        "Wendell Carter Jr": {"PTS": 10.8, "REB": 8.7, "AST": 2.1, "POS": "C", "TEAM": "Orlando Magic"},
        "Walker Kessler": {"PTS": 10.4, "REB": 10.7, "AST": 1.4, "POS": "C", "TEAM": "Utah Jazz"},
        "LeBron James": {"PTS": 24.2, "REB": 7.4, "AST": 6.8, "POS": "F", "TEAM": "Philadelphia 76ers"},
        "Luka Doncic": {"PTS": 37.2, "REB": 9.1, "AST": 7.7, "POS": "G", "TEAM": "Los Angeles Lakers"},
        "Austin Reaves": {"PTS": 24.7, "REB": 4.2, "AST": 5.4, "POS": "G", "TEAM": "Los Angeles Lakers"},
        "Anthony Davis": {"PTS": 25.3, "REB": 10.1, "AST": 3.6, "POS": "C", "TEAM": "Washington Wizards"},
        "Trae Young": {"PTS": 24.2, "REB": 3.1, "AST": 8.6, "POS": "G", "TEAM": "Washington Wizards"},
        "Nikola Jokic": {"PTS": 29.1, "REB": 10.4, "AST": 9.1, "POS": "C", "TEAM": "Denver Nuggets"},
        "Jamal Murray": {"PTS": 24.9, "REB": 4.1, "AST": 6.5, "POS": "G", "TEAM": "Denver Nuggets"},
        "Anthony Edwards": {"PTS": 28.1, "REB": 5.8, "AST": 5.1, "POS": "G", "TEAM": "Minnesota Timberwolves"},
        "Rudy Gobert": {"PTS": 11.4, "REB": 10.9, "AST": 1.6, "POS": "C", "TEAM": "Minnesota Timberwolves"},
        "Devin Booker": {"PTS": 24.7, "REB": 4.6, "AST": 6.9, "POS": "G", "TEAM": "Phoenix Suns"},
        "Victor Wembanyama": {"PTS": 28.6, "REB": 11.9, "AST": 4.1, "POS": "C", "TEAM": "San Antonio Spurs"},
        "De'Aaron Fox": {"PTS": 24.6, "REB": 4.6, "AST": 6.1, "POS": "G", "TEAM": "San Antonio Spurs"},
        "Domantas Sabonis": {"PTS": 19.8, "REB": 13.4, "AST": 7.8, "POS": "C", "TEAM": "Sacramento Kings"},
        "Darius Acuff Jr": {"PTS": 25.7, "REB": 3.8, "AST": 7.3, "POS": "G", "TEAM": "Sacramento Kings"},
        "Giannis Antetokounmpo": {"PTS": 28.2, "REB": 12.6, "AST": 6.1, "POS": "F", "TEAM": "Miami Heat"},
        "Bam Adebayo": {"PTS": 20.2, "REB": 10.4, "AST": 4.1, "POS": "C", "TEAM": "Miami Heat"},
        "Davion Mitchell": {"PTS": 16.5, "REB": 3.2, "AST": 9.0, "POS": "G", "TEAM": "Miami Heat"},
        "Donovan Mitchell": {"PTS": 24.5, "REB": 4.4, "AST": 5.2, "POS": "G", "TEAM": "Cleveland Cavaliers"},
        "Darius Garland": {"PTS": 20.4, "REB": 2.8, "AST": 6.8, "POS": "G", "TEAM": "Cleveland Cavaliers"},
        "Tyrese Maxey": {"PTS": 26.5, "REB": 3.8, "AST": 6.4, "POS": "G", "TEAM": "Philadelphia 76ers"},
        "Joel Embiid": {"PTS": 19.5, "REB": 8.5, "AST": 3.5, "POS": "C", "TEAM": "Philadelphia 76ers"},
        "Cade Cunningham": {"PTS": 25.6, "REB": 6.4, "AST": 7.9, "POS": "G", "TEAM": "Detroit Pistons"},
        "Jalen Johnson": {"PTS": 19.6, "REB": 9.0, "AST": 7.5, "POS": "F", "TEAM": "Atlanta Hawks"},
        "Josh Giddey": {"PTS": 13.2, "REB": 6.8, "AST": 8.4, "POS": "G", "TEAM": "Chicago Bulls"},
        "Andrew Nembhard": {"PTS": 14.5, "REB": 3.5, "AST": 7.7, "POS": "G", "TEAM": "Indiana Pacers"},
        "Pascal Siakam": {"PTS": 20.6, "REB": 7.2, "AST": 3.8, "POS": "F", "TEAM": "Indiana Pacers"},
        "Moussa Diabate": {"PTS": 7.4, "REB": 10.6, "AST": 1.0, "POS": "C", "TEAM": "Charlotte Hornets"}
    }
    return {normalize_name(k): v for k, v in raw_baselines.items()}

def fetch_espn_season_averages():
    """Pulls official full-season player averages (PTS, REB, AST, 3PM, MIN) directly from ESPN."""
    season_map = {}
    base_url = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete"
    for page in range(1, 9):
        params = {
            "region": "us",
            "lang": "en",
            "contentorigin": "espn",
            "isqualified": "false",
            "page": page,
            "limit": 100
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            if res.status_code != 200:
                break
            data = res.json()
            athletes = data.get("athletes", [])
            if not athletes:
                break

            top_cat_names = {}
            for c in data.get("categories", []):
                if c.get("name") and c.get("names"):
                    top_cat_names[c["name"]] = c["names"]

            for item in athletes:
                ath = item.get("athlete", {})
                p_disp = ath.get("displayName", "")
                if not p_disp:
                    continue
                norm_p = normalize_name(p_disp)
                t_raw = (
                    ath.get("team", {}).get("displayName")
                    or ath.get("teamName")
                    or ath.get("teamShortName")
                    or ""
                )
                t_name = normalize_team_name(t_raw) if t_raw else ""
                pos_abbr = ath.get("position", {}).get("abbreviation", "G")
                p_pos = "C" if "C" in pos_abbr else ("F" if "F" in pos_abbr else "G")

                stat_dict = {}
                for cat_blk in item.get("categories", []):
                    c_name = cat_blk.get("name", "")
                    names_list = cat_blk.get("names") or top_cat_names.get(c_name, [])
                    vals_list = cat_blk.get("values") or cat_blk.get("totals") or []
                    for k_name, v_val in zip(names_list, vals_list):
                        try:
                            stat_dict[k_name] = float(v_val)
                        except:
                            continue

                pts = stat_dict.get("avgPoints")
                reb = stat_dict.get("avgRebounds")
                ast = stat_dict.get("avgAssists")
                threes = stat_dict.get("avgThreePointFieldGoalsMade", 1.2)
                mins = stat_dict.get("avgMinutes", 20.0)

                if pts is not None and reb is not None and ast is not None and mins >= 8.0:
                    season_map[norm_p] = {
                        "PTS": round(float(pts), 2),
                        "REB": round(float(reb), 2),
                        "AST": round(float(ast), 2),
                        "3PM": round(float(threes), 2),
                        "PRA": round(float(pts) + float(reb) + float(ast), 2),
                        "MIN": round(float(mins), 1),
                        "POS": p_pos,
                        "TEAM": t_name
                    }

            if page >= data.get("pagination", {}).get("pages", 1):
                break
        except:
            break
    return season_map

def sync_live_nba_data():
    team_db = get_team_efficiency_baselines()
    player_db = get_fallback_player_baselines()

    central_tz = pytz.timezone("America/Chicago")
    now_ct = datetime.now(timezone.utc).astimezone(central_tz)
    today_str = now_ct.strftime("%Y-%m-%d")

    cached = load_json(LIVE_STATS_CACHE_FILE, {})
    if (
        cached.get("date") == today_str
        and cached.get("schema_version") == CACHE_SCHEMA_VERSION
        and cached.get("player_db")
        and cached.get("team_db")
        and "pos_def_mults" in cached
        and "fatigue_map" in cached
    ):
        return (
            cached["team_db"],
            cached["player_db"],
            cached["opp_ranks"],
            cached["pos_ranks"],
            cached["pos_def_mults"],
            cached["fatigue_map"]
        )

    print("🔄 Syncing V16 NBA Engine: Live Pace, Home/Away & L10 Splits, 40/60 FG Form, & 10-Book 1Q...")

    # 1. Layer Official ESPN Season Averages on top of Fallback Dictionary
    espn_season_stats = fetch_espn_season_averages()
    for norm_p, s_data in espn_season_stats.items():
        prev_team = player_db.get(norm_p, {}).get("TEAM", "")
        resolved_team = s_data["TEAM"] if s_data["TEAM"] in TEAM_ABBR else prev_team
        player_db[norm_p] = {
            "PTS": s_data["PTS"],
            "REB": s_data["REB"],
            "AST": s_data["AST"],
            "3PM": s_data["3PM"],
            "PRA": s_data["PRA"],
            "SZN_PTS": s_data["PTS"],
            "SZN_REB": s_data["REB"],
            "SZN_AST": s_data["AST"],
            "SZN_3PM": s_data["3PM"],
            "SZN_PRA": s_data["PRA"],
            "L10_PTS": s_data["PTS"],
            "L10_REB": s_data["REB"],
            "L10_AST": s_data["AST"],
            "L10_3PM": s_data["3PM"],
            "L10_PRA": s_data["PRA"],
            "L5_PTS": s_data["PTS"],
            "L5_REB": s_data["REB"],
            "L5_AST": s_data["AST"],
            "L5_3PM": s_data["3PM"],
            "L5_PRA": s_data["PRA"],
            "SZN_MIN": s_data["MIN"],
            "L5_MIN": s_data["MIN"],
            "MEDIAN_L5_MIN": s_data["MIN"],
            "RECENT_GAMES": 0,
            "LAST_10D_GAMES": 0,
            "IS_STARTER": False,
            "Q1_PTS_L10": [],
            "Q1_REB_L10": [],
            "Q1_AST_L10": [],
            "POS": s_data["POS"],
            "TEAM": resolved_team
        }

    for norm_p, p_data in player_db.items():
        p_data.setdefault("3PM", round(p_data["PTS"] * 0.09, 2))
        p_data.setdefault("PRA", round(p_data["PTS"] + p_data["REB"] + p_data["AST"], 2))
        p_data.setdefault("SZN_MIN", 24.0)
        p_data.setdefault("L5_MIN", 24.0)
        p_data.setdefault("MEDIAN_L5_MIN", 24.0)
        p_data.setdefault("RECENT_GAMES", 0)
        p_data.setdefault("LAST_10D_GAMES", 0)
        p_data.setdefault("IS_STARTER", False)
        for k in ["PTS", "REB", "AST", "3PM", "PRA"]:
            p_data.setdefault(f"SZN_{k}", p_data[k])
            p_data.setdefault(f"L10_{k}", p_data[k])
            p_data.setdefault(f"L5_{k}", p_data[k])
        for k in ["PTS", "REB", "AST"]:
            p_data.setdefault(f"Q1_{k}_L10", [])

    try:
        st_url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
        st_res = requests.get(st_url, timeout=10).json()
        for child in st_res.get("children", []):
            for entry in child.get("standings", {}).get("entries", []):
                t_name = normalize_team_name(entry.get("team", {}).get("displayName", ""))
                if t_name in team_db:
                    for s in entry.get("stats", []):
                        if s.get("name") == "avgPointsFor" and s.get("value"):
                            team_db[t_name]["off"] = float(s["value"])
                        elif s.get("name") == "avgPointsAgainst" and s.get("value"):
                            team_db[t_name]["def"] = float(s["value"])
    except:
        pass

    player_logs = {}
    team_game_logs = {
        t: {"off": [], "def": [], "home_off": [], "home_def": [], "away_off": [], "away_def": [], "pace": []}
        for t in team_db
    }
    opp_allowed_logs = {t: {"PTS": [], "REB": [], "AST": []} for t in team_db}
    opp_pos_allowed_logs = {
        t: {pos: {"PTS": [], "REB": [], "AST": []} for pos in ["G", "F", "C"]}
        for t in team_db
    }
    period_logs = {t: {"q1_off": [], "q1_def": [], "h1_off": [], "h1_def": []} for t in team_db}
    recent_days_played = {t: [] for t in team_db}
    played_ot_yesterday = set()

    for days_ago in range(LOOKBACK_DAYS_FOR_STATS, 0, -1):
        d_str = (now_ct - timedelta(days=days_ago)).strftime("%Y%m%d")
        sb_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}"
        try:
            sb_data = requests.get(sb_url, timeout=8).json()
            for ev in sb_data.get("events", []):
                if not ev.get("status", {}).get("type", {}).get("completed"):
                    continue
                ev_id = ev["id"]
                sum_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={ev_id}"
                sum_data = requests.get(sum_url, timeout=8).json()

                q1_game_map = extract_q1_player_stats_from_summary(sum_data)

                comps = sum_data.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
                if len(comps) == 2:
                    home_c = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
                    away_c = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
                    h_t = normalize_team_name(home_c.get("team", {}).get("displayName", ""))
                    a_t = normalize_team_name(away_c.get("team", {}).get("displayName", ""))

                    if days_ago <= 3:
                        if h_t in recent_days_played:
                            recent_days_played[h_t].append(days_ago)
                        if a_t in recent_days_played:
                            recent_days_played[a_t].append(days_ago)

                    h_lines = [float(x.get("displayValue") or x.get("value") or 0) for x in home_c.get("linescores", [])]
                    a_lines = [float(x.get("displayValue") or x.get("value") or 0) for x in away_c.get("linescores", [])]
                    num_periods = max(len(h_lines), len(a_lines), 4)
                    reg_scale = 48.0 / (48.0 + 5.0 * max(0, num_periods - 4))

                    if days_ago == 1 and num_periods > 4:
                        played_ot_yesterday.add(h_t)
                        played_ot_yesterday.add(a_t)

                    # Live 48-min Normalized Home/Away & L10 Scoring + Live Possessions Pace
                    h_score_raw = float(home_c.get("score") or sum(h_lines) or 0)
                    a_score_raw = float(away_c.get("score") or sum(a_lines) or 0)
                    if h_score_raw > 50 and a_score_raw > 50 and h_t in team_game_logs and a_t in team_game_logs:
                        h_pts_48 = h_score_raw * reg_scale
                        a_pts_48 = a_score_raw * reg_scale
                        team_game_logs[h_t]["off"].append(h_pts_48)
                        team_game_logs[h_t]["def"].append(a_pts_48)
                        team_game_logs[h_t]["home_off"].append(h_pts_48)
                        team_game_logs[h_t]["home_def"].append(a_pts_48)

                        team_game_logs[a_t]["off"].append(a_pts_48)
                        team_game_logs[a_t]["def"].append(h_pts_48)
                        team_game_logs[a_t]["away_off"].append(a_pts_48)
                        team_game_logs[a_t]["away_def"].append(h_pts_48)

                        game_pace_48 = extract_game_pace_48(sum_data, reg_scale)
                        if game_pace_48 is not None:
                            team_game_logs[h_t]["pace"].append(game_pace_48)
                            team_game_logs[a_t]["pace"].append(game_pace_48)

                    if len(h_lines) >= 1 and len(a_lines) >= 1 and h_lines[0] > 0 and a_lines[0] > 0 and h_t in period_logs and a_t in period_logs:
                        period_logs[h_t]["q1_off"].append(h_lines[0])
                        period_logs[h_t]["q1_def"].append(a_lines[0])
                        period_logs[a_t]["q1_off"].append(a_lines[0])
                        period_logs[a_t]["q1_def"].append(h_lines[0])

                    if len(h_lines) >= 2 and len(a_lines) >= 2 and (h_lines[0] + h_lines[1]) > 0 and (a_lines[0] + a_lines[1]) > 0 and h_t in period_logs and a_t in period_logs:
                        h1_h = h_lines[0] + h_lines[1]
                        h1_a = a_lines[0] + a_lines[1]
                        period_logs[h_t]["h1_off"].append(h1_h)
                        period_logs[h_t]["h1_def"].append(h1_a)
                        period_logs[a_t]["h1_off"].append(h1_a)
                        period_logs[a_t]["h1_def"].append(h1_h)

                p_boxes = sum_data.get("boxscore", {}).get("players", [])
                if len(p_boxes) == 2:
                    team_totals = {}
                    team_pos_totals = {}
                    for tb in p_boxes:
                        t_name = normalize_team_name(tb.get("team", {}).get("displayName", ""))
                        stat_blk = tb.get("statistics", [{}])[0]
                        labels = stat_blk.get("labels", [])
                        min_idx = labels.index("MIN") if "MIN" in labels else -1
                        pts_idx = labels.index("PTS") if "PTS" in labels else -1
                        reb_idx = labels.index("REB") if "REB" in labels else -1
                        ast_idx = labels.index("AST") if "AST" in labels else -1
                        tpm_idx = labels.index("3PT") if "3PT" in labels else -1

                        t_pts, t_reb, t_ast = 0.0, 0.0, 0.0
                        pos_buckets = {pos: {"PTS": 0.0, "REB": 0.0, "AST": 0.0} for pos in ["G", "F", "C"]}
                        active_slot = 0

                        for ath in stat_blk.get("athletes", []):
                            if ath.get("didNotPlay"):
                                continue
                            ath_obj = ath.get("athlete", {})
                            p_disp = ath_obj.get("displayName", "")
                            pos_abbr = ath_obj.get("position", {}).get("abbreviation", "G")
                            p_pos_clean = "C" if "C" in pos_abbr else ("F" if "F" in pos_abbr else "G")
                            s_vals = ath.get("stats", [])
                            if not p_disp or not s_vals or pts_idx == -1:
                                continue
                            try:
                                mins = parse_minutes(s_vals[min_idx]) if min_idx != -1 else 0.0
                                if mins <= 0.0:
                                    continue
                                active_slot += 1
                                is_game_starter = bool(ath.get("starter")) if "starter" in ath else (active_slot <= 5)

                                pts = float(s_vals[pts_idx]) if pts_idx != -1 else 0.0
                                reb = float(s_vals[reb_idx]) if reb_idx != -1 else 0.0
                                ast = float(s_vals[ast_idx]) if ast_idx != -1 else 0.0
                                tpm_raw = str(s_vals[tpm_idx]).split("-")[0] if tpm_idx != -1 else "0"
                                tpm = float(tpm_raw) if tpm_raw.isdigit() else 0.0

                                t_pts += pts
                                t_reb += reb
                                t_ast += ast

                                norm_p = normalize_name(p_disp)
                                mapped_pos = player_db.get(norm_p, {}).get("POS", p_pos_clean)
                                pos_buckets[mapped_pos]["PTS"] += pts
                                pos_buckets[mapped_pos]["REB"] += reb
                                pos_buckets[mapped_pos]["AST"] += ast

                                if norm_p not in player_logs:
                                    player_logs[norm_p] = {
                                        "PTS": [], "REB": [], "AST": [], "3PM": [], "PRA": [], "MIN": [],
                                        "STARTS": [], "LAST_10D_GAMES": 0,
                                        "Q1_PTS": [], "Q1_REB": [], "Q1_AST": [],
                                        "TEAM": t_name, "POS": mapped_pos
                                    }
                                player_logs[norm_p]["PTS"].append(pts)
                                player_logs[norm_p]["REB"].append(reb)
                                player_logs[norm_p]["AST"].append(ast)
                                player_logs[norm_p]["3PM"].append(tpm)
                                player_logs[norm_p]["PRA"].append(pts + reb + ast)
                                player_logs[norm_p]["MIN"].append(mins)
                                player_logs[norm_p]["STARTS"].append(1 if is_game_starter else 0)
                                if days_ago <= 10:
                                    player_logs[norm_p]["LAST_10D_GAMES"] += 1
                                player_logs[norm_p]["TEAM"] = t_name

                                if norm_p in q1_game_map and (is_game_starter or mins >= 18.0):
                                    player_logs[norm_p]["Q1_PTS"].append(q1_game_map[norm_p]["PTS"])
                                    player_logs[norm_p]["Q1_REB"].append(q1_game_map[norm_p]["REB"])
                                    player_logs[norm_p]["Q1_AST"].append(q1_game_map[norm_p]["AST"])
                            except:
                                continue
                        team_totals[t_name] = {"PTS": t_pts, "REB": t_reb, "AST": t_ast}
                        team_pos_totals[t_name] = pos_buckets

                    t_names = list(team_totals.keys())
                    if len(t_names) == 2:
                        t1, t2 = t_names[0], t_names[1]
                        if t1 in opp_allowed_logs and t2 in team_totals:
                            for k in ["PTS", "REB", "AST"]:
                                if team_totals[t2][k] > 0:
                                    opp_allowed_logs[t1][k].append(team_totals[t2][k])
                                for pos in ["G", "F", "C"]:
                                    if team_pos_totals[t2][pos][k] > 0:
                                        opp_pos_allowed_logs[t1][pos][k].append(team_pos_totals[t2][pos][k])
                        if t2 in opp_allowed_logs and t1 in team_totals:
                            for k in ["PTS", "REB", "AST"]:
                                if team_totals[t1][k] > 0:
                                    opp_allowed_logs[t2][k].append(team_totals[t1][k])
                                for pos in ["G", "F", "C"]:
                                    if team_pos_totals[t1][pos][k] > 0:
                                        opp_pos_allowed_logs[t2][pos][k].append(team_pos_totals[t1][pos][k])
        except:
            continue

    fatigue_map = {}
    for t_name, days_list in recent_days_played.items():
        unique_days = sorted(list(set(days_list)))
        is_b2b = (1 in unique_days)
        is_3in4 = (len(unique_days) >= 2)
        is_ot = (t_name in played_ot_yesterday)

        tags = []
        if is_b2b:
            tags.append("B2B")
        if is_3in4:
            tags.append("3in4")
        if is_ot:
            tags.append("OT")

        fatigue_map[t_name] = {
            "is_b2b": is_b2b,
            "is_3in4": is_3in4,
            "is_ot": is_ot,
            "tag": "/".join(tags) if tags else ""
        }

    # V16 Player Form Weighting: 40% Official Season Avg (SZN) + 60% Last 10 Games (L10) + L5 Trend
    for norm_p, logs in player_logs.items():
        games_count = len(logs["PTS"])
        if games_count >= 1:
            full_pts = float(np.mean(logs["PTS"]))
            full_reb = float(np.mean(logs["REB"]))
            full_ast = float(np.mean(logs["AST"]))
            full_3pm = float(np.mean(logs["3PM"]))
            full_pra = float(np.mean(logs["PRA"]))
            full_min = float(np.mean(logs["MIN"]))

            l10_pts = float(np.mean(logs["PTS"][-10:]))
            l10_reb = float(np.mean(logs["REB"][-10:]))
            l10_ast = float(np.mean(logs["AST"][-10:]))
            l10_3pm = float(np.mean(logs["3PM"][-10:]))
            l10_pra = float(np.mean(logs["PRA"][-10:]))

            l5_pts = float(np.mean(logs["PTS"][-5:]))
            l5_reb = float(np.mean(logs["REB"][-5:]))
            l5_ast = float(np.mean(logs["AST"][-5:]))
            l5_3pm = float(np.mean(logs["3PM"][-5:]))
            l5_pra = float(np.mean(logs["PRA"][-5:]))

            l5_min = float(np.mean(logs["MIN"][-5:]))
            median_l5_min = float(np.median(logs["MIN"][-5:]))

            if norm_p in espn_season_stats:
                szn_pts = espn_season_stats[norm_p]["PTS"]
                szn_reb = espn_season_stats[norm_p]["REB"]
                szn_ast = espn_season_stats[norm_p]["AST"]
                szn_3pm = espn_season_stats[norm_p]["3PM"]
                szn_pra = espn_season_stats[norm_p]["PRA"]
                szn_min = espn_season_stats[norm_p]["MIN"]
            elif norm_p in player_db and games_count < 6:
                szn_pts = 0.50 * full_pts + 0.50 * player_db[norm_p]["SZN_PTS"]
                szn_reb = 0.50 * full_reb + 0.50 * player_db[norm_p]["SZN_REB"]
                szn_ast = 0.50 * full_ast + 0.50 * player_db[norm_p]["SZN_AST"]
                szn_3pm = full_3pm
                szn_pra = szn_pts + szn_reb + szn_ast
                szn_min = full_min
            else:
                szn_pts, szn_reb, szn_ast, szn_3pm, szn_pra, szn_min = full_pts, full_reb, full_ast, full_3pm, full_pra, full_min

            starts_list = logs.get("STARTS", [])
            start_rate = float(np.mean(starts_list)) if starts_list else 0.0
            recent_starts = sum(starts_list[-5:])
            is_normal_starter = bool(
                games_count >= 3
                and szn_min >= MIN_STARTER_AVG_MINUTES
                and (start_rate >= 0.60 or recent_starts >= 3)
            )

            pos_val = player_db.get(norm_p, {}).get("POS", logs.get("POS", "G"))
            player_db[norm_p] = {
                "PTS": round(0.40 * szn_pts + 0.60 * l10_pts, 2),
                "REB": round(0.40 * szn_reb + 0.60 * l10_reb, 2),
                "AST": round(0.40 * szn_ast + 0.60 * l10_ast, 2),
                "3PM": round(0.40 * szn_3pm + 0.60 * l10_3pm, 2),
                "PRA": round(0.40 * szn_pra + 0.60 * l10_pra, 2),
                "SZN_PTS": round(szn_pts, 1),
                "SZN_REB": round(szn_reb, 1),
                "SZN_AST": round(szn_ast, 1),
                "SZN_3PM": round(szn_3pm, 1),
                "SZN_PRA": round(szn_pra, 1),
                "L10_PTS": round(l10_pts, 1),
                "L10_REB": round(l10_reb, 1),
                "L10_AST": round(l10_ast, 1),
                "L10_3PM": round(l10_3pm, 1),
                "L10_PRA": round(l10_pra, 1),
                "L5_PTS": round(l5_pts, 1),
                "L5_REB": round(l5_reb, 1),
                "L5_AST": round(l5_ast, 1),
                "L5_3PM": round(l5_3pm, 1),
                "L5_PRA": round(l5_pra, 1),
                "SZN_MIN": round(szn_min, 1),
                "L5_MIN": round(l5_min, 1),
                "MEDIAN_L5_MIN": round(median_l5_min, 1),
                "RECENT_GAMES": games_count,
                "LAST_10D_GAMES": logs["LAST_10D_GAMES"],
                "IS_STARTER": is_normal_starter,
                "Q1_PTS_L10": logs["Q1_PTS"][-10:],
                "Q1_REB_L10": logs["Q1_REB"][-10:],
                "Q1_AST_L10": logs["Q1_AST"][-10:],
                "POS": pos_val,
                "TEAM": logs["TEAM"]
            }

    # Update Team DB with Live L10, Home/Away Splits & Live Pace
    for t_name, logs in opp_allowed_logs.items():
        if len(logs["REB"]) >= 2:
            team_db[t_name]["opp_reb"] = round(0.65 * float(np.mean(logs["REB"])) + 0.35 * team_db[t_name]["opp_reb"], 1)
        if len(logs["AST"]) >= 2:
            team_db[t_name]["opp_ast"] = round(0.65 * float(np.mean(logs["AST"])) + 0.35 * team_db[t_name]["opp_ast"], 1)
        if len(logs["PTS"]) >= 2:
            team_db[t_name]["def"] = round(0.80 * float(np.mean(logs["PTS"])) + 0.20 * team_db[t_name]["def"], 1)

    for t_name, g_logs in team_game_logs.items():
        base_off = team_db[t_name]["off"]
        base_def = team_db[t_name]["def"]

        if len(g_logs["off"]) >= 3:
            team_db[t_name]["l10_off"] = round(float(np.mean(g_logs["off"][-10:])), 1)
            team_db[t_name]["l10_def"] = round(float(np.mean(g_logs["def"][-10:])), 1)
        else:
            team_db[t_name]["l10_off"] = base_off
            team_db[t_name]["l10_def"] = base_def

        # Sample-weighted Home/Away splits (prevents 1-game noise)
        n_home = len(g_logs["home_off"])
        if n_home >= 2:
            w_h = min(0.75, n_home / 6.0)
            team_db[t_name]["home_off"] = round(w_h * float(np.mean(g_logs["home_off"])) + (1.0 - w_h) * base_off, 1)
            team_db[t_name]["home_def"] = round(w_h * float(np.mean(g_logs["home_def"])) + (1.0 - w_h) * base_def, 1)
        else:
            team_db[t_name]["home_off"] = base_off
            team_db[t_name]["home_def"] = base_def

        n_away = len(g_logs["away_off"])
        if n_away >= 2:
            w_a = min(0.75, n_away / 6.0)
            team_db[t_name]["away_off"] = round(w_a * float(np.mean(g_logs["away_off"])) + (1.0 - w_a) * base_off, 1)
            team_db[t_name]["away_def"] = round(w_a * float(np.mean(g_logs["away_def"])) + (1.0 - w_a) * base_def, 1)
        else:
            team_db[t_name]["away_off"] = base_off
            team_db[t_name]["away_def"] = base_def

        if len(g_logs["pace"]) >= 3:
            live_pace = float(np.mean(g_logs["pace"][-10:]))
            team_db[t_name]["pace"] = round(0.80 * live_pace + 0.20 * team_db[t_name]["pace"], 1)

    for t_name, p_logs in period_logs.items():
        if len(p_logs["q1_off"]) >= 2:
            team_db[t_name]["q1_off"] = round(0.65 * float(np.mean(p_logs["q1_off"])) + 0.35 * team_db[t_name]["q1_off"], 1)
            team_db[t_name]["q1_def"] = round(0.65 * float(np.mean(p_logs["q1_def"])) + 0.35 * team_db[t_name]["q1_def"], 1)
        if len(p_logs["h1_off"]) >= 2:
            team_db[t_name]["h1_off"] = round(0.65 * float(np.mean(p_logs["h1_off"])) + 0.35 * team_db[t_name]["h1_off"], 1)
            team_db[t_name]["h1_def"] = round(0.65 * float(np.mean(p_logs["h1_def"])) + 0.35 * team_db[t_name]["h1_def"], 1)

    pts_sorted = sorted(team_db.keys(), key=lambda x: team_db[x]["def"])
    reb_sorted = sorted(team_db.keys(), key=lambda x: team_db[x]["opp_reb"])
    ast_sorted = sorted(team_db.keys(), key=lambda x: team_db[x]["opp_ast"])

    opp_ranks = {}
    for t_name in team_db:
        opp_ranks[t_name] = {
            "PTS": pts_sorted.index(t_name) + 1,
            "REB": reb_sorted.index(t_name) + 1,
            "AST": ast_sorted.index(t_name) + 1,
            "3PM": pts_sorted.index(t_name) + 1,
            "PRA": pts_sorted.index(t_name) + 1
        }

    pos_avgs = {pos: {} for pos in ["G", "F", "C"]}
    team_pos_vals = {t: {pos: {} for pos in ["G", "F", "C"]} for t in team_db}

    for pos in ["G", "F", "C"]:
        for stat in ["PTS", "REB", "AST"]:
            all_vals = []
            for t_name in team_db:
                vals = opp_pos_allowed_logs[t_name][pos][stat]
                if vals:
                    all_vals.extend(vals)
            league_pos_avg = float(np.mean(all_vals)) if all_vals else 1.0
            pos_avgs[pos][stat] = max(1.0, league_pos_avg)

            for t_name in team_db:
                vals = opp_pos_allowed_logs[t_name][pos][stat]
                raw_pos_mean = float(np.mean(vals)) if len(vals) >= 2 else league_pos_avg
                team_pos_vals[t_name][pos][stat] = 0.65 * raw_pos_mean + 0.35 * league_pos_avg

    pos_ranks = {t: {pos: {} for pos in ["G", "F", "C"]} for t in team_db}
    pos_def_mults = {t: {pos: {} for pos in ["G", "F", "C"]} for t in team_db}

    for pos in ["G", "F", "C"]:
        for stat in ["PTS", "REB", "AST"]:
            sorted_teams = sorted(team_db.keys(), key=lambda x: team_pos_vals[x][pos][stat])
            for t_name in team_db:
                pos_ranks[t_name][pos][stat] = sorted_teams.index(t_name) + 1
                pos_def_mults[t_name][pos][stat] = round(team_pos_vals[t_name][pos][stat] / pos_avgs[pos][stat], 3)

    save_json(LIVE_STATS_CACHE_FILE, {
        "date": today_str,
        "schema_version": CACHE_SCHEMA_VERSION,
        "team_db": team_db,
        "player_db": player_db,
        "opp_ranks": opp_ranks,
        "pos_ranks": pos_ranks,
        "pos_def_mults": pos_def_mults,
        "fatigue_map": fatigue_map
    })

    # Startup Audits: Live Pace & 1Q Play-by-Play Verification
    pace_sorted = sorted(team_db.items(), key=lambda x: x[1]["pace"], reverse=True)
    fast_str = ", ".join(f"{get_abbr(t)} ({d['pace']:.1f})" for t, d in pace_sorted[:3])
    slow_str = ", ".join(f"{get_abbr(t)} ({d['pace']:.1f})" for t, d in pace_sorted[-3:])
    starters_with_q1 = [(k, v) for k, v in player_db.items() if v.get("IS_STARTER") and len(v.get("Q1_REB_L10", [])) >= 7]
    top_q1_reb = sorted(starters_with_q1, key=lambda x: float(np.mean(x[1]["Q1_REB_L10"])), reverse=True)[:3]
    top_q1_ast = sorted(starters_with_q1, key=lambda x: float(np.mean(x[1]["Q1_AST_L10"])), reverse=True)[:3]
    reb_audit = ", ".join(f"{k.title()} ({np.mean(v['Q1_REB_L10']):.1f})" for k, v in top_q1_reb)
    ast_audit = ", ".join(f"{k.title()} ({np.mean(v['Q1_AST_L10']):.1f})" for k, v in top_q1_ast)
    print(f"✅ Synced {len(player_db)} players ({len(starters_with_q1)} starters with 7+ 1Q logs)!")
    print(f"⏱️ Live Pace Audit — Fastest: [{fast_str}] | Slowest: [{slow_str}]")
    print(f"🔍 1Q PBP Audit — Top Q1 REB: [{reb_audit}] | Top Q1 AST: [{ast_audit}]")
    return team_db, player_db, opp_ranks, pos_ranks, pos_def_mults, fatigue_map

def fetch_injured_players():
    """Returns (prop_excluded, confirmed_out, questionable_gtd) sets."""
    prop_excluded = set()
    confirmed_out = set()
    questionable_gtd = set()

    def classify_injury(p_name, status_str):
        if not p_name:
            return
        norm_p = normalize_name(p_name)
        s = status_str.lower()
        if any(term in s for term in ["out", "doubtful", "injured", "suspension"]):
            prop_excluded.add(norm_p)
            confirmed_out.add(norm_p)
        elif any(term in s for term in ["questionable", "day-to-day", "gtd", "game time decision"]):
            prop_excluded.add(norm_p)
            questionable_gtd.add(norm_p)

    try:
        inj_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        res = requests.get(inj_url, timeout=10).json()
        for team_entry in res.get("injuries", []):
            for inj in team_entry.get("injuries", []):
                classify_injury(inj.get("athlete", {}).get("displayName"), inj.get("status", ""))
    except:
        pass

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        res = requests.get(url, timeout=10).json()
        for event in res.get("events", []):
            for team in event["competitions"][0]["competitors"]:
                for inj in team.get("injuries", []):
                    classify_injury(inj.get("athlete", {}).get("displayName"), inj.get("status", ""))
    except:
        pass
    return prop_excluded, confirmed_out, questionable_gtd

def calc_team_injury_deduction(team_name, player_db, confirmed_out, questionable_gtd):
    """
    Deducts projected team points for Out (100% weight) & Questionable/GTD (55% weight) scorers.
    Requires LAST_10D_GAMES >= 1 so multi-week absences already baked into team ratings aren't double-counted.
    """
    penalty = 0.0
    gtd_star_count = 0
    for norm_p, p_data in player_db.items():
        if p_data.get("TEAM") != team_name or p_data.get("LAST_10D_GAMES", 0) < 1:
            continue
        pts = p_data.get("SZN_PTS", p_data.get("PTS", 0.0))
        if pts < 11.0:
            continue
        if norm_p in confirmed_out:
            penalty += (pts - 8.0) * 0.20
        elif norm_p in questionable_gtd:
            penalty += (pts - 8.0) * 0.11
            if pts >= 13.5:
                gtd_star_count += 1
    return min(penalty, 14.5), gtd_star_count

def calc_team_fatigue_penalty(team_name, is_home, fatigue_map):
    """Calculates graduated point penalty for B2B, 3-in-4, and OT hangover."""
    f_info = fatigue_map.get(team_name, {})
    pen = 0.0
    if f_info.get("is_b2b"):
        pen += 1.9 if is_home else 2.5
    if f_info.get("is_3in4"):
        pen += 0.9 if f_info.get("is_b2b") else (1.4 if is_home else 1.8)
    if f_info.get("is_ot"):
        pen += 0.8
    return min(pen, 4.2)

def calc_teammate_usage_boost(norm_p, team_name, player_pos, stat_code, player_db, confirmed_out, questionable_gtd):
    """
    Fresh-Injury Usage Boost:
    - Only counts teammates who played in the LAST 10 DAYS (LAST_10D_GAMES >= 1) so
      multi-week absences already baked into L10 logs are never double-counted.
    - Caps positive usage boost at +6% (1.06x).
    """
    if not team_name:
        return 1.0, 0.0, 0.0
    missing_pts = 0.0
    missing_ast = 0.0
    missing_reb = 0.0
    missing_team_mins = 0.0
    missing_pos_mins = 0.0

    for other_p, p_data in player_db.items():
        if other_p == norm_p or p_data.get("TEAM") != team_name or p_data.get("LAST_10D_GAMES", 0) < 1:
            continue
        weight = 1.0 if other_p in confirmed_out else (0.55 if other_p in questionable_gtd else 0.0)
        if weight <= 0.0:
            continue
        pts = p_data.get("SZN_PTS", p_data.get("PTS", 0.0))
        ast = p_data.get("SZN_AST", p_data.get("AST", 0.0))
        reb = p_data.get("SZN_REB", p_data.get("REB", 0.0))
        mins = p_data.get("SZN_MIN", 20.0)
        other_pos = p_data.get("POS", "G")

        if mins >= 16.0:
            missing_team_mins += mins * weight
            if other_pos == player_pos:
                missing_pos_mins += mins * weight
        if pts >= 14.0:
            missing_pts += pts * weight
        if ast >= 4.8:
            missing_ast += ast * weight
        if reb >= 6.0:
            missing_reb += reb * weight

    total_missing_impact = missing_pts + missing_ast + missing_reb
    pos_min_boost = min(0.03, missing_pos_mins * 0.0010)

    if stat_code in ["PTS", "3PM"]:
        mult = 1.0 + (missing_pts * 0.0016) + pos_min_boost
    elif stat_code == "AST":
        mult = 1.0 + (missing_ast * 0.0050) + pos_min_boost
    elif stat_code == "REB":
        mult = 1.0 + (missing_reb * 0.0040) + pos_min_boost
    else:
        mult = 1.0 + (total_missing_impact * 0.0013) + pos_min_boost

    return min(1.06, mult), total_missing_impact, missing_team_mins

def is_spot_starter_under_trap(pick_dir, stat_code, line, szn_avg, l10_avg, szn_min, usage_boost_mult, missing_team_impact, missing_team_mins):
    """
    Blocks Under bets when lines or usage spike due to fresh teammate absences:
    - Applies to ALL players (including 30+ MPG stars) when 30+ star impact or 35+ rotation mins are freshly Out/GTD.
    - Blocks inflated lines (+4.5 PRA, +3.5 PTS, +1.6 REB/AST, or +28% jump).
    """
    if pick_dir != "Under":
        return False
    anchor = max(szn_avg, l10_avg, 0.5)
    if szn_min < MIN_FG_PROP_AVG_MINUTES:
        return True
    if missing_team_impact >= 30.0 or missing_team_mins >= 35.0:
        return True
    if szn_min < 29.0 and (missing_team_mins >= 18.0 or usage_boost_mult >= 1.02 or missing_team_impact >= 20.0):
        return True
    if line >= (anchor * 1.28):
        return True
    if stat_code == "PRA":
        if (line - anchor) >= 4.5:
            return True
        if (missing_team_impact >= 15.0 or missing_team_mins >= 18.0) and line >= (anchor * 1.18):
            return True
    if stat_code == "PTS" and (line - anchor) >= 3.5:
        return True
    if stat_code in ["REB", "AST"] and (line - anchor) >= 1.6:
        return True
    return False

def get_matchup_dot(opp_rank, pick_dir):
    if pick_dir == "Over":
        return "🟢" if opp_rank >= 21 else ("🔴" if opp_rank <= 10 else "🟡")
    else:
        return "🟢" if opp_rank <= 10 else ("🔴" if opp_rank >= 21 else "🟡")

def post_alert_to_discord(base_text, prefix="⏳ "):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ Error: DISCORD_WEBHOOK_URL secret is not set in GitHub Settings!")
        return None
    res = requests.post(f"{DISCORD_WEBHOOK_URL}?wait=true", json={"content": f"{prefix}{base_text}"})
    if res.status_code in [200, 204]:
        return res.json()["id"]
    return None

def scan_all_fanduel_markets():
    tracked_bets = load_json(TRACKER_FILE, [])
    already_posted = {b["bet_key"] for b in tracked_bets}
    prop_excluded, confirmed_out, questionable_gtd = fetch_injured_players()

    stats_db, player_db, opp_ranks, pos_ranks, pos_def_mults, fatigue_map = sync_live_nba_data()
    avg_off = float(np.mean([v["off"] for v in stats_db.values()]))
    avg_pace = float(np.mean([v["pace"] for v in stats_db.values()]))
    avg_q1_off = float(np.mean([v["q1_off"] for v in stats_db.values()]))
    avg_h1_off = float(np.mean([v["h1_off"] for v in stats_db.values()]))
    avg_opp_reb = float(np.mean([v["opp_reb"] for v in stats_db.values()]))
    avg_opp_ast = float(np.mean([v["opp_ast"] for v in stats_db.values()]))

    central_tz = pytz.timezone("America/Chicago")
    now_utc = datetime.now(timezone.utc)
    today_ct_str = now_utc.astimezone(central_tz).strftime("%Y-%m-%d")

    # Cross-Market Exposure Trackers for Today (Persisted Across 45-Min Rescans)
    posted_players_today = set()   # (date_str, norm_p) across BOTH FG & 1Q props
    posted_fg_teams_today = set()  # (date_str, team_name) max 1 FG prop per team
    posted_fg_props_today_count = 0

    for b in tracked_bets:
        if not b.get("bet_key", "").startswith(today_ct_str):
            continue
        mkt = b.get("market", "")
        if mkt in ["PLAYER_PROP", "PLAYER_PROP_Q1"] and b.get("player"):
            norm_b_p = normalize_name(b["player"])
            posted_players_today.add((today_ct_str, norm_b_p))
            if mkt == "PLAYER_PROP":
                posted_fg_props_today_count += 1
                p_team = b.get("player_team") or normalize_team_name(player_db.get(norm_b_p, {}).get("TEAM", ""))
                if p_team:
                    posted_fg_teams_today.add((today_ct_str, p_team))

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "markets": "totals,spreads",
        "bookmakers": BOOKMAKERS_LIST,
        "oddsFormat": "american"
    }

    res = requests.get(url, params=params, timeout=10)
    print(f"📊 V16 Multi-Book Sync for {today_ct_str}! (Credits Remaining: {res.headers.get('x-requests-remaining')})")
    if res.status_code != 200:
        print(f"⚠️ Odds API Error ({res.status_code}): {res.text[:200]}")
        return

    events = res.json()
    deep_scans_done = 0

    # Collects FG prop candidates across the entire slate for per-player, per-team, and top-5 daily caps
    slate_fg_prop_candidates = []

    # Diagnostic counters for 1Q props across all 10 books
    q1_offers_by_book = {"PTS": {}, "REB": {}, "AST": {}}
    q1_filter_stats = {
        k: {"raw_players": 0, "not_starter": 0, "low_logs": 0, "lt_7_hits": 0, "heavy_juice": 0, "red_dot": 0, "cons_or_inj": 0, "posted": 0}
        for k in ["PTS", "REB", "AST"]
    }

    for g in events:
        fd_markets, other_books = extract_fanduel_and_consensus(g.get("bookmakers", []))
        if not fd_markets:
            continue
        dt_utc = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        if dt_utc <= now_utc:
            continue

        date_str = dt_utc.astimezone(central_tz).strftime("%Y-%m-%d")
        if date_str != today_ct_str:
            continue

        event_id = g["id"]
        home = normalize_team_name(g["home_team"])
        away = normalize_team_name(g["away_team"])
        h_ab = get_abbr(home)
        a_ab = get_abbr(away)
        time_short = dt_utc.astimezone(central_tz).strftime("%-I:%M%p").lower()
        match_key = f"{date_str}_{away}_vs_{home}"

        # Track whether any total (FG, 1H, or 1Q) has already been posted for this game
        has_posted_game_total = any(
            f"{match_key}_{sfx}" in already_posted
            for sfx in ["FG_TOTAL", "totals_h1", "totals_q1"]
        )
        game_total_candidates = []

        default_team_row = {
            "off": avg_off, "def": avg_off, "l10_off": avg_off, "l10_def": avg_off,
            "home_off": avg_off, "home_def": avg_off, "away_off": avg_off, "away_def": avg_off,
            "pace": avg_pace, "opp_reb": avg_opp_reb, "opp_ast": avg_opp_ast,
            "q1_off": avg_q1_off, "q1_def": avg_q1_off, "h1_off": avg_h1_off, "h1_def": avg_h1_off
        }
        h_stats = stats_db.get(home, default_team_row)
        a_stats = stats_db.get(away, default_team_row)
        h_ranks = opp_ranks.get(home, {"PTS": 15, "REB": 15, "AST": 15, "3PM": 15, "PRA": 15})
        a_ranks = opp_ranks.get(away, {"PTS": 15, "REB": 15, "AST": 15, "3PM": 15, "PRA": 15})

        # 1. Live Pace Normalization & Blended Season (40%) + L10 (35%) + Home/Away Splits (25%)
        exp_pace = (h_stats["pace"] * a_stats["pace"]) / avg_pace
        team_avg_pace = (h_stats["pace"] + a_stats["pace"]) / 2.0
        pace_scaler = exp_pace / team_avg_pace

        h_eff_off = 0.40 * h_stats["off"] + 0.35 * h_stats.get("l10_off", h_stats["off"]) + 0.25 * h_stats.get("home_off", h_stats["off"])
        h_eff_def = 0.40 * h_stats["def"] + 0.35 * h_stats.get("l10_def", h_stats["def"]) + 0.25 * h_stats.get("home_def", h_stats["def"])
        a_eff_off = 0.40 * a_stats["off"] + 0.35 * a_stats.get("l10_off", a_stats["off"]) + 0.25 * a_stats.get("away_off", a_stats["off"])
        a_eff_def = 0.40 * a_stats["def"] + 0.35 * a_stats.get("l10_def", a_stats["def"]) + 0.25 * a_stats.get("away_def", a_stats["def"])

        raw_h_pts = ((h_eff_off + a_eff_def) / 2.0) * pace_scaler
        raw_a_pts = ((a_eff_off + h_eff_def) / 2.0) * pace_scaler

        # 2. Fresh-Injury Deduction & Schedule Fatigue
        h_inj_pen, h_gtd_stars = calc_team_injury_deduction(home, player_db, confirmed_out, questionable_gtd)
        a_inj_pen, a_gtd_stars = calc_team_injury_deduction(away, player_db, confirmed_out, questionable_gtd)

        h_fat_pen = calc_team_fatigue_penalty(home, True, fatigue_map)
        a_fat_pen = calc_team_fatigue_penalty(away, False, fatigue_map)
        hca_val = get_team_hca(home)

        # 3. TRUE NET RATING DIFFERENTIAL & V16 LIGHT-REGRESSION (10%) FULL-GAME TOTAL
        true_net_rtg_diff = ((h_eff_off - h_eff_def) - (a_eff_off - a_eff_def)) * pace_scaler
        shrunk_eff_diff = true_net_rtg_diff * SPREAD_SHRINKAGE
        stat_model_margin = shrunk_eff_diff + hca_val - (h_inj_pen - a_inj_pen) - (h_fat_pen - a_fat_pen)

        raw_total = raw_h_pts + raw_a_pts
        league_avg_total = avg_off * 2.0
        model_total = (0.90 * raw_total + 0.10 * league_avg_total) - (h_inj_pen + a_inj_pen) * 0.65 - (h_fat_pen + a_fat_pen) * 0.50

        fg_spread_abs = abs(stat_model_margin)

        # --- A. FULL-GAME TOTALS (V16: 10% REG + 6.5-PT CAP + 60/40 PINNACLE BLEND) & SPREADS ---
        if SCAN_FULL_GAME:
            for m in fd_markets:
                if m["key"] == "totals":
                    bet_key = f"{match_key}_FG_TOTAL"
                    if not has_posted_game_total and bet_key not in already_posted and (h_gtd_stars + a_gtd_stars) < 3:
                        over_o = next((o for o in m["outcomes"] if o["name"] == "Over"), None)
                        under_o = next((o for o in m["outcomes"] if o["name"] == "Under"), None)
                        if over_o and under_o:
                            line = float(over_o["point"])
                            diff = model_total - line
                            pick_dir = "Over" if diff >= 0 else "Under"
                            juice = int(over_o["price"]) if diff >= 0 else int(under_o["price"])

                            mkt_cons, mkt_prob = get_devigged_total_or_prop_consensus(other_books, "totals", pick_dir, line)

                            # V16 6.5-Point Market Dislocation Cap (Blocks Blind Outlier Fades)
                            anchor_total = mkt_cons if mkt_cons is not None else line
                            if abs(model_total - anchor_total) > MAX_FG_TOTAL_DISLOCATION:
                                continue

                            if mkt_cons is not None:
                                if (pick_dir == "Over" and (mkt_cons - line) <= -1.5) or (pick_dir == "Under" and (mkt_cons - line) >= 1.5):
                                    continue
                            if mkt_prob is not None and mkt_prob < 0.49:
                                continue

                            stat_win_prob = norm.cdf(abs(diff) / NBA_STD_DEV_TOTAL)
                            win_prob = (0.60 * stat_win_prob + 0.40 * mkt_prob) if mkt_prob is not None else stat_win_prob
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_FG_TOTAL:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice, max_units=1.5)
                                mkt_str = f" *(Mkt: {mkt_cons:.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{a_ab} @ {h_ab} — {pick_dir} {line}** `({juice:+d})`\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Proj: **{model_total:.1f}**{mkt_str}\n"
                                    f"🛡️ Def: {a_ab} #{a_ranks['PTS']} vs {h_ab} #{h_ranks['PTS']}  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                payload = {
                                    "bet_key": bet_key,
                                    "away": away,
                                    "home": home,
                                    "market": "FG_TOTAL",
                                    "line": line,
                                    "pick_dir": pick_dir,
                                    "juice": juice,
                                    "units": units,
                                    "base_text": base_text,
                                    "status": "PENDING"
                                }
                                game_total_candidates.append((edge, base_text, payload))

                elif m["key"] == "spreads":
                    bet_key = f"{match_key}_FG_SPREAD"
                    home_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == home), None)
                    away_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == away), None)
                    if home_o and away_o:
                        h_line = float(home_o["point"])
                        fg_spread_abs = abs(h_line)
                        if bet_key not in already_posted:
                            mkt_home_cons, _ = get_devigged_spread_consensus(other_books, "spreads", home, h_line)
                            mkt_margin_anchor = -(mkt_home_cons if mkt_home_cons is not None else h_line)
                            model_margin = 0.75 * stat_model_margin + 0.25 * mkt_margin_anchor

                            sp_diff = model_margin - (-h_line)
                            if abs(sp_diff) > 6.0:
                                continue

                            pick_team = home if sp_diff >= 0 else away
                            opp_team = away if sp_diff >= 0 else home
                            pick_gtd_stars = h_gtd_stars if pick_team == home else a_gtd_stars

                            if pick_gtd_stars >= 2:
                                continue

                            pick_ab = get_abbr(pick_team)
                            opp_ab = get_abbr(opp_team)
                            pick_line = h_line if sp_diff >= 0 else float(away_o["point"])
                            juice = int(home_o["price"]) if sp_diff >= 0 else int(away_o["price"])

                            mkt_cons, mkt_prob = get_devigged_spread_consensus(other_books, "spreads", pick_team, pick_line)
                            if mkt_cons is not None and (pick_line - mkt_cons) <= -1.5:
                                continue
                            if mkt_prob is not None and mkt_prob < 0.46:
                                continue

                            stat_win_prob = norm.cdf(abs(sp_diff) / NBA_STD_DEV_SPREAD)
                            win_prob = (0.75 * stat_win_prob + 0.25 * mkt_prob) if mkt_prob is not None else stat_win_prob
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TEAM_MARKETS:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice, max_units=2.0)
                                opp_def_rk = opp_ranks.get(opp_team, {}).get("PTS", 15)
                                fair_line = -model_margin if pick_team == home else model_margin
                                opp_fat_tag = fatigue_map.get(opp_team, {}).get("tag", "")
                                fat_note = f" ⚠️{opp_fat_tag}" if opp_fat_tag else ""
                                mkt_str = f" *(Mkt: {mkt_cons:+.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{pick_ab} {pick_line:+.1f}** `({juice:+d})`  •  *{a_ab} @ {h_ab}*\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Fair: **{pick_ab} {fair_line:+.1f}**{mkt_str}\n"
                                    f"🛡️ Opp Def: {opp_ab} **#{opp_def_rk}**{fat_note}  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "FG_SPREAD", "pick_team": pick_team, "line": pick_line, "juice": juice, "units": units, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

        # --- B. DEEP SCAN: 1Q STARTER PROPS (UNCHANGED FROM V15), 1H/1Q TOTALS, & V16 FG PROPS ---
        if (SCAN_HALVES_QUARTERS or SCAN_PLAYER_PROPS or SCAN_Q1_PLAYER_PROPS) and (deep_scans_done < MAX_DEEP_SCAN_GAMES_PER_RUN):
            extra_markets = []
            if SCAN_HALVES_QUARTERS:
                extra_markets.extend(["totals_h1", "spreads_h1", "totals_q1", "spreads_q1"])
            if SCAN_PLAYER_PROPS:
                extra_markets.extend(["player_points", "player_rebounds", "player_assists"])
                if SCAN_PRA_AND_3PM:
                    extra_markets.extend(["player_points_rebounds_assists", "player_threes"])
            if SCAN_Q1_PLAYER_PROPS:
                extra_markets.extend(["player_points_q1", "player_rebounds_q1", "player_assists_q1"])

            ev_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            ev_params = {
                "apiKey": THE_ODDS_API_KEY,
                "markets": ",".join(extra_markets),
                "bookmakers": BOOKMAKERS_LIST,
                "oddsFormat": "american"
            }
            ev_res = requests.get(ev_url, params=ev_params, timeout=10)
            if ev_res.status_code == 200:
                deep_scans_done += 1
                ev_data = ev_res.json()
                all_ev_books = ev_data.get("bookmakers", [])
                fd_ev_markets, other_ev_books = extract_fanduel_and_consensus(all_ev_books)

                # Pace-Scaled & Regressed Period Team Projections (Totals)
                q1_h_pts = (((h_stats["q1_off"] + a_stats["q1_def"]) / 2.0) * pace_scaler) - (h_inj_pen * 0.22)
                q1_a_pts = (((a_stats["q1_off"] + h_stats["q1_def"]) / 2.0) * pace_scaler) - (a_inj_pen * 0.22)
                h1_h_pts = (((h_stats["h1_off"] + a_stats["h1_def"]) / 2.0) * pace_scaler) - (h_inj_pen * 0.46) - (h_fat_pen * 0.35)
                h1_a_pts = (((a_stats["h1_off"] + h_stats["h1_def"]) / 2.0) * pace_scaler) - (a_inj_pen * 0.46) - (a_fat_pen * 0.35)

                # True Period Net Rating Differentials
                true_q1_net_diff = (((h_stats["q1_off"] - h_stats["q1_def"]) - (a_stats["q1_off"] - a_stats["q1_def"])) * pace_scaler) - ((h_inj_pen - a_inj_pen) * 0.22)
                true_h1_net_diff = (((h_stats["h1_off"] - h_stats["h1_def"]) - (a_stats["h1_off"] - a_stats["h1_def"])) * pace_scaler) - ((h_inj_pen - a_inj_pen) * 0.46) - ((h_fat_pen - a_fat_pen) * 0.35)

                raw_q1_tot = q1_h_pts + q1_a_pts
                raw_h1_tot = h1_h_pts + h1_a_pts
                reg_q1_tot = PERIOD_TOTAL_SHRINKAGE * raw_q1_tot + (1.0 - PERIOD_TOTAL_SHRINKAGE) * (avg_q1_off * 2.0)
                reg_h1_tot = PERIOD_TOTAL_SHRINKAGE * raw_h1_tot + (1.0 - PERIOD_TOTAL_SHRINKAGE) * (avg_h1_off * 2.0)

                # --- B1. 1ST QUARTER STARTER PROPS (UNCHANGED FROM V15) ---
                if SCAN_Q1_PLAYER_PROPS:
                    q1_stat_map = {
                        "player_points_q1": "PTS",
                        "player_rebounds_q1": "REB",
                        "player_assists_q1": "AST"
                    }
                    q1_board = {}
                    for b in all_ev_books:
                        b_key = b.get("key", "")
                        if b_key not in PLAYABLE_US_BOOKS:
                            continue
                        for m in b.get("markets", []):
                            m_key = m.get("key", "")
                            if m_key not in q1_stat_map:
                                continue
                            stat_code = q1_stat_map[m_key]
                            for o in m.get("outcomes", []):
                                p_name = o.get("description", "")
                                side_name = o.get("name", "")
                                if not p_name or side_name not in ["Over", "Under"] or "point" not in o:
                                    continue
                                norm_p = normalize_name(p_name)
                                line_val = float(o["point"])
                                price_val = int(o["price"])
                                entry = q1_board.setdefault((m_key, norm_p), {"name": p_name, "offers": {}})
                                entry["offers"].setdefault((b_key, line_val), {})[side_name] = price_val
                                q1_offers_by_book[stat_code][b_key] = q1_offers_by_book[stat_code].get(b_key, 0) + 1

                    player_q1_candidates = {}
                    for (m_key, norm_p), entry in q1_board.items():
                        stat_code = q1_stat_map[m_key]
                        p_name = entry["name"]
                        q1_filter_stats[stat_code]["raw_players"] += 1

                        if norm_p in prop_excluded or norm_p not in player_db:
                            continue
                        if (today_ct_str, norm_p) in posted_players_today:
                            continue

                        p_info = player_db[norm_p]
                        if not p_info.get("IS_STARTER", False):
                            q1_filter_stats[stat_code]["not_starter"] += 1
                            continue

                        p_team = normalize_team_name(p_info.get("TEAM", ""))
                        if p_team not in [home, away]:
                            continue

                        bet_key = f"{match_key}_{norm_p}_Q1_{stat_code}"
                        if bet_key in already_posted:
                            continue

                        q1_logs = p_info.get(f"Q1_{stat_code}_L10", [])[-10:]
                        if len(q1_logs) < MIN_Q1_L10_HITS:
                            q1_filter_stats[stat_code]["low_logs"] += 1
                            continue

                        p_pos = p_info.get("POS", "G")
                        usage_boost_mult, missing_team_impact, missing_team_mins = calc_teammate_usage_boost(
                            norm_p, p_team, p_pos, stat_code, player_db, confirmed_out, questionable_gtd
                        )

                        if p_team == home:
                            opp_team_name, opp_r, opp_ab_str = away, a_ranks, a_ab
                        else:
                            opp_team_name, opp_r, opp_ab_str = home, h_ranks, h_ab

                        opp_def_rk = opp_r[stat_code]
                        emp_pos_rk = pos_ranks.get(opp_team_name, {}).get(p_pos, {}).get(stat_code, opp_def_rk)
                        blended_def_rk = int(round(0.50 * opp_def_rk + 0.50 * emp_pos_rk))

                        qualifying_q1 = []
                        saw_7_hits = False
                        saw_playable_juice = False
                        saw_non_red = False

                        for (b_key, line), sides in entry["offers"].items():
                            # Evaluate Over
                            if "Over" in sides:
                                over_hits = sum(1 for v in q1_logs if v > line)
                                juice = sides["Over"]
                                if over_hits >= MIN_Q1_L10_HITS:
                                    saw_7_hits = True
                                    if juice >= MAX_Q1_PROP_JUICE:
                                        saw_playable_juice = True
                                        dot = get_matchup_dot(blended_def_rk, "Over")
                                        if dot != "🔴":
                                            saw_non_red = True
                                            comp_books = [bk for bk in all_ev_books if bk.get("key") != b_key]
                                            mkt_cons, mkt_prob = get_devigged_total_or_prop_consensus(comp_books, m_key, "Over", line, p_name)
                                            if mkt_cons is None or (mkt_cons - line) > -1.0:
                                                is_fd = 1 if b_key == "fanduel" else 0
                                                qualifying_q1.append((is_fd, over_hits, juice, b_key, line, "Over", dot, mkt_prob))

                            # Evaluate Under
                            if "Under" in sides:
                                under_hits = sum(1 for v in q1_logs if v < line)
                                juice = sides["Under"]
                                if under_hits >= MIN_Q1_L10_HITS:
                                    saw_7_hits = True
                                    if juice >= MAX_Q1_PROP_JUICE:
                                        saw_playable_juice = True
                                        if not (usage_boost_mult >= 1.03 or missing_team_mins >= 22.0 or missing_team_impact >= 25.0):
                                            dot = get_matchup_dot(blended_def_rk, "Under")
                                            if dot != "🔴":
                                                saw_non_red = True
                                                comp_books = [bk for bk in all_ev_books if bk.get("key") != b_key]
                                                mkt_cons, mkt_prob = get_devigged_total_or_prop_consensus(comp_books, m_key, "Under", line, p_name)
                                                if mkt_cons is None or (mkt_cons - line) < 1.0:
                                                    is_fd = 1 if b_key == "fanduel" else 0
                                                    qualifying_q1.append((is_fd, under_hits, juice, b_key, line, "Under", dot, mkt_prob))

                        if not qualifying_q1:
                            if not saw_7_hits:
                                q1_filter_stats[stat_code]["lt_7_hits"] += 1
                            elif not saw_playable_juice:
                                q1_filter_stats[stat_code]["heavy_juice"] += 1
                            elif not saw_non_red:
                                q1_filter_stats[stat_code]["red_dot"] += 1
                            else:
                                q1_filter_stats[stat_code]["cons_or_inj"] += 1
                            continue

                        _, hits, juice, best_book, line, pick_dir, dot, mkt_prob = max(
                            qualifying_q1, key=lambda x: (x[0], x[1], x[2])
                        )

                        n_games = len(q1_logs)
                        hit_rate = hits / float(n_games)
                        q1_avg = float(np.mean(q1_logs))
                        est_win_prob = (0.70 * hit_rate + 0.30 * mkt_prob) if mkt_prob is not None else (0.82 * hit_rate)
                        units = calc_kelly_units(est_win_prob, juice, max_units=1.5)
                        usage_tag = " ⚡+Usage" if (usage_boost_mult >= 1.03 and pick_dir == "Over") else ""
                        book_tag = "" if best_book == "fanduel" else f" • {BOOK_LABELS.get(best_book, best_book.upper())}"

                        base_text = (
                            f"**{p_name} — 1Q {pick_dir} {line} {stat_code}** `({juice:+d}{book_tag})`\n"
                            f"🎯 **L{n_games} Hit Rate: {hits}/{n_games} ({hit_rate*100:.0f}%)** `({units:.1f}u)`  •  📊 1Q Avg: **{q1_avg:.1f}**\n"
                            f"{dot} vs {opp_ab_str} Def: **#{opp_def_rk}** *({p_pos}: #{emp_pos_rk})*{usage_tag}  •  🏀 {a_ab}@{h_ab} *({time_short})*\n"
                            f"{DIVIDER}"
                        )
                        payload = {
                            "bet_key": bet_key,
                            "away": away,
                            "home": home,
                            "market": "PLAYER_PROP_Q1",
                            "player": p_name,
                            "player_team": p_team,
                            "stat": stat_code,
                            "line": line,
                            "pick_dir": pick_dir,
                            "juice": juice,
                            "units": units,
                            "dot": dot,
                            "book": best_book,
                            "base_text": base_text,
                            "status": "PENDING"
                        }
                        player_q1_candidates.setdefault(norm_p, []).append((hit_rate, juice, stat_code, base_text, payload))

                    for norm_p, cand_list in player_q1_candidates.items():
                        _, _, best_stat, best_text, best_payload = max(cand_list, key=lambda x: (x[0], x[1]))
                        msg_id = post_alert_to_discord(best_text)
                        if msg_id:
                            best_payload["message_id"] = msg_id
                            tracked_bets.append(best_payload)
                            already_posted.add(best_payload["bet_key"])
                            posted_players_today.add((today_ct_str, norm_p))
                            q1_filter_stats[best_stat]["posted"] += 1

                # --- B2. PERIOD TOTALS / SPREADS & V16 FULL-GAME PROPS (NO 7/10 GATE) ---
                for m in fd_ev_markets:
                    m_key = m["key"]

                    if m_key in ["totals_h1", "totals_q1"]:
                        if has_posted_game_total or (h_gtd_stars + a_gtd_stars) >= 3:
                            continue
                        is_q1 = (m_key == "totals_q1")
                        period_label = "1Q" if is_q1 else "1H"
                        std_dev = NBA_STD_DEV_Q1_TOTAL if is_q1 else NBA_STD_DEV_H1_TOTAL
                        bet_key = f"{match_key}_{m_key}"

                        over_o = next((o for o in m["outcomes"] if o["name"] == "Over"), None)
                        under_o = next((o for o in m["outcomes"] if o["name"] == "Under"), None)
                        if over_o and under_o and bet_key not in already_posted:
                            line = float(over_o["point"])
                            proj_period_tot = reg_q1_tot if is_q1 else reg_h1_tot
                            diff = proj_period_tot - line
                            pick_dir = "Over" if diff >= 0 else "Under"
                            juice = int(over_o["price"]) if diff >= 0 else int(under_o["price"])

                            mkt_cons, mkt_prob = get_devigged_total_or_prop_consensus(other_ev_books, m_key, pick_dir, line)
                            if mkt_cons is not None:
                                if (pick_dir == "Over" and (mkt_cons - line) <= -1.5) or (pick_dir == "Under" and (mkt_cons - line) >= 1.5):
                                    continue
                            if mkt_prob is not None and mkt_prob < 0.46:
                                continue

                            stat_win_prob = norm.cdf(abs(diff) / std_dev)
                            win_prob = (0.75 * stat_win_prob + 0.25 * mkt_prob) if mkt_prob is not None else stat_win_prob
                            edge = win_prob - american_to_implied_prob(juice)

                            if edge >= MIN_EDGE_TEAM_MARKETS:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice, max_units=2.0)
                                mkt_str = f" *(Mkt: {mkt_cons:.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{a_ab} @ {h_ab} — {period_label} {pick_dir} {line}** `({juice:+d})`\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Proj: **{proj_period_tot:.1f}**{mkt_str}\n"
                                    f"⚡ Period Matchup Sync  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                payload = {
                                    "bet_key": bet_key,
                                    "away": away,
                                    "home": home,
                                    "market": m_key.upper(),
                                    "line": line,
                                    "pick_dir": pick_dir,
                                    "juice": juice,
                                    "units": units,
                                    "base_text": base_text,
                                    "status": "PENDING"
                                }
                                game_total_candidates.append((edge, base_text, payload))

                    elif m_key in ["spreads_h1", "spreads_q1"]:
                        is_q1 = (m_key == "spreads_q1")
                        period_label = "1Q" if is_q1 else "1H"
                        std_dev = NBA_STD_DEV_Q1_SPREAD if is_q1 else NBA_STD_DEV_H1_SPREAD
                        bet_key = f"{match_key}_{m_key}"

                        home_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == home), None)
                        away_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == away), None)
                        if home_o and away_o and bet_key not in already_posted:
                            h_line = float(home_o["point"])
                            raw_p_diff = true_q1_net_diff if is_q1 else true_h1_net_diff
                            hca_p = (hca_val * 0.25) if is_q1 else (hca_val * 0.50)
                            stat_p_margin = (raw_p_diff * SPREAD_SHRINKAGE) + hca_p

                            mkt_p_home, _ = get_devigged_spread_consensus(other_ev_books, m_key, home, h_line)
                            mkt_p_anchor = -(mkt_p_home if mkt_p_home is not None else h_line)
                            p_margin = 0.75 * stat_p_margin + 0.25 * mkt_p_anchor

                            sp_diff = p_margin - (-h_line)
                            max_p_dislocation = 2.5 if is_q1 else 3.8
                            if abs(sp_diff) > max_p_dislocation:
                                continue

                            pick_team = home if sp_diff >= 0 else away
                            pick_gtd_stars = h_gtd_stars if pick_team == home else a_gtd_stars
                            if pick_gtd_stars >= 2:
                                continue

                            pick_ab = get_abbr(pick_team)
                            pick_line = h_line if sp_diff >= 0 else float(away_o["point"])
                            juice = int(home_o["price"]) if sp_diff >= 0 else int(away_o["price"])

                            mkt_cons, mkt_prob = get_devigged_spread_consensus(other_ev_books, m_key, pick_team, pick_line)
                            if mkt_cons is not None and (pick_line - mkt_cons) <= -1.5:
                                continue
                            if mkt_prob is not None and mkt_prob < 0.46:
                                continue

                            stat_win_prob = norm.cdf(abs(sp_diff) / std_dev)
                            win_prob = (0.75 * stat_win_prob + 0.25 * mkt_prob) if mkt_prob is not None else stat_win_prob
                            edge = win_prob - american_to_implied_prob(juice)

                            if edge >= MIN_EDGE_TEAM_MARKETS:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice, max_units=2.0)
                                fair_p = -p_margin if pick_team == home else p_margin
                                mkt_str = f" *(Mkt: {mkt_cons:+.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{pick_ab} {period_label} {pick_line:+.1f}** `({juice:+d})`  •  *{a_ab} @ {h_ab}*\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Fair: **{pick_ab} {fair_p:+.1f}**{mkt_str}\n"
                                    f"⚡ {period_label} Matchup Sync  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": m_key.upper(), "pick_team": pick_team, "line": pick_line, "juice": juice, "units": units, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

                    elif m_key in ["player_points", "player_rebounds", "player_assists", "player_points_rebounds_assists", "player_threes"]:
                        stat_code_map = {
                            "player_points": "PTS",
                            "player_rebounds": "REB",
                            "player_assists": "AST",
                            "player_points_rebounds_assists": "PRA",
                            "player_threes": "3PM"
                        }
                        stat_code = stat_code_map[m_key]
                        players_in_market = {}
                        for o in m["outcomes"]:
                            p_name = o.get("description", "")
                            if p_name not in players_in_market:
                                players_in_market[p_name] = {}
                            players_in_market[p_name][o["name"]] = o

                        for p_name, sides in players_in_market.items():
                            norm_p = normalize_name(p_name)
                            if norm_p in prop_excluded or norm_p not in player_db or "Over" not in sides or "Under" not in sides:
                                continue

                            # Cross-Market 1-Prop-Per-Player Cap (FG + 1Q combined)
                            if (today_ct_str, norm_p) in posted_players_today:
                                continue

                            p_info = player_db[norm_p]
                            # Core Starter Gate: Verified normal starter averaging 24.0+ MPG
                            szn_min = p_info.get("SZN_MIN", 0.0)
                            if not p_info.get("IS_STARTER", False) or szn_min < MIN_FG_PROP_AVG_MINUTES:
                                continue

                            p_team = normalize_team_name(p_info.get("TEAM", ""))
                            if p_team not in [home, away]:
                                continue
                            # Max 1 FG Prop Per Team
                            if (today_ct_str, p_team) in posted_fg_teams_today:
                                continue

                            bet_key = f"{match_key}_{norm_p}_{stat_code}"
                            if bet_key in already_posted:
                                continue

                            line = float(sides["Over"]["point"])
                            if stat_code == "PTS" and line < MIN_PROP_LINE_PTS:
                                continue
                            elif stat_code == "3PM" and line < MIN_PROP_LINE_3PM:
                                continue
                            elif stat_code in ["REB", "AST"] and line < MIN_PROP_LINE_OTHER:
                                continue
                            elif stat_code == "PRA" and line < 14.5:
                                continue

                            base_avg = p_info.get(stat_code, 0.0)
                            if base_avg <= 0.0:
                                continue
                            szn_avg = p_info.get(f"SZN_{stat_code}", base_avg)
                            l10_avg = p_info.get(f"L10_{stat_code}", base_avg)
                            l5_avg = p_info.get(f"L5_{stat_code}", l10_avg)
                            median_l5_min = p_info.get("MEDIAN_L5_MIN", szn_min)
                            p_pos = p_info.get("POS", "G")

                            # --- 1. TRUE SINGLE-OPPONENT + EMPIRICAL DvP (G/F/C) MAPPING ---
                            dvp_stat_key = "PTS" if stat_code in ["PTS", "3PM", "PRA"] else stat_code
                            if p_team == home:
                                opp_team_name, opp_s, opp_r, opp_ab_str = away, a_stats, a_ranks, a_ab
                            else:
                                opp_team_name, opp_s, opp_r, opp_ab_str = home, h_stats, h_ranks, h_ab

                            if stat_code in ["PTS", "3PM", "PRA"]:
                                team_def_mult = opp_s["def"] / avg_off
                                opp_def_rk = opp_r["PTS"]
                            elif stat_code == "REB":
                                team_def_mult = opp_s["opp_reb"] / avg_opp_reb
                                opp_def_rk = opp_r["REB"]
                            else:
                                team_def_mult = opp_s["opp_ast"] / avg_opp_ast
                                opp_def_rk = opp_r["AST"]

                            emp_pos_mult = pos_def_mults.get(opp_team_name, {}).get(p_pos, {}).get(dvp_stat_key, team_def_mult)
                            emp_pos_rk = pos_ranks.get(opp_team_name, {}).get(p_pos, {}).get(dvp_stat_key, opp_def_rk)

                            matchup_mult = (0.50 * team_def_mult) + (0.50 * emp_pos_mult)
                            blended_def_rk = int(round(0.50 * opp_def_rk + 0.50 * emp_pos_rk))

                            # --- 2. MEDIAN L5 MINUTES (BLOWOUT-PROOF) & FRESH-INJURY USAGE BOOST (+6% CAP) ---
                            min_ratio = median_l5_min / max(12.0, szn_min)
                            if min_ratio < 0.90:
                                min_weight_mult = float(np.clip(min_ratio, 0.80, 0.92))
                            else:
                                min_weight_mult = 0.50 + 0.50 * float(np.clip(min_ratio, 0.90, 1.10))

                            usage_boost_mult, missing_team_impact, missing_team_mins = calc_teammate_usage_boost(
                                norm_p, p_team, p_pos, stat_code, player_db, confirmed_out, questionable_gtd
                            )

                            pace_mult = exp_pace / avg_pace
                            lambda_proj = base_avg * (0.75 * matchup_mult + 0.25 * pace_mult) * min_weight_mult * usage_boost_mult

                            if lambda_proj > line:
                                if fg_spread_abs >= 12.0:
                                    lambda_proj *= 0.92
                                elif fg_spread_abs >= 8.5:
                                    lambda_proj *= 0.96

                            # --- 3. V16 22% MAX PROJECTION-GAP CAP (BLOCKS STALE ROLE/LINEUP TRAPS) ---
                            if (abs(lambda_proj - line) / max(1.0, line)) > MAX_FG_PROP_GAP_PCT:
                                continue

                            # --- 4. WIDENED REALISTIC VARIANCE (OVERDISPERSED NBA DISTRIBUTIONS) ---
                            if stat_code == "PRA":
                                std_val = 1.85 * np.sqrt(max(6.0, lambda_proj))
                                prob_under = norm.cdf(line, loc=lambda_proj, scale=std_val)
                                prob_over = 1.0 - prob_under
                            elif stat_code == "PTS":
                                std_val = 1.60 * np.sqrt(max(4.0, lambda_proj))
                                prob_under = norm.cdf(line, loc=lambda_proj, scale=std_val)
                                prob_over = 1.0 - prob_under
                            elif stat_code in ["REB", "AST"]:
                                std_val = 1.22 * np.sqrt(max(2.0, lambda_proj))
                                prob_under = norm.cdf(line, loc=lambda_proj, scale=std_val)
                                prob_over = 1.0 - prob_under
                            else:
                                p_under_pois = poisson.cdf(int(np.floor(line)), lambda_proj)
                                p_under_norm = norm.cdf(line, loc=lambda_proj, scale=1.20 * np.sqrt(max(1.0, lambda_proj)))
                                prob_under = 0.50 * p_under_pois + 0.50 * p_under_norm
                                prob_over = 1.0 - prob_under

                            if prob_over >= prob_under:
                                pick_dir, stat_win_prob, juice = "Over", prob_over, int(sides["Over"]["price"])
                            else:
                                pick_dir, stat_win_prob, juice = "Under", prob_under, int(sides["Under"]["price"])

                            # Block heavy juice (-141 or worse)
                            if juice < MAX_FG_PROP_JUICE:
                                continue

                            # --- 5. V16 POSITIONAL DvP CONFIRMATION & L5 TREND CHECK (NO 7/10 GATE) ---
                            if (pick_dir == "Over" and emp_pos_rk < 18) or (pick_dir == "Under" and emp_pos_rk > 13):
                                continue
                            if pick_dir == "Over" and l5_avg < line and l5_avg < (szn_avg * 0.90):
                                continue
                            if pick_dir == "Under" and l5_avg > line and l5_avg > (szn_avg * 1.10):
                                continue

                            if pick_dir == "Over" and min_ratio <= 0.85 and usage_boost_mult < 1.03:
                                continue

                            # --- 6. SPOT-STARTER / FRESH STAR-ABSENCE / PRA-JUMP UNDER SAFEGUARD ---
                            if is_spot_starter_under_trap(pick_dir, stat_code, line, szn_avg, l10_avg, szn_min, usage_boost_mult, missing_team_impact, missing_team_mins):
                                continue

                            # --- 7. PINNACLE-WEIGHTED 60/40 MARKET BLEND & >= 50% CONSENSUS GUARD ---
                            mkt_cons, mkt_prob = get_devigged_total_or_prop_consensus(other_ev_books, m_key, pick_dir, line, p_name)
                            if mkt_cons is not None:
                                if (pick_dir == "Over" and (mkt_cons - line) <= -1.0) or (pick_dir == "Under" and (mkt_cons - line) >= 1.0):
                                    continue
                            if mkt_prob is not None and mkt_prob < 0.50:
                                continue

                            win_prob = (0.60 * stat_win_prob + 0.40 * mkt_prob) if mkt_prob is not None else (0.85 * stat_win_prob)
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_PLAYER_PROPS:
                                dot = get_matchup_dot(blended_def_rk, pick_dir)
                                if dot == "🔴":
                                    continue
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice, max_units=MAX_FG_PROP_UNITS)
                                usage_tag = " ⚡+Usage" if (usage_boost_mult >= 1.03 and pick_dir == "Over") else ""
                                base_text = (
                                    f"**{p_name} — {pick_dir} {line} {stat_code}** `({juice:+d})`\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Szn: **{szn_avg:.1f}** *(L10: {l10_avg:.1f} | Proj: {lambda_proj:.1f})*\n"
                                    f"{dot} vs {opp_ab_str} Def: **#{opp_def_rk}** *({p_pos}: #{emp_pos_rk})*{usage_tag}  •  🏀 {a_ab}@{h_ab} *({time_short})*\n"
                                    f"{DIVIDER}"
                                )
                                payload = {
                                    "bet_key": bet_key,
                                    "away": away,
                                    "home": home,
                                    "market": "PLAYER_PROP",
                                    "player": p_name,
                                    "player_team": p_team,
                                    "stat": stat_code,
                                    "line": line,
                                    "pick_dir": pick_dir,
                                    "juice": juice,
                                    "units": units,
                                    "dot": dot,
                                    "base_text": base_text,
                                    "status": "PENDING"
                                }
                                slate_fg_prop_candidates.append((edge, norm_p, p_team, base_text, payload))

        # --- C. POST SINGLE HIGHEST-EV TOTAL FOR THIS GAME (FG vs 1H vs 1Q) ---
        if game_total_candidates and not has_posted_game_total:
            best_edge, best_text, best_payload = max(game_total_candidates, key=lambda x: x[0])
            msg_id = post_alert_to_discord(best_text)
            if msg_id:
                best_payload["message_id"] = msg_id
                tracked_bets.append(best_payload)
                already_posted.add(best_payload["bet_key"])

    # --- D. POST TOP FULL-GAME PROPS (1/PLAYER, 1/TEAM, MAX 5/DAY) ---
    remaining_fg_slots = max(0, MAX_FG_PROPS_PER_DAY - posted_fg_props_today_count)
    if slate_fg_prop_candidates and remaining_fg_slots > 0:
        best_by_player = {}
        for item in slate_fg_prop_candidates:
            edge, norm_p, p_team, base_text, payload = item
            if (today_ct_str, norm_p) in posted_players_today or (today_ct_str, p_team) in posted_fg_teams_today:
                continue
            if norm_p not in best_by_player or edge > best_by_player[norm_p][0]:
                best_by_player[norm_p] = item

        best_by_team = {}
        for item in best_by_player.values():
            edge, norm_p, p_team, base_text, payload = item
            if p_team not in best_by_team or edge > best_by_team[p_team][0]:
                best_by_team[p_team] = item

        top_fg_plays = sorted(best_by_team.values(), key=lambda x: x[0], reverse=True)[:remaining_fg_slots]
        for edge, norm_p, p_team, base_text, payload in top_fg_plays:
            msg_id = post_alert_to_discord(base_text)
            if msg_id:
                payload["message_id"] = msg_id
                tracked_bets.append(payload)
                already_posted.add(payload["bet_key"])
                posted_players_today.add((today_ct_str, norm_p))
                posted_fg_teams_today.add((today_ct_str, p_team))

    # Print 1Q Diagnostic Summary to GitHub Actions Logs
    for stat_k in ["PTS", "REB", "AST"]:
        books_str = ", ".join(f"{BOOK_LABELS.get(bk, bk)}:{cnt}" for bk, cnt in sorted(q1_offers_by_book[stat_k].items())) or "None"
        st = q1_filter_stats[stat_k]
        print(
            f"🎯 1Q {stat_k} Audit — Books: [{books_str}] | "
            f"Players:{st['raw_players']} -> Posted:{st['posted']} "
            f"(NoStart:{st['not_starter']}, LowLogs:{st['low_logs']}, <7/10:{st['lt_7_hits']}, "
            f"Juice<-165:{st['heavy_juice']}, Red:{st['red_dot']}, Cons/Inj:{st['cons_or_inj']})"
        )

    save_json(TRACKER_FILE, tracked_bets)

def post_daily_recaps_if_complete(tracked_bets):
    """Automatically posts a Daily W-L & Unit ROI Recap Card once a slate finishes grading."""
    recaps_posted = set(load_json(RECAP_POSTED_FILE, []))
    bets_by_date = {}
    for b in tracked_bets:
        d_str = b.get("bet_key", "")[:10]
        if len(d_str) == 10 and d_str[4] == "-" and d_str[7] == "-":
            bets_by_date.setdefault(d_str, []).append(b)

    for d_str, b_list in sorted(bets_by_date.items()):
        if d_str in recaps_posted or len(b_list) < 3:
            continue
        if any(b.get("status") == "PENDING" for b in b_list):
            continue

        wins = [b for b in b_list if b.get("result") == "WIN"]
        losses = [b for b in b_list if b.get("result") == "LOSS"]
        tot_w, tot_l = len(wins), len(losses)
        if (tot_w + tot_l) == 0:
            continue

        win_pct = (tot_w / (tot_w + tot_l)) * 100.0
        net_kelly_u = 0.0
        net_flat_u = 0.0
        for b in b_list:
            u = float(b.get("units", 1.0))
            j = int(b.get("juice", -110))
            if b.get("result") == "WIN":
                net_kelly_u += u * profit_per_unit(j)
                net_flat_u += 1.0 * profit_per_unit(j)
            elif b.get("result") == "LOSS":
                net_kelly_u -= u
                net_flat_u -= 1.0

        prop_w = sum(1 for b in wins if b.get("market") == "PLAYER_PROP")
        prop_l = sum(1 for b in losses if b.get("market") == "PLAYER_PROP")
        q1p_w = sum(1 for b in wins if b.get("market") == "PLAYER_PROP_Q1")
        q1p_l = sum(1 for b in losses if b.get("market") == "PLAYER_PROP_Q1")
        fg_w = sum(1 for b in wins if b.get("market") in ["FG_TOTAL", "FG_SPREAD"])
        fg_l = sum(1 for b in losses if b.get("market") in ["FG_TOTAL", "FG_SPREAD"])
        per_w = sum(1 for b in wins if b.get("market") in ["TOTALS_Q1", "SPREADS_Q1", "TOTALS_H1", "SPREADS_H1"])
        per_l = sum(1 for b in losses if b.get("market") in ["TOTALS_Q1", "SPREADS_Q1", "TOTALS_H1", "SPREADS_H1"])

        recap_text = (
            f"**📊 NBA BOT DAILY RECAP ({d_str})**\n"
            f"🏆 **Record: {tot_w}–{tot_l} ({win_pct:.1f}%)**  •  💰 **Kelly: {net_kelly_u:+.2f}u** *(Flat: {net_flat_u:+.2f}u)*\n"
            f"🏀 FG Props: **{prop_w}–{prop_l}**  •  🎯 1Q Props: **{q1p_w}–{q1p_l}**  •  🛡️ FG: **{fg_w}–{fg_l}**  •  ⚡ 1Q/1H: **{per_w}–{per_l}**\n"
            f"{DIVIDER}"
        )
        if post_alert_to_discord(recap_text, prefix=""):
            recaps_posted.add(d_str)
            save_json(RECAP_POSTED_FILE, list(recaps_posted))

def check_scores_and_grade_all_markets():
    tracked_bets = load_json(TRACKER_FILE, [])
    pending = [b for b in tracked_bets if b["status"] == "PENDING"]
    if not pending:
        post_daily_recaps_if_complete(tracked_bets)
        return

    central_tz = pytz.timezone("America/Chicago")
    now_ct = datetime.now(timezone.utc).astimezone(central_tz)
    date_pairs = [
        ((now_ct - timedelta(days=d)).strftime("%Y%m%d"), (now_ct - timedelta(days=d)).strftime("%Y-%m-%d"))
        for d in [2, 1, 0]
    ]

    events_with_dates = []
    for d_espn, d_iso in date_pairs:
        try:
            espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_espn}"
            res = requests.get(espn_url, timeout=10).json()
            for ev in res.get("events", []):
                events_with_dates.append((d_iso, ev))
        except:
            continue

    updated = False
    for d_iso, event in events_with_dates:
        is_final = event["status"]["type"]["completed"]
        period = event["status"].get("period", 0)
        comp = event["competitions"][0]
        competitors = comp["competitors"]

        home_c = next(c for c in competitors if c["homeAway"] == "home")
        away_c = next(c for c in competitors if c["homeAway"] == "away")

        h_name = normalize_team_name(home_c["team"]["displayName"])
        a_name = normalize_team_name(away_c["team"]["displayName"])

        h_pts = float(home_c.get("score", 0))
        a_pts = float(away_c.get("score", 0))

        h_lines = [float(x.get("displayValue") or x.get("value") or 0) for x in home_c.get("linescores", [])]
        a_lines = [float(x.get("displayValue") or x.get("value") or 0) for x in away_c.get("linescores", [])]

        player_stats_map = {}
        q1_player_stats_map = {}

        has_fg_props = any(
            b["status"] == "PENDING"
            and b["market"] == "PLAYER_PROP"
            and b["home"] == h_name
            and b["bet_key"].startswith(d_iso)
            for b in tracked_bets
        )
        has_q1_props = any(
            b["status"] == "PENDING"
            and b["market"] == "PLAYER_PROP_Q1"
            and b["home"] == h_name
            and b["bet_key"].startswith(d_iso)
            for b in tracked_bets
        )

        if (is_final and has_fg_props) or ((period > 1 or is_final) and has_q1_props):
            sum_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event['id']}"
            try:
                sum_data = requests.get(sum_url, timeout=10).json()
                if (period > 1 or is_final) and has_q1_props:
                    q1_player_stats_map = extract_q1_player_stats_from_summary(sum_data)

                if is_final and has_fg_props:
                    for team_box in sum_data.get("boxscore", {}).get("players", []):
                        stat_Block = team_box["statistics"][0]
                        labels = stat_Block["labels"]
                        pts_idx = labels.index("PTS") if "PTS" in labels else -1
                        reb_idx = labels.index("REB") if "REB" in labels else -1
                        ast_idx = labels.index("AST") if "AST" in labels else -1
                        tpm_idx = labels.index("3PT") if "3PT" in labels else -1
                        for ath in stat_Block.get("athletes", []):
                            p_disp = normalize_name(ath["athlete"]["displayName"])
                            s_vals = ath.get("stats", [])
                            if s_vals:
                                pts_v = float(s_vals[pts_idx]) if pts_idx != -1 else 0.0
                                reb_v = float(s_vals[reb_idx]) if reb_idx != -1 else 0.0
                                ast_v = float(s_vals[ast_idx]) if ast_idx != -1 else 0.0
                                tpm_raw = str(s_vals[tpm_idx]).split("-")[0] if tpm_idx != -1 else "0"
                                tpm_v = float(tpm_raw) if tpm_raw.isdigit() else 0.0
                                player_stats_map[p_disp] = {
                                    "PTS": pts_v,
                                    "REB": reb_v,
                                    "AST": ast_v,
                                    "3PM": tpm_v,
                                    "PRA": pts_v + reb_v + ast_v
                                }
            except:
                pass

        for bet in tracked_bets:
            if (
                bet["status"] != "PENDING"
                or bet["home"] != h_name
                or bet["away"] != a_name
                or not bet["bet_key"].startswith(d_iso)
            ):
                continue

            outcome = None
            mkt = bet["market"]

            if mkt == "TOTALS_Q1" and (period > 1 or is_final) and len(h_lines) >= 1 and len(a_lines) >= 1:
                q1_tot = h_lines[0] + a_lines[0]
                if q1_tot == bet["line"]:
                    outcome = "PUSH"
                else:
                    outcome = "WIN" if ((q1_tot > bet["line"]) if bet["pick_dir"] == "Over" else (q1_tot < bet["line"])) else "LOSS"
            elif mkt == "SPREADS_Q1" and (period > 1 or is_final) and len(h_lines) >= 1 and len(a_lines) >= 1:
                margin = (h_lines[0] + bet["line"] - a_lines[0]) if bet["pick_team"] == h_name else (a_lines[0] + bet["line"] - h_lines[0])
                outcome = "PUSH" if margin == 0 else ("WIN" if margin > 0 else "LOSS")
            elif mkt == "TOTALS_H1" and (period > 2 or is_final) and len(h_lines) >= 2 and len(a_lines) >= 2:
                h1_tot = sum(h_lines[:2]) + sum(a_lines[:2])
                if h1_tot == bet["line"]:
                    outcome = "PUSH"
                else:
                    outcome = "WIN" if ((h1_tot > bet["line"]) if bet["pick_dir"] == "Over" else (h1_tot < bet["line"])) else "LOSS"
            elif mkt == "SPREADS_H1" and (period > 2 or is_final) and len(h_lines) >= 2 and len(a_lines) >= 2:
                h1_h = sum(h_lines[:2])
                h1_a = sum(a_lines[:2])
                margin = (h1_h + bet["line"] - h1_a) if bet["pick_team"] == h_name else (h1_a + bet["line"] - h1_h)
                outcome = "PUSH" if margin == 0 else ("WIN" if margin > 0 else "LOSS")
            elif mkt == "FG_TOTAL" and is_final:
                fg_tot = h_pts + a_pts
                if fg_tot == bet["line"]:
                    outcome = "PUSH"
                else:
                    outcome = "WIN" if ((fg_tot > bet["line"]) if bet["pick_dir"] == "Over" else (fg_tot < bet["line"])) else "LOSS"
            elif mkt == "FG_SPREAD" and is_final:
                margin = (h_pts + bet["line"] - a_pts) if bet["pick_team"] == h_name else (a_pts + bet["line"] - h_pts)
                outcome = "PUSH" if margin == 0 else ("WIN" if margin > 0 else "LOSS")
            elif mkt == "PLAYER_PROP" and is_final:
                norm_bet_p = normalize_name(bet["player"])
                if norm_bet_p in player_stats_map:
                    actual_stat = player_stats_map[norm_bet_p].get(bet["stat"], 0.0)
                    if actual_stat == bet["line"]:
                        outcome = "PUSH"
                    else:
                        outcome = "WIN" if ((actual_stat > bet["line"]) if bet["pick_dir"] == "Over" else (actual_stat < bet["line"])) else "LOSS"
            elif mkt == "PLAYER_PROP_Q1" and (period > 1 or is_final):
                norm_bet_p = normalize_name(bet["player"])
                if norm_bet_p in q1_player_stats_map:
                    actual_q1_stat = q1_player_stats_map[norm_bet_p].get(bet["stat"], 0.0)
                    if actual_q1_stat == bet["line"]:
                        outcome = "PUSH"
                    else:
                        outcome = "WIN" if ((actual_q1_stat > bet["line"]) if bet["pick_dir"] == "Over" else (actual_q1_stat < bet["line"])) else "LOSS"

            if outcome is not None and DISCORD_WEBHOOK_URL:
                icon = "✅" if outcome == "WIN" else ("➖" if outcome == "PUSH" else "❌")
                patch_res = requests.patch(f"{DISCORD_WEBHOOK_URL}/messages/{bet['message_id']}", json={"content": f"{icon} {bet['base_text']}"})
                if patch_res.status_code == 200:
                    bet["status"] = "GRADED"
                    bet["result"] = outcome
                    updated = True

    if updated:
        save_json(TRACKER_FILE, tracked_bets)
    post_daily_recaps_if_complete(tracked_bets)

if __name__ == "__main__":
    if "--grade-only" in sys.argv:
        print("🏁 Running V16 Overnight Grading & Daily Recap Check...")
        check_scores_and_grade_all_markets()
        print("✅ Grading pass complete.")
        sys.exit(0)

    print("🤖 LIVE NBA EV BOT (V16: LIVE SPLITS/PACE + DVP/FORM FG PROPS + V15 1Q) IS LIVE!")
    start_ts = time.time()
    last_scan = 0
    while (time.time() - start_ts) < MAX_RUNTIME_SECONDS:
        now_ts = time.time()
        if (now_ts - last_scan) >= FANDUEL_SCAN_SECONDS:
            scan_all_fanduel_markets()
            last_scan = now_ts
        check_scores_and_grade_all_markets()
        time.sleep(SCORE_CHECK_SECONDS)

    print("⏰ Reached 5.5h clean-exit window. Running final grading check before saving state...")
    check_scores_and_grade_all_markets()
