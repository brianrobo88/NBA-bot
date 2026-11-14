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
# 1. CREDENTIALS & 15.0% MINIMUM EV FILTER
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "9ecb23956a7654b951782fd28ec3a8b5").strip()

# Market Toggles
SCAN_FULL_GAME = True         # Full-Game Spreads & Totals
SCAN_HALVES_QUARTERS = True   # 1st Half & 1st Quarter Spreads & Totals
SCAN_PLAYER_PROPS = True      # Starters & Role Players: PTS, REB, AST
SCAN_PRA_AND_3PM = False      # Optional: Set True to also scan PRA & 3PM (uses +2 credits/game)

# Up to 10 bookmakers cost the exact same 1-region credit in The Odds API
BOOKMAKERS_LIST = "fanduel,draftkings,betmgm,caesars"

MAX_DEEP_SCAN_GAMES_PER_RUN = 8
MIN_EDGE_TO_POST = 0.150      # STRICT 15.0% OR BETTER +EV FILTER
LOOKBACK_DAYS_FOR_STATS = 25  # Covers full season-to-date + rolling L10/L5 game logs

# Minimum Prop Line Filters (Avoids micro-line traps)
MIN_PROP_LINE_PTS = 6.5
MIN_PROP_LINE_OTHER = 2.5
MIN_PROP_LINE_3PM = 1.5

TRACKER_FILE = "tracked_all_markets.json"
SCANNED_EVENTS_FILE = "scanned_prop_events.json"
LIVE_STATS_CACHE_FILE = "live_nba_stats_cache.json"
CACHE_SCHEMA_VERSION = "v4_sharp_consensus"

SCORE_CHECK_SECONDS = 300     # Checks ESPN scores every 5 mins (FREE)
FANDUEL_SCAN_SECONDS = 7200   # Scans FanDuel every 2 hours

NBA_STD_DEV_TOTAL = 16.0
NBA_STD_DEV_SPREAD = 12.5
NBA_STD_DEV_H1_TOTAL = 11.5
NBA_STD_DEV_H1_SPREAD = 8.8
NBA_STD_DEV_Q1_TOTAL = 8.2
NBA_STD_DEV_Q1_SPREAD = 6.2
HOME_COURT_ADVANTAGE = 2.2
SPREAD_SHRINKAGE = 0.65       # Regresses raw efficiency gaps toward mean

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
    if name == "LA Clippers":
        return "Los Angeles Clippers"
    return name

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

def calc_kelly_units(win_prob, odds_val):
    """Calculates Quarter-Kelly unit recommendation bounded between 0.5u and 2.0u."""
    try:
        odds = float(odds_val)
        b = (100.0 / abs(odds)) if odds < 0 else (odds / 100.0)
        q = 1.0 - win_prob
        kelly_f = max(0.0, (b * win_prob - q) / b)
        units = round(kelly_f * 0.25 * 20.0, 1)  # Quarter-Kelly scaled to standard 1u bankroll
        return max(0.5, min(2.0, units))
    except:
        return 1.0

def extract_fanduel_and_consensus(bookmakers_list):
    """Separates FanDuel markets from other major books to build market consensus lines."""
    fd_markets = []
    other_books = []
    for b in bookmakers_list or []:
        if b.get("key") == "fanduel":
            fd_markets = b.get("markets", [])
        else:
            other_books.append(b)
    return fd_markets, other_books

def get_consensus_total_or_prop(other_books, market_key, player_name=None):
    """Returns average line across non-FanDuel sportsbooks for totals or player props."""
    pts = []
    for b in other_books:
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") == "Over" and "point" in o:
                    if player_name is None or normalize_name(o.get("description", "")) == normalize_name(player_name):
                        pts.append(float(o["point"]))
    return round(float(np.mean(pts)), 1) if pts else None

def get_consensus_spread(other_books, market_key, target_team):
    """Returns average spread line across non-FanDuel sportsbooks for a given team."""
    pts = []
    for b in other_books:
        for m in b.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                if normalize_team_name(o.get("name", "")) == target_team and "point" in o:
                    pts.append(float(o["point"]))
    return round(float(np.mean(pts)), 1) if pts else None

def get_team_efficiency_baselines():
    return {
        "San Antonio Spurs": {"off": 119.4, "def": 107.5, "pace": 100.8, "opp_reb": 42.1, "opp_ast": 24.8, "q1_off": 30.2, "q1_def": 27.5, "h1_off": 60.5, "h1_def": 54.2},
        "Houston Rockets": {"off": 119.2, "def": 109.7, "pace": 99.5, "opp_reb": 41.5, "opp_ast": 24.2, "q1_off": 29.8, "q1_def": 27.1, "h1_off": 59.8, "h1_def": 54.8},
        "Golden State Warriors": {"off": 121.3, "def": 111.7, "pace": 101.2, "opp_reb": 44.2, "opp_ast": 26.1, "q1_off": 31.4, "q1_def": 28.2, "h1_off": 62.1, "h1_def": 56.0},
        "Minnesota Timberwolves": {"off": 117.2, "def": 106.0, "pace": 98.4, "opp_reb": 41.2, "opp_ast": 23.9, "q1_off": 29.1, "q1_def": 26.4, "h1_off": 58.4, "h1_def": 53.0},
        "Oklahoma City Thunder": {"off": 113.4, "def": 105.6, "pace": 100.5, "opp_reb": 45.8, "opp_ast": 24.0, "q1_off": 28.8, "q1_def": 26.2, "h1_off": 57.6, "h1_def": 52.8},
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
        "Jayson Tatum": {"PTS": 28.4, "REB": 8.5, "AST": 5.4, "POS": "F", "TEAM": "Boston Celtics"},
        "Jaylen Brown": {"PTS": 25.2, "REB": 6.1, "AST": 4.2, "POS": "G", "TEAM": "Boston Celtics"},
        "Derrick White": {"PTS": 16.8, "REB": 4.3, "AST": 5.1, "POS": "G", "TEAM": "Boston Celtics"},
        "Jrue Holiday": {"PTS": 12.8, "REB": 4.8, "AST": 4.6, "POS": "G", "TEAM": "Boston Celtics"},
        "Kristaps Porzingis": {"PTS": 19.5, "REB": 7.1, "AST": 2.0, "POS": "C", "TEAM": "Boston Celtics"},
        "Payton Pritchard": {"PTS": 13.4, "REB": 3.2, "AST": 3.8, "POS": "G", "TEAM": "Boston Celtics"},
        "Al Horford": {"PTS": 8.6, "REB": 6.2, "AST": 2.6, "POS": "C", "TEAM": "Boston Celtics"},
        "Jalen Brunson": {"PTS": 27.9, "REB": 3.6, "AST": 6.8, "POS": "G", "TEAM": "New York Knicks"},
        "Karl-Anthony Towns": {"PTS": 24.5, "REB": 11.8, "AST": 3.1, "POS": "C", "TEAM": "New York Knicks"},
        "Mikal Bridges": {"PTS": 17.8, "REB": 3.9, "AST": 3.4, "POS": "F", "TEAM": "New York Knicks"},
        "OG Anunoby": {"PTS": 16.4, "REB": 5.1, "AST": 2.2, "POS": "F", "TEAM": "New York Knicks"},
        "Josh Hart": {"PTS": 13.2, "REB": 8.6, "AST": 5.2, "POS": "G", "TEAM": "New York Knicks"},
        "Miles McBride": {"PTS": 10.4, "REB": 2.1, "AST": 2.4, "POS": "G", "TEAM": "New York Knicks"},
        "Shai Gilgeous-Alexander": {"PTS": 31.2, "REB": 5.6, "AST": 6.4, "POS": "G", "TEAM": "Oklahoma City Thunder"},
        "Jalen Williams": {"PTS": 21.4, "REB": 5.4, "AST": 5.1, "POS": "F", "TEAM": "Oklahoma City Thunder"},
        "Chet Holmgren": {"PTS": 18.2, "REB": 8.8, "AST": 2.6, "POS": "C", "TEAM": "Oklahoma City Thunder"},
        "Lu Dort": {"PTS": 10.8, "REB": 3.8, "AST": 1.6, "POS": "G", "TEAM": "Oklahoma City Thunder"},
        "Isaiah Hartenstein": {"PTS": 11.4, "REB": 10.8, "AST": 3.6, "POS": "C", "TEAM": "Oklahoma City Thunder"},
        "Alex Caruso": {"PTS": 7.8, "REB": 3.2, "AST": 2.8, "POS": "G", "TEAM": "Oklahoma City Thunder"},
        "Cason Wallace": {"PTS": 8.4, "REB": 2.8, "AST": 2.1, "POS": "G", "TEAM": "Oklahoma City Thunder"},
        "Aaron Wiggins": {"PTS": 9.8, "REB": 3.4, "AST": 1.5, "POS": "G", "TEAM": "Oklahoma City Thunder"},
        "Damian Lillard": {"PTS": 20.4, "REB": 3.8, "AST": 4.9, "POS": "G", "TEAM": "Portland Trail Blazers"},
        "Anfernee Simons": {"PTS": 19.6, "REB": 3.1, "AST": 4.8, "POS": "G", "TEAM": "Portland Trail Blazers"},
        "Jerami Grant": {"PTS": 18.4, "REB": 4.1, "AST": 2.4, "POS": "F", "TEAM": "Portland Trail Blazers"},
        "Deandre Ayton": {"PTS": 14.8, "REB": 10.4, "AST": 1.6, "POS": "C", "TEAM": "Portland Trail Blazers"},
        "Scoot Henderson": {"PTS": 13.6, "REB": 3.2, "AST": 5.4, "POS": "G", "TEAM": "Portland Trail Blazers"},
        "Shaedon Sharpe": {"PTS": 16.2, "REB": 4.4, "AST": 2.6, "POS": "G", "TEAM": "Portland Trail Blazers"},
        "Deni Avdija": {"PTS": 13.8, "REB": 6.6, "AST": 3.6, "POS": "F", "TEAM": "Portland Trail Blazers"},
        "Toumani Camara": {"PTS": 9.6, "REB": 5.8, "AST": 1.8, "POS": "F", "TEAM": "Portland Trail Blazers"},
        "Donovan Clingan": {"PTS": 7.8, "REB": 7.4, "AST": 1.2, "POS": "C", "TEAM": "Portland Trail Blazers"},
        "Stephen Curry": {"PTS": 26.4, "REB": 4.5, "AST": 5.2, "POS": "G", "TEAM": "Golden State Warriors"},
        "Draymond Green": {"PTS": 8.8, "REB": 6.8, "AST": 6.2, "POS": "F", "TEAM": "Golden State Warriors"},
        "Andrew Wiggins": {"PTS": 16.4, "REB": 4.8, "AST": 2.1, "POS": "F", "TEAM": "Golden State Warriors"},
        "Jonathan Kuminga": {"PTS": 15.8, "REB": 4.6, "AST": 2.2, "POS": "F", "TEAM": "Golden State Warriors"},
        "Buddy Hield": {"PTS": 14.6, "REB": 3.4, "AST": 2.1, "POS": "G", "TEAM": "Golden State Warriors"},
        "Brandin Podziemski": {"PTS": 10.2, "REB": 5.1, "AST": 4.2, "POS": "G", "TEAM": "Golden State Warriors"},
        "Trayce Jackson-Davis": {"PTS": 8.9, "REB": 6.4, "AST": 1.8, "POS": "C", "TEAM": "Golden State Warriors"},
        "Kevin Durant": {"PTS": 27.4, "REB": 6.6, "AST": 4.8, "POS": "F", "TEAM": "Houston Rockets"},
        "Alperen Sengun": {"PTS": 21.6, "REB": 11.2, "AST": 5.1, "POS": "C", "TEAM": "Houston Rockets"},
        "Jalen Green": {"PTS": 22.4, "REB": 4.8, "AST": 3.4, "POS": "G", "TEAM": "Houston Rockets"},
        "Fred VanVleet": {"PTS": 15.8, "REB": 4.1, "AST": 6.8, "POS": "G", "TEAM": "Houston Rockets"},
        "Jabari Smith Jr": {"PTS": 13.6, "REB": 7.8, "AST": 1.6, "POS": "F", "TEAM": "Houston Rockets"},
        "Dillon Brooks": {"PTS": 12.4, "REB": 3.8, "AST": 1.6, "POS": "G", "TEAM": "Houston Rockets"},
        "Amen Thompson": {"PTS": 11.8, "REB": 7.2, "AST": 3.4, "POS": "G", "TEAM": "Houston Rockets"},
        "Tari Eason": {"PTS": 11.2, "REB": 6.4, "AST": 1.4, "POS": "F", "TEAM": "Houston Rockets"},
        "Paolo Banchero": {"PTS": 25.8, "REB": 7.6, "AST": 5.4, "POS": "F", "TEAM": "Orlando Magic"},
        "Franz Wagner": {"PTS": 24.8, "REB": 5.7, "AST": 4.6, "POS": "F", "TEAM": "Orlando Magic"},
        "Jalen Suggs": {"PTS": 15.6, "REB": 4.2, "AST": 3.8, "POS": "G", "TEAM": "Orlando Magic"},
        "Wendell Carter Jr": {"PTS": 10.8, "REB": 8.4, "AST": 2.1, "POS": "C", "TEAM": "Orlando Magic"},
        "Kentavious Caldwell-Pope": {"PTS": 9.4, "REB": 2.4, "AST": 1.9, "POS": "G", "TEAM": "Orlando Magic"},
        "Cole Anthony": {"PTS": 11.2, "REB": 3.6, "AST": 3.2, "POS": "G", "TEAM": "Orlando Magic"},
        "Moritz Wagner": {"PTS": 10.6, "REB": 4.8, "AST": 1.4, "POS": "C", "TEAM": "Orlando Magic"},
        "Anthony Black": {"PTS": 8.4, "REB": 3.1, "AST": 3.6, "POS": "G", "TEAM": "Orlando Magic"},
        "Lauri Markkanen": {"PTS": 23.6, "REB": 8.2, "AST": 2.1, "POS": "F", "TEAM": "Utah Jazz"},
        "Keyonte George": {"PTS": 16.4, "REB": 3.4, "AST": 5.8, "POS": "G", "TEAM": "Utah Jazz"},
        "Collin Sexton": {"PTS": 17.8, "REB": 2.8, "AST": 4.4, "POS": "G", "TEAM": "Utah Jazz"},
        "John Collins": {"PTS": 16.2, "REB": 8.4, "AST": 2.1, "POS": "F", "TEAM": "Utah Jazz"},
        "Walker Kessler": {"PTS": 10.4, "REB": 10.8, "AST": 1.4, "POS": "C", "TEAM": "Utah Jazz"},
        "Jordan Clarkson": {"PTS": 15.4, "REB": 3.2, "AST": 3.8, "POS": "G", "TEAM": "Utah Jazz"},
        "LeBron James": {"PTS": 24.2, "REB": 7.4, "AST": 8.9, "POS": "F", "TEAM": "Los Angeles Lakers"},
        "Anthony Davis": {"PTS": 26.8, "REB": 12.4, "AST": 3.6, "POS": "C", "TEAM": "Washington Wizards"},
        "Austin Reaves": {"PTS": 17.6, "REB": 4.2, "AST": 5.4, "POS": "G", "TEAM": "Los Angeles Lakers"},
        "Rui Hachimura": {"PTS": 13.4, "REB": 5.2, "AST": 1.4, "POS": "F", "TEAM": "Los Angeles Lakers"},
        "D'Angelo Russell": {"PTS": 14.2, "REB": 2.8, "AST": 5.6, "POS": "G", "TEAM": "Los Angeles Lakers"},
        "Dalton Knecht": {"PTS": 11.4, "REB": 3.2, "AST": 1.4, "POS": "G", "TEAM": "Los Angeles Lakers"},
        "James Harden": {"PTS": 21.4, "REB": 5.8, "AST": 8.9, "POS": "G", "TEAM": "Cleveland Cavaliers"},
        "Norman Powell": {"PTS": 22.6, "REB": 3.4, "AST": 2.4, "POS": "G", "TEAM": "Los Angeles Clippers"},
        "Ivica Zubac": {"PTS": 15.8, "REB": 12.2, "AST": 2.4, "POS": "C", "TEAM": "Los Angeles Clippers"},
        "Derrick Jones Jr": {"PTS": 10.2, "REB": 3.6, "AST": 1.1, "POS": "F", "TEAM": "Los Angeles Clippers"},
        "Terance Mann": {"PTS": 8.4, "REB": 3.4, "AST": 2.1, "POS": "G", "TEAM": "Los Angeles Clippers"},
        "Nikola Jokic": {"PTS": 29.1, "REB": 13.2, "AST": 10.4, "POS": "C", "TEAM": "Denver Nuggets"},
        "Jamal Murray": {"PTS": 21.8, "REB": 4.1, "AST": 6.5, "POS": "G", "TEAM": "Denver Nuggets"},
        "Michael Porter Jr": {"PTS": 18.2, "REB": 7.1, "AST": 2.4, "POS": "F", "TEAM": "Denver Nuggets"},
        "Aaron Gordon": {"PTS": 14.6, "REB": 6.4, "AST": 3.2, "POS": "F", "TEAM": "Denver Nuggets"},
        "Christian Braun": {"PTS": 13.8, "REB": 5.1, "AST": 2.2, "POS": "G", "TEAM": "Denver Nuggets"},
        "Russell Westbrook": {"PTS": 11.4, "REB": 4.6, "AST": 5.8, "POS": "G", "TEAM": "Denver Nuggets"},
        "Luka Doncic": {"PTS": 34.5, "REB": 9.1, "AST": 9.6, "POS": "G", "TEAM": "Los Angeles Lakers"},
        "Kyrie Irving": {"PTS": 25.4, "REB": 4.9, "AST": 5.3, "POS": "G", "TEAM": "Dallas Mavericks"},
        "Klay Thompson": {"PTS": 14.2, "REB": 3.4, "AST": 2.0, "POS": "G", "TEAM": "Dallas Mavericks"},
        "PJ Washington": {"PTS": 12.4, "REB": 7.6, "AST": 1.9, "POS": "F", "TEAM": "Dallas Mavericks"},
        "Daniel Gafford": {"PTS": 11.8, "REB": 7.2, "AST": 1.4, "POS": "C", "TEAM": "Dallas Mavericks"},
        "Dereck Lively II": {"PTS": 9.4, "REB": 7.8, "AST": 2.1, "POS": "C", "TEAM": "Dallas Mavericks"},
        "Caleb Martin": {"PTS": 10.4, "REB": 4.2, "AST": 2.2, "POS": "F", "TEAM": "Dallas Mavericks"},
        "Anthony Edwards": {"PTS": 28.1, "REB": 5.8, "AST": 5.1, "POS": "G", "TEAM": "Minnesota Timberwolves"},
        "Julius Randle": {"PTS": 21.2, "REB": 7.4, "AST": 4.6, "POS": "F", "TEAM": "Minnesota Timberwolves"},
        "Rudy Gobert": {"PTS": 11.4, "REB": 11.2, "AST": 1.6, "POS": "C", "TEAM": "Minnesota Timberwolves"},
        "Naz Reid": {"PTS": 14.4, "REB": 5.6, "AST": 1.8, "POS": "C", "TEAM": "Minnesota Timberwolves"},
        "Jaden McDaniels": {"PTS": 13.8, "REB": 4.1, "AST": 1.6, "POS": "F", "TEAM": "Minnesota Timberwolves"},
        "Donte DiVincenzo": {"PTS": 11.8, "REB": 3.8, "AST": 3.4, "POS": "G", "TEAM": "Minnesota Timberwolves"},
        "Mike Conley": {"PTS": 9.2, "REB": 2.8, "AST": 5.4, "POS": "G", "TEAM": "Minnesota Timberwolves"},
        "Terance Shannon Jr": {"PTS": 11.1, "REB": 3.5, "AST": 2.6, "POS": "G", "TEAM": "Minnesota Timberwolves"},
        "Devin Booker": {"PTS": 27.8, "REB": 4.6, "AST": 6.9, "POS": "G", "TEAM": "Phoenix Suns"},
        "Bradley Beal": {"PTS": 17.8, "REB": 4.1, "AST": 4.6, "POS": "G", "TEAM": "Phoenix Suns"},
        "Jusuf Nurkic": {"PTS": 9.6, "REB": 9.8, "AST": 2.8, "POS": "C", "TEAM": "Phoenix Suns"},
        "Tyus Jones": {"PTS": 10.4, "REB": 2.8, "AST": 6.4, "POS": "G", "TEAM": "Phoenix Suns"},
        "Grayson Allen": {"PTS": 13.6, "REB": 3.2, "AST": 2.4, "POS": "G", "TEAM": "Phoenix Suns"},
        "Royce O'Neale": {"PTS": 8.4, "REB": 4.8, "AST": 2.8, "POS": "F", "TEAM": "Phoenix Suns"},
        "Victor Wembanyama": {"PTS": 26.8, "REB": 11.2, "AST": 4.1, "POS": "C", "TEAM": "San Antonio Spurs"},
        "De'Aaron Fox": {"PTS": 24.6, "REB": 4.6, "AST": 6.1, "POS": "G", "TEAM": "San Antonio Spurs"},
        "Devin Vassell": {"PTS": 17.6, "REB": 3.8, "AST": 3.6, "POS": "G", "TEAM": "San Antonio Spurs"},
        "Jeremy Sochan": {"PTS": 13.8, "REB": 7.8, "AST": 3.2, "POS": "F", "TEAM": "San Antonio Spurs"},
        "Chris Paul": {"PTS": 9.8, "REB": 3.8, "AST": 8.2, "POS": "G", "TEAM": "San Antonio Spurs"},
        "Harrison Barnes": {"PTS": 11.4, "REB": 4.4, "AST": 1.8, "POS": "F", "TEAM": "San Antonio Spurs"},
        "Stephon Castle": {"PTS": 11.2, "REB": 3.2, "AST": 3.8, "POS": "G", "TEAM": "San Antonio Spurs"},
        "Domantas Sabonis": {"PTS": 19.8, "REB": 14.1, "AST": 7.8, "POS": "C", "TEAM": "Sacramento Kings"},
        "DeMar DeRozan": {"PTS": 22.8, "REB": 4.2, "AST": 4.4, "POS": "F", "TEAM": "Sacramento Kings"},
        "Keegan Murray": {"PTS": 13.8, "REB": 7.2, "AST": 1.8, "POS": "F", "TEAM": "Sacramento Kings"},
        "Malik Monk": {"PTS": 15.2, "REB": 3.2, "AST": 4.8, "POS": "G", "TEAM": "Sacramento Kings"},
        "Darius Acuff Jr": {"PTS": 23.8, "REB": 3.8, "AST": 5.4, "POS": "G", "TEAM": "Sacramento Kings"},
        "Ja Morant": {"PTS": 25.4, "REB": 5.1, "AST": 8.4, "POS": "G", "TEAM": "Memphis Grizzlies"},
        "Jaren Jackson Jr": {"PTS": 22.4, "REB": 6.1, "AST": 1.8, "POS": "F", "TEAM": "Memphis Grizzlies"},
        "Desmond Bane": {"PTS": 19.8, "REB": 4.6, "AST": 4.8, "POS": "G", "TEAM": "Memphis Grizzlies"},
        "Santi Aldama": {"PTS": 12.6, "REB": 6.8, "AST": 2.4, "POS": "F", "TEAM": "Memphis Grizzlies"},
        "Scotty Pippen Jr": {"PTS": 11.4, "REB": 3.8, "AST": 5.6, "POS": "G", "TEAM": "Memphis Grizzlies"},
        "Zach Edey": {"PTS": 10.8, "REB": 7.6, "AST": 1.0, "POS": "C", "TEAM": "Memphis Grizzlies"},
        "Cam Spencer": {"PTS": 9.3, "REB": 3.3, "AST": 4.8, "POS": "G", "TEAM": "Memphis Grizzlies"},
        "GG Jackson": {"PTS": 7.8, "REB": 3.4, "AST": 1.2, "POS": "F", "TEAM": "Memphis Grizzlies"},
        "Cedric Coward": {"PTS": 11.2, "REB": 7.8, "AST": 2.1, "POS": "F", "TEAM": "Memphis Grizzlies"},
        "Giannis Antetokounmpo": {"PTS": 30.8, "REB": 11.6, "AST": 6.1, "POS": "F", "TEAM": "Milwaukee Bucks"},
        "Bobby Portis": {"PTS": 13.4, "REB": 7.8, "AST": 1.4, "POS": "F", "TEAM": "Milwaukee Bucks"},
        "Brook Lopez": {"PTS": 11.8, "REB": 5.2, "AST": 1.6, "POS": "C", "TEAM": "Milwaukee Bucks"},
        "Gary Trent Jr": {"PTS": 11.2, "REB": 2.6, "AST": 1.6, "POS": "G", "TEAM": "Milwaukee Bucks"},
        "Donovan Mitchell": {"PTS": 27.2, "REB": 4.4, "AST": 5.2, "POS": "G", "TEAM": "Cleveland Cavaliers"},
        "Darius Garland": {"PTS": 20.4, "REB": 2.8, "AST": 6.6, "POS": "G", "TEAM": "Cleveland Cavaliers"},
        "Evan Mobley": {"PTS": 18.2, "REB": 9.2, "AST": 2.8, "POS": "C", "TEAM": "Cleveland Cavaliers"},
        "Jarrett Allen": {"PTS": 14.6, "REB": 10.8, "AST": 2.1, "POS": "C", "TEAM": "Cleveland Cavaliers"},
        "Caris LeVert": {"PTS": 11.8, "REB": 3.4, "AST": 4.2, "POS": "G", "TEAM": "Cleveland Cavaliers"},
        "Dean Wade": {"PTS": 6.2, "REB": 4.2, "AST": 1.6, "POS": "F", "TEAM": "Cleveland Cavaliers"},
        "Tyrese Maxey": {"PTS": 27.6, "REB": 3.8, "AST": 6.4, "POS": "G", "TEAM": "Philadelphia 76ers"},
        "Joel Embiid": {"PTS": 26.8, "REB": 10.4, "AST": 4.8, "POS": "C", "TEAM": "Philadelphia 76ers"},
        "Paul George": {"PTS": 18.6, "REB": 5.6, "AST": 4.4, "POS": "F", "TEAM": "Philadelphia 76ers"},
        "Kelly Oubre Jr": {"PTS": 13.8, "REB": 4.8, "AST": 1.6, "POS": "F", "TEAM": "Philadelphia 76ers"},
        "Jared McCain": {"PTS": 15.2, "REB": 2.6, "AST": 2.8, "POS": "G", "TEAM": "Philadelphia 76ers"},
        "Bam Adebayo": {"PTS": 20.2, "REB": 10.4, "AST": 4.1, "POS": "C", "TEAM": "Miami Heat"},
        "Tyler Herro": {"PTS": 23.4, "REB": 5.2, "AST": 4.8, "POS": "G", "TEAM": "Miami Heat"},
        "Jimmy Butler": {"PTS": 18.8, "REB": 5.4, "AST": 4.8, "POS": "F", "TEAM": "Miami Heat"},
        "Terry Rozier": {"PTS": 13.4, "REB": 3.8, "AST": 4.2, "POS": "G", "TEAM": "Miami Heat"},
        "Jaime Jaquez Jr": {"PTS": 10.8, "REB": 4.8, "AST": 2.4, "POS": "F", "TEAM": "Miami Heat"},
        "Duncan Robinson": {"PTS": 11.8, "REB": 2.4, "AST": 2.2, "POS": "G", "TEAM": "Detroit Pistons"},
        "Tyrese Haliburton": {"PTS": 21.4, "REB": 3.9, "AST": 10.4, "POS": "G", "TEAM": "Indiana Pacers"},
        "Pascal Siakam": {"PTS": 20.6, "REB": 7.2, "AST": 3.8, "POS": "F", "TEAM": "Indiana Pacers"},
        "Myles Turner": {"PTS": 15.8, "REB": 7.1, "AST": 1.6, "POS": "C", "TEAM": "Indiana Pacers"},
        "Bennedict Mathurin": {"PTS": 16.8, "REB": 5.8, "AST": 2.0, "POS": "G", "TEAM": "Indiana Pacers"},
        "Andrew Nembhard": {"PTS": 10.8, "REB": 3.1, "AST": 4.8, "POS": "G", "TEAM": "Indiana Pacers"},
        "TJ McConnell": {"PTS": 10.2, "REB": 2.6, "AST": 5.2, "POS": "G", "TEAM": "Indiana Pacers"},
        "Trae Young": {"PTS": 24.2, "REB": 3.1, "AST": 10.8, "POS": "G", "TEAM": "Washington Wizards"},
        "Jalen Johnson": {"PTS": 19.6, "REB": 10.1, "AST": 5.2, "POS": "F", "TEAM": "Atlanta Hawks"},
        "Dyson Daniels": {"PTS": 13.8, "REB": 5.1, "AST": 3.6, "POS": "G", "TEAM": "Atlanta Hawks"},
        "Clint Capela": {"PTS": 10.4, "REB": 9.2, "AST": 1.2, "POS": "C", "TEAM": "Atlanta Hawks"},
        "Onyeka Okongwu": {"PTS": 11.2, "REB": 6.8, "AST": 1.6, "POS": "C", "TEAM": "Atlanta Hawks"},
        "Zaccharie Risacher": {"PTS": 11.6, "REB": 3.8, "AST": 1.4, "POS": "F", "TEAM": "Atlanta Hawks"},
        "Cade Cunningham": {"PTS": 24.6, "REB": 6.4, "AST": 9.1, "POS": "G", "TEAM": "Detroit Pistons"},
        "Jaden Ivey": {"PTS": 17.8, "REB": 4.2, "AST": 4.2, "POS": "G", "TEAM": "Detroit Pistons"},
        "Tobias Harris": {"PTS": 13.8, "REB": 6.8, "AST": 2.4, "POS": "F", "TEAM": "Detroit Pistons"},
        "Jalen Duren": {"PTS": 10.6, "REB": 10.2, "AST": 2.4, "POS": "C", "TEAM": "Detroit Pistons"},
        "Malik Beasley": {"PTS": 15.4, "REB": 3.1, "AST": 1.8, "POS": "G", "TEAM": "Detroit Pistons"},
        "Paul Reed Jr": {"PTS": 7.4, "REB": 5.2, "AST": 1.4, "POS": "C", "TEAM": "Detroit Pistons"},
        "Kevin Huerter": {"PTS": 11.8, "REB": 3.4, "AST": 2.6, "POS": "G", "TEAM": "Detroit Pistons"},
        "LaMelo Ball": {"PTS": 26.9, "REB": 5.4, "AST": 8.2, "POS": "G", "TEAM": "Charlotte Hornets"},
        "Brandon Miller": {"PTS": 21.2, "REB": 5.2, "AST": 3.6, "POS": "F", "TEAM": "Charlotte Hornets"},
        "Miles Bridges": {"PTS": 17.4, "REB": 6.8, "AST": 3.1, "POS": "F", "TEAM": "Charlotte Hornets"},
        "Tidjane Salaun": {"PTS": 7.6, "REB": 4.4, "AST": 1.4, "POS": "F", "TEAM": "Charlotte Hornets"},
        "Moussa Diabate": {"PTS": 6.4, "REB": 7.8, "AST": 1.0, "POS": "C", "TEAM": "Charlotte Hornets"},
        "Zach LaVine": {"PTS": 23.4, "REB": 4.8, "AST": 4.1, "POS": "G", "TEAM": "Chicago Bulls"},
        "Nikola Vucevic": {"PTS": 18.2, "REB": 10.8, "AST": 3.4, "POS": "C", "TEAM": "Chicago Bulls"},
        "Coby White": {"PTS": 19.4, "REB": 4.2, "AST": 4.8, "POS": "G", "TEAM": "Chicago Bulls"},
        "Josh Giddey": {"PTS": 13.2, "REB": 6.8, "AST": 6.9, "POS": "G", "TEAM": "Chicago Bulls"},
        "Ayo Dosunmu": {"PTS": 12.4, "REB": 3.6, "AST": 3.8, "POS": "G", "TEAM": "Chicago Bulls"},
        "Scottie Barnes": {"PTS": 20.8, "REB": 8.2, "AST": 6.4, "POS": "F", "TEAM": "Toronto Raptors"},
        "RJ Barrett": {"PTS": 22.6, "REB": 6.2, "AST": 5.8, "POS": "F", "TEAM": "Toronto Raptors"},
        "Immanuel Quickley": {"PTS": 17.4, "REB": 3.8, "AST": 6.4, "POS": "G", "TEAM": "Toronto Raptors"},
        "Jakob Poeltl": {"PTS": 14.4, "REB": 11.4, "AST": 2.6, "POS": "C", "TEAM": "Toronto Raptors"},
        "Gradey Dick": {"PTS": 16.8, "REB": 3.4, "AST": 2.1, "POS": "G", "TEAM": "Toronto Raptors"},
        "Cam Thomas": {"PTS": 24.2, "REB": 3.4, "AST": 3.4, "POS": "G", "TEAM": "Brooklyn Nets"},
        "Dennis Schroder": {"PTS": 17.8, "REB": 3.1, "AST": 6.4, "POS": "G", "TEAM": "Brooklyn Nets"},
        "Cameron Johnson": {"PTS": 18.2, "REB": 4.4, "AST": 2.9, "POS": "F", "TEAM": "Brooklyn Nets"},
        "Nic Claxton": {"PTS": 10.4, "REB": 8.4, "AST": 2.2, "POS": "C", "TEAM": "Brooklyn Nets"},
        "Jordan Poole": {"PTS": 20.2, "REB": 2.6, "AST": 4.8, "POS": "G", "TEAM": "Washington Wizards"},
        "Kyle Kuzma": {"PTS": 17.8, "REB": 6.2, "AST": 3.2, "POS": "F", "TEAM": "Washington Wizards"},
        "Alex Sarr": {"PTS": 10.8, "REB": 7.6, "AST": 2.1, "POS": "C", "TEAM": "Washington Wizards"},
        "Bilal Coulibaly": {"PTS": 12.4, "REB": 5.4, "AST": 2.8, "POS": "F", "TEAM": "Washington Wizards"},
        "Kyshawn George": {"PTS": 15.4, "REB": 4.6, "AST": 4.2, "POS": "G", "TEAM": "Washington Wizards"},
        "Jonas Valanciunas": {"PTS": 12.2, "REB": 8.1, "AST": 2.2, "POS": "C", "TEAM": "Washington Wizards"},
        "Brandon Ingram": {"PTS": 22.8, "REB": 5.6, "AST": 5.2, "POS": "F", "TEAM": "New Orleans Pelicans"},
        "Trey Murphy III": {"PTS": 18.4, "REB": 4.8, "AST": 2.6, "POS": "F", "TEAM": "New Orleans Pelicans"},
        "Yves Missi": {"PTS": 9.2, "REB": 8.4, "AST": 1.4, "POS": "C", "TEAM": "New Orleans Pelicans"}
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
                t_raw = ath.get("teamName") or ath.get("team", {}).get("displayName", "")
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

                if pts is not None and reb is not None and ast is not None and mins >= 10.0:
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
        and "b2b_teams" in cached
    ):
        return cached["team_db"], cached["player_db"], cached["opp_ranks"], set(cached["b2b_teams"])

    print("🔄 Syncing official ESPN season averages, L10/L5 logs, minute trends, B2B rest, and period splits...")

    # 1. Layer Official ESPN Season Averages on top of Fallback Dictionary
    espn_season_stats = fetch_espn_season_averages()
    for norm_p, s_data in espn_season_stats.items():
        prev_team = player_db.get(norm_p, {}).get("TEAM", "")
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
            "SZN_MIN": s_data["MIN"],
            "L5_MIN": s_data["MIN"],
            "POS": s_data["POS"],
            "TEAM": s_data["TEAM"] or prev_team
        }

    # Ensure fallback players also carry SZN/L10/MIN/PRA/3PM keys
    for norm_p, p_data in player_db.items():
        p_data.setdefault("3PM", round(p_data["PTS"] * 0.09, 2))
        p_data.setdefault("PRA", round(p_data["PTS"] + p_data["REB"] + p_data["AST"], 2))
        p_data.setdefault("SZN_MIN", 26.0)
        p_data.setdefault("L5_MIN", 26.0)
        for k in ["PTS", "REB", "AST", "3PM", "PRA"]:
            p_data.setdefault(f"SZN_{k}", p_data[k])
            p_data.setdefault(f"L10_{k}", p_data[k])

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
    opp_allowed_logs = {t: {"PTS": [], "REB": [], "AST": []} for t in team_db}
    period_logs = {t: {"q1_off": [], "q1_def": [], "h1_off": [], "h1_def": []} for t in team_db}
    b2b_teams = set()

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

                comps = sum_data.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
                if len(comps) == 2:
                    t1_name = normalize_team_name(comps[0].get("team", {}).get("displayName", ""))
                    t2_name = normalize_team_name(comps[1].get("team", {}).get("displayName", ""))

                    if days_ago == 1:
                        b2b_teams.add(t1_name)
                        b2b_teams.add(t2_name)

                    t1_lines = [float(x.get("displayValue") or x.get("value") or 0) for x in comps[0].get("linescores", [])]
                    t2_lines = [float(x.get("displayValue") or x.get("value") or 0) for x in comps[1].get("linescores", [])]

                    if len(t1_lines) >= 1 and len(t2_lines) >= 1 and t1_lines[0] > 0 and t2_lines[0] > 0 and t1_name in period_logs and t2_name in period_logs:
                        period_logs[t1_name]["q1_off"].append(t1_lines[0])
                        period_logs[t1_name]["q1_def"].append(t2_lines[0])
                        period_logs[t2_name]["q1_off"].append(t2_lines[0])
                        period_logs[t2_name]["q1_def"].append(t1_lines[0])

                    if len(t1_lines) >= 2 and len(t2_lines) >= 2 and (t1_lines[0] + t1_lines[1]) > 0 and (t2_lines[0] + t2_lines[1]) > 0 and t1_name in period_logs and t2_name in period_logs:
                        h1_t1 = t1_lines[0] + t1_lines[1]
                        h1_t2 = t2_lines[0] + t2_lines[1]
                        period_logs[t1_name]["h1_off"].append(h1_t1)
                        period_logs[t1_name]["h1_def"].append(h1_t2)
                        period_logs[t2_name]["h1_off"].append(h1_t2)
                        period_logs[t2_name]["h1_def"].append(h1_t1)

                p_boxes = sum_data.get("boxscore", {}).get("players", [])
                if len(p_boxes) == 2:
                    team_totals = {}
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
                                min_raw = str(s_vals[min_idx]).split(":")[0] if min_idx != -1 else "0"
                                mins = float(min_raw) if min_raw.isdigit() else 0.0
                                if mins <= 0.0:
                                    continue
                                pts = float(s_vals[pts_idx]) if pts_idx != -1 else 0.0
                                reb = float(s_vals[reb_idx]) if reb_idx != -1 else 0.0
                                ast = float(s_vals[ast_idx]) if ast_idx != -1 else 0.0
                                tpm_raw = str(s_vals[tpm_idx]).split("-")[0] if tpm_idx != -1 else "0"
                                tpm = float(tpm_raw) if tpm_raw.isdigit() else 0.0

                                t_pts += pts
                                t_reb += reb
                                t_ast += ast

                                norm_p = normalize_name(p_disp)
                                if norm_p not in player_logs:
                                    player_logs[norm_p] = {"PTS": [], "REB": [], "AST": [], "3PM": [], "PRA": [], "MIN": [], "TEAM": t_name, "POS": p_pos_clean}
                                player_logs[norm_p]["PTS"].append(pts)
                                player_logs[norm_p]["REB"].append(reb)
                                player_logs[norm_p]["AST"].append(ast)
                                player_logs[norm_p]["3PM"].append(tpm)
                                player_logs[norm_p]["PRA"].append(pts + reb + ast)
                                player_logs[norm_p]["MIN"].append(mins)
                                player_logs[norm_p]["TEAM"] = t_name
                            except:
                                continue
                        team_totals[t_name] = {"PTS": t_pts, "REB": t_reb, "AST": t_ast}

                    t_names = list(team_totals.keys())
                    if len(t_names) == 2:
                        t1, t2 = t_names[0], t_names[1]
                        if t1 in opp_allowed_logs and t2 in team_totals:
                            for k in ["PTS", "REB", "AST"]:
                                if team_totals[t2][k] > 0:
                                    opp_allowed_logs[t1][k].append(team_totals[t2][k])
                        if t2 in opp_allowed_logs and t1 in team_totals:
                            for k in ["PTS", "REB", "AST"]:
                                if team_totals[t1][k] > 0:
                                    opp_allowed_logs[t2][k].append(team_totals[t1][k])
        except:
            continue

    # Blend 50% Official Season Avg (SZN) + 50% Last 10 Games (L10) & Track L5 Minute Trends
    for norm_p, logs in player_logs.items():
        if len(logs["PTS"]) >= 3 and float(np.mean(logs["MIN"])) >= 12.0:
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
            l5_min = float(np.mean(logs["MIN"][-5:]))

            if norm_p in espn_season_stats:
                szn_pts = espn_season_stats[norm_p]["PTS"]
                szn_reb = espn_season_stats[norm_p]["REB"]
                szn_ast = espn_season_stats[norm_p]["AST"]
                szn_3pm = espn_season_stats[norm_p]["3PM"]
                szn_pra = espn_season_stats[norm_p]["PRA"]
                szn_min = espn_season_stats[norm_p]["MIN"]
            elif norm_p in player_db and len(logs["PTS"]) < 6:
                szn_pts = 0.50 * full_pts + 0.50 * player_db[norm_p]["SZN_PTS"]
                szn_reb = 0.50 * full_reb + 0.50 * player_db[norm_p]["SZN_REB"]
                szn_ast = 0.50 * full_ast + 0.50 * player_db[norm_p]["SZN_AST"]
                szn_3pm = full_3pm
                szn_pra = szn_pts + szn_reb + szn_ast
                szn_min = full_min
            else:
                szn_pts, szn_reb, szn_ast, szn_3pm, szn_pra, szn_min = full_pts, full_reb, full_ast, full_3pm, full_pra, full_min

            pos_val = player_db.get(norm_p, {}).get("POS", logs.get("POS", "G"))
            player_db[norm_p] = {
                "PTS": round(0.50 * szn_pts + 0.50 * l10_pts, 2),
                "REB": round(0.50 * szn_reb + 0.50 * l10_reb, 2),
                "AST": round(0.50 * szn_ast + 0.50 * l10_ast, 2),
                "3PM": round(0.50 * szn_3pm + 0.50 * l10_3pm, 2),
                "PRA": round(0.50 * szn_pra + 0.50 * l10_pra, 2),
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
                "SZN_MIN": round(szn_min, 1),
                "L5_MIN": round(l5_min, 1),
                "POS": pos_val,
                "TEAM": logs["TEAM"]
            }

    for t_name, logs in opp_allowed_logs.items():
        if len(logs["REB"]) >= 2:
            team_db[t_name]["opp_reb"] = round(0.65 * float(np.mean(logs["REB"])) + 0.35 * team_db[t_name]["opp_reb"], 1)
        if len(logs["AST"]) >= 2:
            team_db[t_name]["opp_ast"] = round(0.65 * float(np.mean(logs["AST"])) + 0.35 * team_db[t_name]["opp_ast"], 1)
        if len(logs["PTS"]) >= 2:
            team_db[t_name]["def"] = round(0.65 * float(np.mean(logs["PTS"])) + 0.35 * team_db[t_name]["def"], 1)

    for t_name, p_logs in period_logs.items():
        if len(p_logs["q1_off"]) >= 2:
            team_db[t_name]["q1_off"] = round(0.60 * float(np.mean(p_logs["q1_off"])) + 0.40 * team_db[t_name]["q1_off"], 1)
            team_db[t_name]["q1_def"] = round(0.60 * float(np.mean(p_logs["q1_def"])) + 0.40 * team_db[t_name]["q1_def"], 1)
        if len(p_logs["h1_off"]) >= 2:
            team_db[t_name]["h1_off"] = round(0.60 * float(np.mean(p_logs["h1_off"])) + 0.40 * team_db[t_name]["h1_off"], 1)
            team_db[t_name]["h1_def"] = round(0.60 * float(np.mean(p_logs["h1_def"])) + 0.40 * team_db[t_name]["h1_def"], 1)

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

    save_json(LIVE_STATS_CACHE_FILE, {
        "date": today_str,
        "schema_version": CACHE_SCHEMA_VERSION,
        "team_db": team_db,
        "player_db": player_db,
        "opp_ranks": opp_ranks,
        "b2b_teams": list(b2b_teams)
    })
    print(f"✅ Synced {len(player_db)} players (Szn + L10 + L5 Min), {len(b2b_teams)} B2B teams, & 30 defensive rankings!")
    return team_db, player_db, opp_ranks, b2b_teams

def fetch_injured_players():
    """Returns (prop_excluded_players, confirmed_out_or_doubtful_players)."""
    prop_excluded = set()
    confirmed_out = set()
    try:
        inj_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        res = requests.get(inj_url, timeout=10).json()
        for team_entry in res.get("injuries", []):
            for inj in team_entry.get("injuries", []):
                p_name = inj.get("athlete", {}).get("displayName")
                status = inj.get("status", "").lower()
                if p_name:
                    norm_p = normalize_name(p_name)
                    if any(term in status for term in ["out", "doubtful", "questionable", "injured", "day-to-day"]):
                        prop_excluded.add(norm_p)
                    if any(term in status for term in ["out", "doubtful", "injured"]):
                        confirmed_out.add(norm_p)
    except:
        pass

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        res = requests.get(url, timeout=10).json()
        for event in res.get("events", []):
            for team in event["competitions"][0]["competitors"]:
                for inj in team.get("injuries", []):
                    p_name = inj.get("athlete", {}).get("displayName")
                    status = inj.get("status", "").lower()
                    if p_name:
                        norm_p = normalize_name(p_name)
                        if any(term in status for term in ["out", "doubtful", "questionable", "injured"]):
                            prop_excluded.add(norm_p)
                        if any(term in status for term in ["out", "doubtful", "injured"]):
                            confirmed_out.add(norm_p)
    except:
        pass
    return prop_excluded, confirmed_out

def calc_team_injury_deduction(team_name, player_db, confirmed_out):
    """Deducts projected team points when key scorers (12+ PPG) are Out/Doubtful."""
    penalty = 0.0
    for norm_p, p_data in player_db.items():
        if p_data.get("TEAM") == team_name and norm_p in confirmed_out:
            pts = p_data.get("SZN_PTS", p_data.get("PTS", 0.0))
            if pts >= 12.0:
                penalty += (pts - 8.5) * 0.18
    return min(penalty, 7.5)

def calc_teammate_usage_boost(norm_p, team_name, stat_code, player_db, confirmed_out):
    """Calculates 'Next Man Up' usage multiplier when high-usage teammates are Out/Doubtful."""
    if not team_name:
        return 1.0
    missing_pts = 0.0
    missing_ast = 0.0
    missing_reb = 0.0
    for other_p, p_data in player_db.items():
        if other_p == norm_p or p_data.get("TEAM") != team_name or other_p not in confirmed_out:
            continue
        pts = p_data.get("SZN_PTS", p_data.get("PTS", 0.0))
        ast = p_data.get("SZN_AST", p_data.get("AST", 0.0))
        reb = p_data.get("SZN_REB", p_data.get("REB", 0.0))
        if pts >= 16.0:
            missing_pts += pts
        if ast >= 5.5:
            missing_ast += ast
        if reb >= 7.5:
            missing_reb += reb

    if stat_code in ["PTS", "3PM"] and missing_pts > 0:
        return min(1.12, 1.0 + (missing_pts * 0.0028))
    elif stat_code == "AST" and missing_ast > 0:
        return min(1.14, 1.0 + (missing_ast * 0.0085))
    elif stat_code == "REB" and missing_reb > 0:
        return min(1.12, 1.0 + (missing_reb * 0.0065))
    elif stat_code == "PRA" and (missing_pts + missing_ast + missing_reb) > 0:
        return min(1.12, 1.0 + ((missing_pts + missing_ast + missing_reb) * 0.0022))
    return 1.0

def get_matchup_dot(opp_rank, pick_dir):
    if pick_dir == "Over":
        return "🟢" if opp_rank >= 21 else ("🔴" if opp_rank <= 10 else "🟡")
    else:
        return "🟢" if opp_rank <= 10 else ("🔴" if opp_rank >= 21 else "🟡")

def post_alert_to_discord(base_text):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ Error: DISCORD_WEBHOOK_URL secret is not set in GitHub Settings!")
        return None
    res = requests.post(f"{DISCORD_WEBHOOK_URL}?wait=true", json={"content": f"⏳ {base_text}"})
    if res.status_code in [200, 204]:
        return res.json()["id"]
    return None

def scan_all_fanduel_markets():
    tracked_bets = load_json(TRACKER_FILE, [])
    already_posted = {b["bet_key"] for b in tracked_bets}
    scanned_events = set(load_json(SCANNED_EVENTS_FILE, []))
    prop_excluded, confirmed_out = fetch_injured_players()

    stats_db, player_db, opp_ranks, b2b_teams = sync_live_nba_data()
    avg_off = float(np.mean([v["off"] for v in stats_db.values()]))
    avg_pace = float(np.mean([v["pace"] for v in stats_db.values()]))
    avg_q1_off = float(np.mean([v["q1_off"] for v in stats_db.values()]))
    avg_h1_off = float(np.mean([v["h1_off"] for v in stats_db.values()]))
    avg_opp_reb = float(np.mean([v["opp_reb"] for v in stats_db.values()]))
    avg_opp_ast = float(np.mean([v["opp_ast"] for v in stats_db.values()]))

    central_tz = pytz.timezone("America/Chicago")
    now_utc = datetime.now(timezone.utc)
    today_ct_str = now_utc.astimezone(central_tz).strftime("%Y-%m-%d")

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us",
        "markets": "totals,spreads",
        "bookmakers": BOOKMAKERS_LIST,
        "oddsFormat": "american"
    }

    res = requests.get(url, params=params, timeout=10)
    print(f"📊 Multi-Book FanDuel Sync for {today_ct_str}! (Credits Remaining: {res.headers.get('x-requests-remaining')})")
    if res.status_code != 200:
        return

    events = res.json()
    deep_scans_done = 0

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

        h_stats = stats_db.get(home, {"off": avg_off, "def": avg_off, "pace": avg_pace, "opp_reb": avg_opp_reb, "opp_ast": avg_opp_ast, "q1_off": avg_q1_off, "q1_def": avg_q1_off, "h1_off": avg_h1_off, "h1_def": avg_h1_off})
        a_stats = stats_db.get(away, {"off": avg_off, "def": avg_off, "pace": avg_pace, "opp_reb": avg_opp_reb, "opp_ast": avg_opp_ast, "q1_off": avg_q1_off, "q1_def": avg_q1_off, "h1_off": avg_h1_off, "h1_def": avg_h1_off})
        h_ranks = opp_ranks.get(home, {"PTS": 15, "REB": 15, "AST": 15, "3PM": 15, "PRA": 15})
        a_ranks = opp_ranks.get(away, {"PTS": 15, "REB": 15, "AST": 15, "3PM": 15, "PRA": 15})

        # 1. Pace Normalization (No Double-Counting)
        exp_pace = (h_stats["pace"] * a_stats["pace"]) / avg_pace
        team_avg_pace = (h_stats["pace"] + a_stats["pace"]) / 2.0
        pace_scaler = exp_pace / team_avg_pace

        raw_h_pts = ((h_stats["off"] + a_stats["def"]) / 2.0) * pace_scaler
        raw_a_pts = ((a_stats["off"] + h_stats["def"]) / 2.0) * pace_scaler

        # 2. Star Injury & Back-to-Back (B2B) Adjustments
        h_inj_pen = calc_team_injury_deduction(home, player_db, confirmed_out)
        a_inj_pen = calc_team_injury_deduction(away, player_db, confirmed_out)

        h_b2b_pen = 2.0 if home in b2b_teams else 0.0
        a_b2b_pen = 2.6 if away in b2b_teams else 0.0

        # 3. 0.65 Bayesian Shrinkage for Sharp Margin & Total
        shrunk_eff_diff = (raw_h_pts - raw_a_pts) * SPREAD_SHRINKAGE
        model_margin = shrunk_eff_diff + HOME_COURT_ADVANTAGE - (h_inj_pen - a_inj_pen) - (h_b2b_pen - a_b2b_pen)

        raw_total = raw_h_pts + raw_a_pts
        league_avg_total = avg_off * 2.0
        model_total = (0.75 * raw_total + 0.25 * league_avg_total) - (h_inj_pen + a_inj_pen) * 0.65 - (h_b2b_pen + a_b2b_pen) * 0.50

        fg_spread_abs = abs(model_margin)

        # --- A. FULL-GAME TOTALS & SPREADS ---
        if SCAN_FULL_GAME:
            for m in fd_markets:
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

                            mkt_cons = get_consensus_total_or_prop(other_books, "totals")
                            if mkt_cons is not None:
                                if (pick_dir == "Over" and (mkt_cons - line) <= -1.5) or (pick_dir == "Under" and (mkt_cons - line) >= 1.5):
                                    continue

                            win_prob = norm.cdf(abs(diff) / NBA_STD_DEV_TOTAL)
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice)
                                mkt_str = f" *(Mkt: {mkt_cons:.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{a_ab} @ {h_ab} — {pick_dir} {line}** `({juice:+d})`\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Proj: **{model_total:.1f}**{mkt_str}\n"
                                    f"🛡️ Def: {a_ab} #{a_ranks['PTS']} vs {h_ab} #{h_ranks['PTS']}  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "FG_TOTAL", "line": line, "pick_dir": pick_dir, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

                elif m["key"] == "spreads":
                    bet_key = f"{match_key}_FG_SPREAD"
                    home_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == home), None)
                    away_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == away), None)
                    if home_o and away_o:
                        h_line = float(home_o["point"])
                        fg_spread_abs = abs(h_line)
                        if bet_key not in already_posted:
                            sp_diff = model_margin - (-h_line)
                            pick_team = home if sp_diff >= 0 else away
                            opp_team = away if sp_diff >= 0 else home
                            pick_ab = get_abbr(pick_team)
                            opp_ab = get_abbr(opp_team)
                            pick_line = h_line if sp_diff >= 0 else float(away_o["point"])
                            juice = int(home_o["price"]) if sp_diff >= 0 else int(away_o["price"])

                            mkt_cons = get_consensus_spread(other_books, "spreads", pick_team)
                            if mkt_cons is not None and (pick_line - mkt_cons) <= -1.5:
                                continue

                            win_prob = norm.cdf(abs(sp_diff) / NBA_STD_DEV_SPREAD)
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice)
                                opp_def_rk = opp_ranks.get(opp_team, {}).get("PTS", 15)
                                fair_line = -model_margin if pick_team == home else model_margin
                                b2b_note = " ⚠️B2B" if opp_team in b2b_teams else ""
                                mkt_str = f" *(Mkt: {mkt_cons:+.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{pick_ab} {pick_line:+.1f}** `({juice:+d})`  •  *{a_ab} @ {h_ab}*\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Fair: **{pick_ab} {fair_line:+.1f}**{mkt_str}\n"
                                    f"🛡️ Opp Def: {opp_ab} **#{opp_def_rk}**{b2b_note}  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "FG_SPREAD", "pick_team": pick_team, "line": pick_line, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

        # --- B. DEEP SCAN: 1H / 1Q & ALL STARTERS / ROLE PLAYER PROPS ---
        if (SCAN_HALVES_QUARTERS or SCAN_PLAYER_PROPS) and (event_id not in scanned_events) and (deep_scans_done < MAX_DEEP_SCAN_GAMES_PER_RUN):
            extra_markets = []
            if SCAN_HALVES_QUARTERS:
                extra_markets.extend(["totals_h1", "spreads_h1", "totals_q1", "spreads_q1"])
            if SCAN_PLAYER_PROPS:
                extra_markets.extend(["player_points", "player_rebounds", "player_assists"])
                if SCAN_PRA_AND_3PM:
                    extra_markets.extend(["player_points_rebounds_assists", "player_threes"])

            ev_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            ev_params = {
                "apiKey": THE_ODDS_API_KEY,
                "regions": "us",
                "markets": ",".join(extra_markets),
                "bookmakers": BOOKMAKERS_LIST,
                "oddsFormat": "american"
            }
            ev_res = requests.get(ev_url, params=ev_params, timeout=10)
            if ev_res.status_code == 200:
                deep_scans_done += 1
                scanned_events.add(event_id)
                ev_data = ev_res.json()

                fd_ev_markets, other_ev_books = extract_fanduel_and_consensus(ev_data.get("bookmakers", []))

                # Pre-calculate period team projections with injury/B2B scaling
                q1_h_pts = ((h_stats["q1_off"] + a_stats["q1_def"]) / 2.0) - (h_inj_pen * 0.22)
                q1_a_pts = ((a_stats["q1_off"] + h_stats["q1_def"]) / 2.0) - (a_inj_pen * 0.22)
                h1_h_pts = ((h_stats["h1_off"] + a_stats["h1_def"]) / 2.0) - (h_inj_pen * 0.46) - (h_b2b_pen * 0.35)
                h1_a_pts = ((a_stats["h1_off"] + h_stats["h1_def"]) / 2.0) - (a_inj_pen * 0.46) - (a_b2b_pen * 0.35)

                for m in fd_ev_markets:
                    m_key = m["key"]

                    if m_key in ["totals_h1", "totals_q1"]:
                        is_q1 = (m_key == "totals_q1")
                        period_label = "1Q" if is_q1 else "1H"
                        std_dev = NBA_STD_DEV_Q1_TOTAL if is_q1 else NBA_STD_DEV_H1_TOTAL
                        bet_key = f"{match_key}_{m_key}"

                        over_o = next((o for o in m["outcomes"] if o["name"] == "Over"), None)
                        under_o = next((o for o in m["outcomes"] if o["name"] == "Under"), None)
                        if over_o and under_o and bet_key not in already_posted:
                            line = float(over_o["point"])
                            proj_period_tot = (q1_h_pts + q1_a_pts) if is_q1 else (h1_h_pts + h1_a_pts)
                            diff = proj_period_tot - line
                            pick_dir = "Over" if diff >= 0 else "Under"
                            juice = int(over_o["price"]) if diff >= 0 else int(under_o["price"])

                            mkt_cons = get_consensus_total_or_prop(other_ev_books, m_key)
                            if mkt_cons is not None:
                                if (pick_dir == "Over" and (mkt_cons - line) <= -1.5) or (pick_dir == "Under" and (mkt_cons - line) >= 1.5):
                                    continue

                            win_prob = norm.cdf(abs(diff) / std_dev)
                            edge = win_prob - american_to_implied_prob(juice)

                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice)
                                mkt_str = f" *(Mkt: {mkt_cons:.1f})*" if mkt_cons is not None else ""
                                base_text = (
                                    f"**{a_ab} @ {h_ab} — {period_label} {pick_dir} {line}** `({juice:+d})`\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Proj: **{proj_period_tot:.1f}**{mkt_str}\n"
                                    f"⚡ Period Matchup Sync  •  🕒 {time_short}\n"
                                    f"{DIVIDER}"
                                )
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": m_key.upper(), "line": line, "pick_dir": pick_dir, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

                    elif m_key in ["spreads_h1", "spreads_q1"]:
                        is_q1 = (m_key == "spreads_q1")
                        period_label = "1Q" if is_q1 else "1H"
                        std_dev = NBA_STD_DEV_Q1_SPREAD if is_q1 else NBA_STD_DEV_H1_SPREAD
                        bet_key = f"{match_key}_{m_key}"

                        home_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == home), None)
                        away_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == away), None)
                        if home_o and away_o and bet_key not in already_posted:
                            h_line = float(home_o["point"])
                            raw_p_diff = (q1_h_pts - q1_a_pts) if is_q1 else (h1_h_pts - h1_a_pts)
                            hca_p = 0.55 if is_q1 else 1.10
                            p_margin = (raw_p_diff * SPREAD_SHRINKAGE) + hca_p
                            sp_diff = p_margin - (-h_line)

                            pick_team = home if sp_diff >= 0 else away
                            pick_ab = get_abbr(pick_team)
                            pick_line = h_line if sp_diff >= 0 else float(away_o["point"])
                            juice = int(home_o["price"]) if sp_diff >= 0 else int(away_o["price"])

                            mkt_cons = get_consensus_spread(other_ev_books, m_key, pick_team)
                            if mkt_cons is not None and (pick_line - mkt_cons) <= -1.5:
                                continue

                            win_prob = norm.cdf(abs(sp_diff) / std_dev)
                            edge = win_prob - american_to_implied_prob(juice)

                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice)
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
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": m_key.upper(), "pick_team": pick_team, "line": pick_line, "base_text": base_text, "status": "PENDING"})
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
                            elif stat_code == "PRA" and line < 12.5:
                                continue

                            p_info = player_db[norm_p]
                            base_avg = p_info.get(stat_code, 0.0)
                            if base_avg <= 0.0:
                                continue
                            szn_avg = p_info.get(f"SZN_{stat_code}", base_avg)
                            l10_avg = p_info.get(f"L10_{stat_code}", base_avg)
                            szn_min = p_info.get("SZN_MIN", 26.0)
                            l5_min = p_info.get("L5_MIN", szn_min)
                            p_pos = p_info.get("POS", "G")
                            p_team = p_info.get("TEAM", "")

                            # --- 1. TRUE SINGLE-OPPONENT DEFENSE MAPPING ---
                            if p_team == home:
                                opp_s, opp_r, opp_ab_str = a_stats, a_ranks, a_ab
                            elif p_team == away:
                                opp_s, opp_r, opp_ab_str = h_stats, h_ranks, h_ab
                            else:
                                opp_s = {
                                    "def": (h_stats["def"] + a_stats["def"]) / 2.0,
                                    "opp_reb": (h_stats["opp_reb"] + a_stats["opp_reb"]) / 2.0,
                                    "opp_ast": (h_stats["opp_ast"] + a_stats["opp_ast"]) / 2.0
                                }
                                opp_r = {
                                    "PTS": int(round((h_ranks["PTS"] + a_ranks["PTS"]) / 2.0)),
                                    "REB": int(round((h_ranks["REB"] + a_ranks["REB"]) / 2.0)),
                                    "AST": int(round((h_ranks["AST"] + a_ranks["AST"]) / 2.0)),
                                    "3PM": int(round((h_ranks["PTS"] + a_ranks["PTS"]) / 2.0)),
                                    "PRA": int(round((h_ranks["PTS"] + a_ranks["PTS"]) / 2.0))
                                }
                                opp_ab_str = f"{a_ab}/{h_ab}"

                            if stat_code in ["PTS", "3PM", "PRA"]:
                                team_def_mult = opp_s["def"] / avg_off
                                opp_def_rk = opp_r["PTS"]
                                opp_stat_val = opp_s["def"]
                            elif stat_code == "REB":
                                team_def_mult = opp_s["opp_reb"] / avg_opp_reb
                                opp_def_rk = opp_r["REB"]
                                opp_stat_val = opp_s["opp_reb"]
                            else:
                                team_def_mult = opp_s["opp_ast"] / avg_opp_ast
                                opp_def_rk = opp_r["AST"]
                                opp_stat_val = opp_s["opp_ast"]

                            pos_modifier = 1.03 if p_pos == "G" and stat_code in ["PTS", "AST", "3PM", "PRA"] else (1.03 if p_pos == "C" and stat_code in ["REB", "PRA"] else 1.0)
                            matchup_mult = (0.60 * team_def_mult) + (0.40 * (team_def_mult * pos_modifier))

                            # --- 2. MINUTE-TREND SCALING & NEXT-MAN-UP USAGE BOOST ---
                            min_trend_mult = float(np.clip(l5_min / max(12.0, szn_min), 0.90, 1.12))
                            usage_boost_mult = calc_teammate_usage_boost(norm_p, p_team, stat_code, player_db, confirmed_out)

                            pace_mult = exp_pace / avg_pace
                            lambda_proj = base_avg * (0.75 * matchup_mult + 0.25 * pace_mult) * (0.50 + 0.50 * min_trend_mult) * usage_boost_mult

                            # Apply blowout 4Q risk discount if full-game spread >= 13.5
                            if fg_spread_abs >= 13.5 and lambda_proj > line:
                                lambda_proj *= 0.95

                            # --- 3. OVERDISPERSED SCORING MATH (PTS/PRA) vs POISSON (REB/AST/3PM) ---
                            if stat_code in ["PTS", "PRA"]:
                                std_pts = 1.42 * np.sqrt(max(4.0, lambda_proj))
                                prob_under = norm.cdf(line, loc=lambda_proj, scale=std_pts)
                                prob_over = 1.0 - prob_under
                            else:
                                prob_under = poisson.cdf(int(np.floor(line)), lambda_proj)
                                prob_over = 1.0 - prob_under

                            if prob_over >= prob_under:
                                pick_dir, win_prob, juice = "Over", prob_over, int(sides["Over"]["price"])
                            else:
                                pick_dir, win_prob, juice = "Under", prob_under, int(sides["Under"]["price"])

                            # --- 4. MULTI-BOOK MARKET CONSENSUS FILTER ---
                            mkt_cons = get_consensus_total_or_prop(other_ev_books, m_key, p_name)
                            if mkt_cons is not None:
                                if (pick_dir == "Over" and (mkt_cons - line) <= -1.5) or (pick_dir == "Under" and (mkt_cons - line) >= 1.5):
                                    continue

                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TO_POST:
                                dot = get_matchup_dot(opp_def_rk, pick_dir)
                                if dot == "🔴":
                                    continue
                                ev_pct = edge * 100.0
                                units = calc_kelly_units(win_prob, juice)
                                usage_tag = " ⚡+Usage" if usage_boost_mult >= 1.04 else ""
                                base_text = (
                                    f"**{p_name} — {pick_dir} {line} {stat_code}** `({juice:+d})`\n"
                                    f"🔥 **+{ev_pct:.1f}% EV** `({units:.1f}u)`  •  📊 Szn: **{szn_avg:.1f}** *(L10: {l10_avg:.1f} | Proj: {lambda_proj:.1f})*\n"
                                    f"{dot} vs {opp_ab_str} Def: **#{opp_def_rk}** *({opp_stat_val:.1f}/g)*{usage_tag}  •  🏀 {a_ab}@{h_ab} *({time_short})*\n"
                                    f"{DIVIDER}"
                                )
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

    central_tz = pytz.timezone("America/Chicago")
    now_ct = datetime.now(timezone.utc).astimezone(central_tz)
    date_pairs = [
        ((now_ct - timedelta(days=1)).strftime("%Y%m%d"), (now_ct - timedelta(days=1)).strftime("%Y-%m-%d")),
        (now_ct.strftime("%Y%m%d"), now_ct.strftime("%Y-%m-%d"))
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
        has_props = any(
            b["status"] == "PENDING"
            and b["market"] == "PLAYER_PROP"
            and b["home"] == h_name
            and b["bet_key"].startswith(d_iso)
            for b in tracked_bets
        )
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

            did_win = None
            mkt = bet["market"]

            if mkt == "TOTALS_Q1" and (period > 1 or is_final) and len(h_lines) >= 1 and len(a_lines) >= 1:
                q1_tot = h_lines[0] + a_lines[0]
                did_win = (q1_tot > bet["line"]) if bet["pick_dir"] == "Over" else (q1_tot < bet["line"])
            elif mkt == "SPREADS_Q1" and (period > 1 or is_final) and len(h_lines) >= 1 and len(a_lines) >= 1:
                if bet["pick_team"] == h_name:
                    did_win = (h_lines[0] + bet["line"]) > a_lines[0]
                else:
                    did_win = (a_lines[0] + bet["line"]) > h_lines[0]
            elif mkt == "TOTALS_H1" and (period > 2 or is_final) and len(h_lines) >= 2 and len(a_lines) >= 2:
                h1_tot = sum(h_lines[:2]) + sum(a_lines[:2])
                did_win = (h1_tot > bet["line"]) if bet["pick_dir"] == "Over" else (h1_tot < bet["line"])
            elif mkt == "SPREADS_H1" and (period > 2 or is_final) and len(h_lines) >= 2 and len(a_lines) >= 2:
                h1_h = sum(h_lines[:2])
                h1_a = sum(a_lines[:2])
                if bet["pick_team"] == h_name:
                    did_win = (h1_h + bet["line"]) > h1_a
                else:
                    did_win = (h1_a + bet["line"]) > h1_h
            elif mkt == "FG_TOTAL" and is_final:
                fg_tot = h_pts + a_pts
                did_win = (fg_tot > bet["line"]) if bet["pick_dir"] == "Over" else (fg_tot < bet["line"])
            elif mkt == "FG_SPREAD" and is_final:
                if bet["pick_team"] == h_name:
                    did_win = (h_pts + bet["line"]) > a_pts
                else:
                    did_win = (a_pts + bet["line"]) > h_pts
            elif mkt == "PLAYER_PROP" and is_final:
                norm_bet_p = normalize_name(bet["player"])
                if norm_bet_p in player_stats_map:
                    actual_stat = player_stats_map[norm_bet_p].get(bet["stat"], 0.0)
                    did_win = (actual_stat > bet["line"]) if bet["pick_dir"] == "Over" else (actual_stat < bet["line"])

            if did_win is not None and DISCORD_WEBHOOK_URL:
                icon = "✅" if did_win else "❌"
                patch_res = requests.patch(f"{DISCORD_WEBHOOK_URL}/messages/{bet['message_id']}", json={"content": f"{icon} {bet['base_text']}"})
                if patch_res.status_code == 200:
                    bet["status"] = "GRADED"
                    updated = True

    if updated:
        save_json(TRACKER_FILE, tracked_bets)

if __name__ == "__main__":
    print("🤖 LIVE NBA 15%+ EV BOT (V4 MULTI-BOOK + USAGE ENGINE) IS LIVE!")
    last_scan = 0
    while True:
        now_ts = time.time()
        if (now_ts - last_scan) >= FANDUEL_SCAN_SECONDS:
            scan_all_fanduel_markets()
            last_scan = now_ts
        check_scores_and_grade_all_markets()
        time.sleep(SCORE_CHECK_SECONDS)
