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
SCAN_PLAYER_PROPS = True      # Starters & Role Players: PTS, REB, AST

MAX_DEEP_SCAN_GAMES_PER_RUN = 8
MIN_EDGE_TO_POST = 0.040      # STRICT 4.0% OR BETTER +EV FILTER
LOOKBACK_DAYS_FOR_STATS = 12  # Rolling days of ESPN boxscores for live player/opponent stats

TRACKER_FILE = "tracked_all_markets.json"
SCANNED_EVENTS_FILE = "scanned_prop_events.json"
LIVE_STATS_CACHE_FILE = "live_nba_stats_cache.json"

SCORE_CHECK_SECONDS = 300     # Checks ESPN scores every 5 mins (FREE)
FANDUEL_SCAN_SECONDS = 7200   # Scans FanDuel every 2 hours

NBA_STD_DEV_TOTAL = 16.0
NBA_STD_DEV_SPREAD = 12.5
NBA_STD_DEV_H1_TOTAL = 11.5
NBA_STD_DEV_H1_SPREAD = 8.8
NBA_STD_DEV_Q1_TOTAL = 8.2
NBA_STD_DEV_Q1_SPREAD = 6.2
HOME_COURT_ADVANTAGE = 2.2

def normalize_name(name):
    """Strips accents and suffixes so ESPN and FanDuel player names always match."""
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

def get_team_efficiency_baselines():
    return {
        "San Antonio Spurs": {"off": 119.4, "def": 107.5, "pace": 100.8, "opp_reb": 42.1, "opp_ast": 24.8},
        "Houston Rockets": {"off": 119.2, "def": 109.7, "pace": 99.5, "opp_reb": 41.5, "opp_ast": 24.2},
        "Golden State Warriors": {"off": 121.3, "def": 111.7, "pace": 101.2, "opp_reb": 44.2, "opp_ast": 26.1},
        "Minnesota Timberwolves": {"off": 117.2, "def": 106.0, "pace": 98.4, "opp_reb": 41.2, "opp_ast": 23.9},
        "Oklahoma City Thunder": {"off": 113.4, "def": 105.6, "pace": 100.5, "opp_reb": 45.8, "opp_ast": 24.0},
        "Los Angeles Lakers": {"off": 122.3, "def": 114.7, "pace": 101.4, "opp_reb": 44.6, "opp_ast": 27.4},
        "Portland Trail Blazers": {"off": 119.6, "def": 113.8, "pace": 99.8, "opp_reb": 44.9, "opp_ast": 26.8},
        "Memphis Grizzlies": {"off": 117.3, "def": 110.2, "pace": 100.9, "opp_reb": 43.2, "opp_ast": 25.4},
        "Dallas Mavericks": {"off": 115.8, "def": 111.0, "pace": 99.2, "opp_reb": 43.8, "opp_ast": 25.9},
        "Denver Nuggets": {"off": 116.0, "def": 115.6, "pace": 98.1, "opp_reb": 42.6, "opp_ast": 26.5},
        "Detroit Pistons": {"off": 114.3, "def": 106.2, "pace": 99.0, "opp_reb": 42.0, "opp_ast": 24.5},
        "Boston Celtics": {"off": 119.6, "def": 108.6, "pace": 98.7, "opp_reb": 42.4, "opp_ast": 24.1},
        "Los Angeles Clippers": {"off": 111.9, "def": 116.0, "pace": 97.8, "opp_reb": 43.5, "opp_ast": 26.2},
        "Phoenix Suns": {"off": 112.6, "def": 121.5, "pace": 99.1, "opp_reb": 45.1, "opp_ast": 27.6},
        "Toronto Raptors": {"off": 116.6, "def": 114.0, "pace": 100.2, "opp_reb": 44.0, "opp_ast": 26.7},
        "Atlanta Hawks": {"off": 115.5, "def": 113.3, "pace": 101.8, "opp_reb": 45.3, "opp_ast": 27.8},
        "New York Knicks": {"off": 114.6, "def": 108.5, "pace": 96.9, "opp_reb": 40.8, "opp_ast": 24.3},
        "Cleveland Cavaliers": {"off": 115.9, "def": 114.5, "pace": 98.5, "opp_reb": 43.1, "opp_ast": 25.6},
        "New Orleans Pelicans": {"off": 113.2, "def": 118.6, "pace": 99.4, "opp_reb": 45.6, "opp_ast": 27.9},
        "Miami Heat": {"off": 111.8, "def": 116.9, "pace": 97.2, "opp_reb": 43.4, "opp_ast": 26.0},
        "Utah Jazz": {"off": 104.4, "def": 116.0, "pace": 99.6, "opp_reb": 46.2, "opp_ast": 28.4},
        "Philadelphia 76ers": {"off": 111.3, "def": 108.3, "pace": 98.0, "opp_reb": 43.0, "opp_ast": 25.0},
        "Sacramento Kings": {"off": 108.5, "def": 121.6, "pace": 100.4, "opp_reb": 44.8, "opp_ast": 27.5},
        "Indiana Pacers": {"off": 110.4, "def": 114.3, "pace": 102.1, "opp_reb": 45.9, "opp_ast": 27.1},
        "Orlando Magic": {"off": 108.2, "def": 117.4, "pace": 98.3, "opp_reb": 41.9, "opp_ast": 24.6},
        "Washington Wizards": {"off": 109.0, "def": 114.2, "pace": 101.5, "opp_reb": 47.4, "opp_ast": 28.9},
        "Brooklyn Nets": {"off": 106.3, "def": 115.3, "pace": 97.6, "opp_reb": 44.5, "opp_ast": 26.4},
        "Milwaukee Bucks": {"off": 108.6, "def": 121.9, "pace": 99.3, "opp_reb": 45.2, "opp_ast": 27.2},
        "Charlotte Hornets": {"off": 103.5, "def": 115.6, "pace": 98.8, "opp_reb": 44.7, "opp_ast": 26.9},
        "Chicago Bulls": {"off": 108.6, "def": 124.6, "pace": 100.7, "opp_reb": 46.5, "opp_ast": 28.2}
    }

def get_fallback_player_baselines():
    """Expanded baseline dictionary of Starters & Role Players blended with live ESPN boxscores."""
    raw_baselines = {
        # Boston Celtics
        "Jayson Tatum": {"PTS": 28.4, "REB": 8.5, "AST": 5.4},
        "Jaylen Brown": {"PTS": 25.2, "REB": 6.1, "AST": 4.2},
        "Derrick White": {"PTS": 16.8, "REB": 4.3, "AST": 5.1},
        "Jrue Holiday": {"PTS": 12.8, "REB": 4.8, "AST": 4.6},
        "Kristaps Porzingis": {"PTS": 19.5, "REB": 7.1, "AST": 2.0},
        "Payton Pritchard": {"PTS": 13.4, "REB": 3.2, "AST": 3.8},
        "Al Horford": {"PTS": 8.6, "REB": 6.2, "AST": 2.6},
        # New York Knicks
        "Jalen Brunson": {"PTS": 27.9, "REB": 3.6, "AST": 6.8},
        "Karl-Anthony Towns": {"PTS": 24.5, "REB": 11.8, "AST": 3.1},
        "Mikal Bridges": {"PTS": 17.8, "REB": 3.9, "AST": 3.4},
        "OG Anunoby": {"PTS": 16.4, "REB": 5.1, "AST": 2.2},
        "Josh Hart": {"PTS": 13.2, "REB": 8.6, "AST": 5.2},
        "Miles McBride": {"PTS": 10.4, "REB": 2.1, "AST": 2.4},
        # Oklahoma City Thunder
        "Shai Gilgeous-Alexander": {"PTS": 31.2, "REB": 5.6, "AST": 6.4},
        "Jalen Williams": {"PTS": 21.4, "REB": 5.4, "AST": 5.1},
        "Chet Holmgren": {"PTS": 18.2, "REB": 8.8, "AST": 2.6},
        "Lu Dort": {"PTS": 10.8, "REB": 3.8, "AST": 1.6},
        "Isaiah Hartenstein": {"PTS": 11.4, "REB": 10.8, "AST": 3.6},
        "Alex Caruso": {"PTS": 7.8, "REB": 3.2, "AST": 2.8},
        "Cason Wallace": {"PTS": 8.4, "REB": 2.8, "AST": 2.1},
        "Aaron Wiggins": {"PTS": 9.8, "REB": 3.4, "AST": 1.5},
        # Portland Trail Blazers
        "Damian Lillard": {"PTS": 20.4, "REB": 3.8, "AST": 4.9},
        "Anfernee Simons": {"PTS": 19.6, "REB": 3.1, "AST": 4.8},
        "Jerami Grant": {"PTS": 18.4, "REB": 4.1, "AST": 2.4},
        "Deandre Ayton": {"PTS": 14.8, "REB": 10.4, "AST": 1.6},
        "Scoot Henderson": {"PTS": 13.6, "REB": 3.2, "AST": 5.4},
        "Shaedon Sharpe": {"PTS": 16.2, "REB": 4.4, "AST": 2.6},
        "Deni Avdija": {"PTS": 13.8, "REB": 6.6, "AST": 3.6},
        "Toumani Camara": {"PTS": 9.6, "REB": 5.8, "AST": 1.8},
        "Donovan Clingan": {"PTS": 7.8, "REB": 7.4, "AST": 1.2},
        # Golden State Warriors
        "Stephen Curry": {"PTS": 26.4, "REB": 4.5, "AST": 5.2},
        "Draymond Green": {"PTS": 8.8, "REB": 6.8, "AST": 6.2},
        "Andrew Wiggins": {"PTS": 16.4, "REB": 4.8, "AST": 2.1},
        "Jonathan Kuminga": {"PTS": 15.8, "REB": 4.6, "AST": 2.2},
        "Buddy Hield": {"PTS": 14.6, "REB": 3.4, "AST": 2.1},
        "Brandin Podziemski": {"PTS": 10.2, "REB": 5.1, "AST": 4.2},
        "Trayce Jackson-Davis": {"PTS": 8.9, "REB": 6.4, "AST": 1.8},
        # Houston Rockets
        "Alperen Sengun": {"PTS": 21.6, "REB": 11.2, "AST": 5.1},
        "Jalen Green": {"PTS": 22.4, "REB": 4.8, "AST": 3.4},
        "Fred VanVleet": {"PTS": 15.8, "REB": 4.1, "AST": 6.8},
        "Jabari Smith Jr": {"PTS": 13.6, "REB": 7.8, "AST": 1.6},
        "Dillon Brooks": {"PTS": 12.4, "REB": 3.8, "AST": 1.6},
        "Amen Thompson": {"PTS": 11.8, "REB": 7.2, "AST": 3.4},
        "Tari Eason": {"PTS": 11.2, "REB": 6.4, "AST": 1.4},
        # Orlando Magic
        "Paolo Banchero": {"PTS": 25.8, "REB": 7.6, "AST": 5.4},
        "Franz Wagner": {"PTS": 24.8, "REB": 5.7, "AST": 4.6},
        "Jalen Suggs": {"PTS": 15.6, "REB": 4.2, "AST": 3.8},
        "Wendell Carter Jr": {"PTS": 10.8, "REB": 8.4, "AST": 2.1},
        "Kentavious Caldwell-Pope": {"PTS": 9.4, "REB": 2.4, "AST": 1.9},
        "Cole Anthony": {"PTS": 11.2, "REB": 3.6, "AST": 3.2},
        "Moritz Wagner": {"PTS": 10.6, "REB": 4.8, "AST": 1.4},
        "Anthony Black": {"PTS": 8.4, "REB": 3.1, "AST": 3.6},
        # Utah Jazz
        "Lauri Markkanen": {"PTS": 23.6, "REB": 8.2, "AST": 2.1},
        "Keyonte George": {"PTS": 16.4, "REB": 3.4, "AST": 5.8},
        "Collin Sexton": {"PTS": 17.8, "REB": 2.8, "AST": 4.4},
        "John Collins": {"PTS": 16.2, "REB": 8.4, "AST": 2.1},
        "Walker Kessler": {"PTS": 10.4, "REB": 10.8, "AST": 1.4},
        "Jordan Clarkson": {"PTS": 15.4, "REB": 3.2, "AST": 3.8},
        # Los Angeles Lakers & Clippers
        "LeBron James": {"PTS": 24.2, "REB": 7.4, "AST": 8.9},
        "Anthony Davis": {"PTS": 26.8, "REB": 12.4, "AST": 3.6},
        "Austin Reaves": {"PTS": 17.6, "REB": 4.2, "AST": 5.4},
        "Rui Hachimura": {"PTS": 13.4, "REB": 5.2, "AST": 1.4},
        "D'Angelo Russell": {"PTS": 14.2, "REB": 2.8, "AST": 5.6},
        "Dalton Knecht": {"PTS": 11.4, "REB": 3.2, "AST": 1.4},
        "James Harden": {"PTS": 21.4, "REB": 5.8, "AST": 8.9},
        "Norman Powell": {"PTS": 22.6, "REB": 3.4, "AST": 2.4},
        "Ivica Zubac": {"PTS": 15.8, "REB": 12.2, "AST": 2.4},
        "Derrick Jones Jr": {"PTS": 10.2, "REB": 3.6, "AST": 1.1},
        "Terance Mann": {"PTS": 8.4, "REB": 3.4, "AST": 2.1},
        # Denver, Dallas, Minnesota, Phoenix, San Antonio, Sacramento, Memphis
        "Nikola Jokic": {"PTS": 29.1, "REB": 13.2, "AST": 10.4},
        "Jamal Murray": {"PTS": 21.8, "REB": 4.1, "AST": 6.5},
        "Michael Porter Jr": {"PTS": 18.2, "REB": 7.1, "AST": 2.4},
        "Aaron Gordon": {"PTS": 14.6, "REB": 6.4, "AST": 3.2},
        "Christian Braun": {"PTS": 13.8, "REB": 5.1, "AST": 2.2},
        "Russell Westbrook": {"PTS": 11.4, "REB": 4.6, "AST": 5.8},
        "Luka Doncic": {"PTS": 34.5, "REB": 9.1, "AST": 9.6},
        "Kyrie Irving": {"PTS": 25.4, "REB": 4.9, "AST": 5.3},
        "Klay Thompson": {"PTS": 14.2, "REB": 3.4, "AST": 2.0},
        "PJ Washington": {"PTS": 12.4, "REB": 7.6, "AST": 1.9},
        "Daniel Gafford": {"PTS": 11.8, "REB": 7.2, "AST": 1.4},
        "Dereck Lively II": {"PTS": 9.4, "REB": 7.8, "AST": 2.1},
        "Anthony Edwards": {"PTS": 28.1, "REB": 5.8, "AST": 5.1},
        "Julius Randle": {"PTS": 21.2, "REB": 7.4, "AST": 4.6},
        "Rudy Gobert": {"PTS": 11.4, "REB": 11.2, "AST": 1.6},
        "Naz Reid": {"PTS": 14.4, "REB": 5.6, "AST": 1.8},
        "Jaden McDaniels": {"PTS": 10.6, "REB": 4.1, "AST": 1.6},
        "Donte DiVincenzo": {"PTS": 11.8, "REB": 3.8, "AST": 3.4},
        "Mike Conley": {"PTS": 9.2, "REB": 2.8, "AST": 5.4},
        "Kevin Durant": {"PTS": 27.4, "REB": 6.6, "AST": 4.8},
        "Devin Booker": {"PTS": 27.8, "REB": 4.6, "AST": 6.9},
        "Bradley Beal": {"PTS": 17.8, "REB": 4.1, "AST": 4.6},
        "Jusuf Nurkic": {"PTS": 9.6, "REB": 9.8, "AST": 2.8},
        "Tyus Jones": {"PTS": 10.4, "REB": 2.8, "AST": 6.4},
        "Grayson Allen": {"PTS": 11.6, "REB": 3.2, "AST": 2.4},
        "Victor Wembanyama": {"PTS": 26.8, "REB": 11.2, "AST": 4.1},
        "Devin Vassell": {"PTS": 17.6, "REB": 3.8, "AST": 3.6},
        "Jeremy Sochan": {"PTS": 13.8, "REB": 7.8, "AST": 3.2},
        "Chris Paul": {"PTS": 9.8, "REB": 3.8, "AST": 8.2},
        "Harrison Barnes": {"PTS": 11.4, "REB": 4.4, "AST": 1.8},
        "Stephon Castle": {"PTS": 11.2, "REB": 3.2, "AST": 3.8},
        "Domantas Sabonis": {"PTS": 19.8, "REB": 14.1, "AST": 7.8},
        "De'Aaron Fox": {"PTS": 26.9, "REB": 4.6, "AST": 6.1},
        "DeMar DeRozan": {"PTS": 22.8, "REB": 4.2, "AST": 4.4},
        "Keegan Murray": {"PTS": 13.8, "REB": 7.2, "AST": 1.8},
        "Malik Monk": {"PTS": 15.2, "REB": 3.2, "AST": 4.8},
        "Ja Morant": {"PTS": 25.4, "REB": 5.1, "AST": 8.4},
        "Jaren Jackson Jr": {"PTS": 22.4, "REB": 6.1, "AST": 1.8},
        "Desmond Bane": {"PTS": 19.8, "REB": 4.6, "AST": 4.8},
        "Santi Aldama": {"PTS": 12.6, "REB": 6.8, "AST": 2.4},
        "Scotty Pippen Jr": {"PTS": 11.4, "REB": 3.8, "AST": 5.6},
        "Zach Edey": {"PTS": 10.8, "REB": 7.6, "AST": 1.0},
        # East Rest of League
        "Giannis Antetokounmpo": {"PTS": 30.8, "REB": 11.6, "AST": 6.1},
        "Bobby Portis": {"PTS": 13.4, "REB": 7.8, "AST": 1.4},
        "Brook Lopez": {"PTS": 11.8, "REB": 5.2, "AST": 1.6},
        "Gary Trent Jr": {"PTS": 11.2, "REB": 2.6, "AST": 1.6},
        "Donovan Mitchell": {"PTS": 27.2, "REB": 4.4, "AST": 5.2},
        "Darius Garland": {"PTS": 20.4, "REB": 2.8, "AST": 6.6},
        "Evan Mobley": {"PTS": 18.2, "REB": 9.2, "AST": 2.8},
        "Jarrett Allen": {"PTS": 14.6, "REB": 10.8, "AST": 2.1},
        "Caris LeVert": {"PTS": 11.8, "REB": 3.4, "AST": 4.2},
        "Tyrese Maxey": {"PTS": 27.6, "REB": 3.8, "AST": 6.4},
        "Joel Embiid": {"PTS": 26.8, "REB": 10.4, "AST": 4.8},
        "Paul George": {"PTS": 18.6, "REB": 5.6, "AST": 4.4},
        "Kelly Oubre Jr": {"PTS": 13.8, "REB": 4.8, "AST": 1.6},
        "Jared McCain": {"PTS": 15.2, "REB": 2.6, "AST": 2.8},
        "Bam Adebayo": {"PTS": 20.2, "REB": 10.4, "AST": 4.1},
        "Tyler Herro": {"PTS": 23.4, "REB": 5.2, "AST": 4.8},
        "Jimmy Butler": {"PTS": 18.8, "REB": 5.4, "AST": 4.8},
        "Terry Rozier": {"PTS": 13.4, "REB": 3.8, "AST": 4.2},
        "Jaime Jaquez Jr": {"PTS": 10.8, "REB": 4.8, "AST": 2.4},
        "Tyrese Haliburton": {"PTS": 21.4, "REB": 3.9, "AST": 10.4},
        "Pascal Siakam": {"PTS": 20.6, "REB": 7.2, "AST": 3.8},
        "Myles Turner": {"PTS": 15.8, "REB": 7.1, "AST": 1.6},
        "Bennedict Mathurin": {"PTS": 16.8, "REB": 5.8, "AST": 2.0},
        "Andrew Nembhard": {"PTS": 10.8, "REB": 3.1, "AST": 4.8},
        "TJ McConnell": {"PTS": 10.2, "REB": 2.6, "AST": 5.2},
        "Trae Young": {"PTS": 26.2, "REB": 3.1, "AST": 11.2},
        "Jalen Johnson": {"PTS": 19.6, "REB": 10.1, "AST": 5.2},
        "Dyson Daniels": {"PTS": 13.8, "REB": 5.1, "AST": 3.6},
        "Clint Capela": {"PTS": 10.4, "REB": 9.2, "AST": 1.2},
        "Onyeka Okongwu": {"PTS": 11.2, "REB": 6.8, "AST": 1.6},
        "Zaccharie Risacher": {"PTS": 11.6, "REB": 3.8, "AST": 1.4},
        "Cade Cunningham": {"PTS": 24.6, "REB": 6.4, "AST": 9.1},
        "Jaden Ivey": {"PTS": 17.8, "REB": 4.2, "AST": 4.2},
        "Tobias Harris": {"PTS": 13.8, "REB": 6.8, "AST": 2.4},
        "Jalen Duren": {"PTS": 10.6, "REB": 10.2, "AST": 2.4},
        "Malik Beasley": {"PTS": 15.4, "REB": 3.1, "AST": 1.8},
        "LaMelo Ball": {"PTS": 26.9, "REB": 5.4, "AST": 8.2},
        "Brandon Miller": {"PTS": 21.2, "REB": 5.2, "AST": 3.6},
        "Miles Bridges": {"PTS": 17.4, "REB": 6.8, "AST": 3.1},
        "Zach LaVine": {"PTS": 23.4, "REB": 4.8, "AST": 4.1},
        "Nikola Vucevic": {"PTS": 18.2, "REB": 10.8, "AST": 3.4},
        "Coby White": {"PTS": 19.4, "REB": 4.2, "AST": 4.8},
        "Josh Giddey": {"PTS": 13.2, "REB": 6.8, "AST": 6.9},
        "Ayo Dosunmu": {"PTS": 12.4, "REB": 3.6, "AST": 3.8},
        "Scottie Barnes": {"PTS": 20.8, "REB": 8.2, "AST": 6.4},
        "RJ Barrett": {"PTS": 22.6, "REB": 6.2, "AST": 5.8},
        "Immanuel Quickley": {"PTS": 17.4, "REB": 3.8, "AST": 6.4},
        "Jakob Poeltl": {"PTS": 15.2, "REB": 11.4, "AST": 2.6},
        "Gradey Dick": {"PTS": 16.8, "REB": 3.4, "AST": 2.1},
        "Cam Thomas": {"PTS": 24.2, "REB": 3.4, "AST": 3.4},
        "Dennis Schroder": {"PTS": 17.8, "REB": 3.1, "AST": 6.4},
        "Cameron Johnson": {"PTS": 18.2, "REB": 4.4, "AST": 2.9},
        "Nic Claxton": {"PTS": 10.4, "REB": 8.4, "AST": 2.2},
        "Jordan Poole": {"PTS": 20.2, "REB": 2.6, "AST": 4.8},
        "Kyle Kuzma": {"PTS": 17.8, "REB": 6.2, "AST": 3.2},
        "Alex Sarr": {"PTS": 10.8, "REB": 6.6, "AST": 2.1},
        "Bilal Coulibaly": {"PTS": 13.4, "REB": 5.4, "AST": 2.8},
        "Jonas Valanciunas": {"PTS": 12.2, "REB": 8.1, "AST": 2.2},
        "Brandon Ingram": {"PTS": 22.8, "REB": 5.6, "AST": 5.2},
        "Trey Murphy III": {"PTS": 18.4, "REB": 4.8, "AST": 2.6},
        "Yves Missi": {"PTS": 9.2, "REB": 8.4, "AST": 1.4}
    }
    return {normalize_name(k): v for k, v in raw_baselines.items()}

def sync_live_nba_data():
    """
    Automatically pulls live NBA player stats (starters & role players) and
    opponent defensive stats/rankings from ESPN's public endpoints every day.
    """
    team_db = get_team_efficiency_baselines()
    player_db = get_fallback_player_baselines()

    # Check if we already synced today
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = load_json(LIVE_STATS_CACHE_FILE, {})
    if cached.get("date") == today_str and cached.get("player_db") and cached.get("team_db"):
        return cached["team_db"], cached["player_db"], cached["opp_ranks"]

    print("🔄 Syncing live NBA starters, role players, and opponent defensive stats from ESPN...")

    # 1. Pull Live Season Team Points & Opponent Points from ESPN Standings
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

    # 2. Scan Recent Boxscores for All Starters/Role Players + Opponent REB/AST Allowed
    player_logs = {}
    opp_allowed_logs = {t: {"PTS": [], "REB": [], "AST": []} for t in team_db}

    now_utc = datetime.now(timezone.utc)
    for days_ago in range(1, LOOKBACK_DAYS_FOR_STATS + 1):
        d_str = (now_utc - timedelta(days=days_ago)).strftime("%Y%m%d")
        sb_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}"
        try:
            sb_data = requests.get(sb_url, timeout=8).json()
            for ev in sb_data.get("events", []):
                if not ev.get("status", {}).get("type", {}).get("completed"):
                    continue
                ev_id = ev["id"]
                sum_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={ev_id}"
                sum_data = requests.get(sum_url, timeout=8).json()

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

                        t_pts, t_reb, t_ast = 0.0, 0.0, 0.0
                        for ath in stat_blk.get("athletes", []):
                            p_disp = ath.get("athlete", {}).get("displayName", "")
                            s_vals = ath.get("stats", [])
                            if not p_disp or not s_vals or pts_idx == -1:
                                continue
                            try:
                                min_raw = str(s_vals[min_idx]).split(":")[0] if min_idx != -1 else "20"
                                mins = float(min_raw) if min_raw.isdigit() else 0.0
                                pts = float(s_vals[pts_idx]) if pts_idx != -1 else 0.0
                                reb = float(s_vals[reb_idx]) if reb_idx != -1 else 0.0
                                ast = float(s_vals[ast_idx]) if ast_idx != -1 else 0.0
                                t_pts += pts
                                t_reb += reb
                                t_ast += ast

                                # Track every starter and rotation role player (12+ mins)
                                if mins >= 12.0:
                                    norm_p = normalize_name(p_disp)
                                    if norm_p not in player_logs:
                                        player_logs[norm_p] = {"PTS": [], "REB": [], "AST": []}
                                    player_logs[norm_p]["PTS"].append(pts)
                                    player_logs[norm_p]["REB"].append(reb)
                                    player_logs[norm_p]["AST"].append(ast)
                            except:
                                continue
                        team_totals[t_name] = {"PTS": t_pts, "REB": t_reb, "AST": t_ast}

                    # Record what each defense allowed to its opponent
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

    # Blend live player boxscore averages into player_db (adds all role players automatically!)
    for norm_p, logs in player_logs.items():
        if len(logs["PTS"]) >= 2:
            live_pts = float(np.mean(logs["PTS"]))
            live_reb = float(np.mean(logs["REB"]))
            live_ast = float(np.mean(logs["AST"]))
            if norm_p in player_db:
                # 60% live recent form + 40% season baseline prior
                player_db[norm_p] = {
                    "PTS": round(0.60 * live_pts + 0.40 * player_db[norm_p]["PTS"], 2),
                    "REB": round(0.60 * live_reb + 0.40 * player_db[norm_p]["REB"], 2),
                    "AST": round(0.60 * live_ast + 0.40 * player_db[norm_p]["AST"], 2)
                }
            else:
                player_db[norm_p] = {
                    "PTS": round(live_pts, 2),
                    "REB": round(live_reb, 2),
                    "AST": round(live_ast, 2)
                }

    # Update opponent REB/AST allowed averages
    for t_name, logs in opp_allowed_logs.items():
        if len(logs["REB"]) >= 2:
            team_db[t_name]["opp_reb"] = round(0.65 * float(np.mean(logs["REB"])) + 0.35 * team_db[t_name]["opp_reb"], 1)
        if len(logs["AST"]) >= 2:
            team_db[t_name]["opp_ast"] = round(0.65 * float(np.mean(logs["AST"])) + 0.35 * team_db[t_name]["opp_ast"], 1)
        if len(logs["PTS"]) >= 2:
            team_db[t_name]["def"] = round(0.65 * float(np.mean(logs["PTS"])) + 0.35 * team_db[t_name]["def"], 1)

    # 3. Compute League-Wide Opponent Defensive Rankings (#1 Toughest to #30 Softest)
    pts_sorted = sorted(team_db.keys(), key=lambda x: team_db[x]["def"])
    reb_sorted = sorted(team_db.keys(), key=lambda x: team_db[x]["opp_reb"])
    ast_sorted = sorted(team_db.keys(), key=lambda x: team_db[x]["opp_ast"])

    opp_ranks = {}
    for t_name in team_db:
        opp_ranks[t_name] = {
            "PTS": pts_sorted.index(t_name) + 1,
            "REB": reb_sorted.index(t_name) + 1,
            "AST": ast_sorted.index(t_name) + 1
        }

    save_json(LIVE_STATS_CACHE_FILE, {
        "date": today_str,
        "team_db": team_db,
        "player_db": player_db,
        "opp_ranks": opp_ranks
    })
    print(f"✅ Synced {len(player_db)} active NBA players & 30 team defensive rankings!")
    return team_db, player_db, opp_ranks

def fetch_injured_players():
    """Pulls live injured/out player names from ESPN's league-wide injury feed + scoreboard."""
    injured_set = set()
    # 1. League-wide injury endpoint
    try:
        inj_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        res = requests.get(inj_url, timeout=10).json()
        for team_entry in res.get("injuries", []):
            for inj in team_entry.get("injuries", []):
                p_name = inj.get("athlete", {}).get("displayName")
                status = inj.get("status", "").lower()
                if p_name and any(term in status for term in ["out", "doubtful", "questionable", "injured", "day-to-day"]):
                    injured_set.add(normalize_name(p_name))
    except:
        pass

    # 2. Scoreboard injury endpoint
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        res = requests.get(url, timeout=10).json()
        for event in res.get("events", []):
            for team in event["competitions"][0]["competitors"]:
                for inj in team.get("injuries", []):
                    p_name = inj.get("athlete", {}).get("displayName")
                    status = inj.get("status", "").lower()
                    if p_name and any(term in status for term in ["out", "doubtful", "questionable", "injured"]):
                        injured_set.add(normalize_name(p_name))
    except:
        pass
    return injured_set

def format_matchup_tag(opp_rank, pick_dir):
    """Returns an insight badge showing whether the opponent defense favors the pick."""
    if pick_dir == "Over":
        if opp_rank >= 21:
            return "🟢 Favorable vs Weak Def"
        elif opp_rank <= 10:
            return "🔴 Tough Def"
        return "🟡 Neutral Def"
    else:
        if opp_rank <= 10:
            return "🟢 Favorable vs Elite Def"
        elif opp_rank >= 21:
            return "🔴 Soft Def"
        return "🟡 Neutral Def"

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

    stats_db, player_db, opp_ranks = sync_live_nba_data()
    avg_off = float(np.mean([v["off"] for v in stats_db.values()]))
    avg_pace = float(np.mean([v["pace"] for v in stats_db.values()]))
    avg_opp_reb = float(np.mean([v["opp_reb"] for v in stats_db.values()]))
    avg_opp_ast = float(np.mean([v["opp_ast"] for v in stats_db.values()]))

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
        if date_str != today_ct_str:
            continue

        event_id = g["id"]
        home = normalize_team_name(g["home_team"])
        away = normalize_team_name(g["away_team"])
        time_ct = dt_utc.astimezone(central_tz).strftime("%-I:%M %p CT")
        match_key = f"{date_str}_{away}_vs_{home}"

        h_stats = stats_db.get(home, {"off": avg_off, "def": avg_off, "pace": avg_pace, "opp_reb": avg_opp_reb, "opp_ast": avg_opp_ast})
        a_stats = stats_db.get(away, {"off": avg_off, "def": avg_off, "pace": avg_pace, "opp_reb": avg_opp_reb, "opp_ast": avg_opp_ast})
        h_ranks = opp_ranks.get(home, {"PTS": 15, "REB": 15, "AST": 15})
        a_ranks = opp_ranks.get(away, {"PTS": 15, "REB": 15, "AST": 15})

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
                                base_text = (
                                    f"🏀 NBA: {away} vs {home}\n"
                                    f"{time_ct} - Total: {line} {pick_dir} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV\n"
                                    f"📊 Proj: {model_total:.1f} | Def Ranks: {away} #{a_ranks['PTS']} vs {home} #{h_ranks['PTS']}"
                                )
                                msg_id = post_alert_to_discord(base_text)
                                if msg_id:
                                    tracked_bets.append({"bet_key": bet_key, "message_id": msg_id, "away": away, "home": home, "market": "FG_TOTAL", "line": line, "pick_dir": pick_dir, "base_text": base_text, "status": "PENDING"})
                                    already_posted.add(bet_key)

                elif m["key"] == "spreads":
                    bet_key = f"{match_key}_FG_SPREAD"
                    if bet_key not in already_posted:
                        home_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == home), None)
                        away_o = next((o for o in m["outcomes"] if normalize_team_name(o["name"]) == away), None)
                        if home_o and away_o:
                            h_line = float(home_o["point"])
                            sp_diff = model_margin - (-h_line)
                            pick_team = home if sp_diff >= 0 else away
                            opp_team = away if sp_diff >= 0 else home
                            pick_line = h_line if sp_diff >= 0 else float(away_o["point"])
                            juice = int(home_o["price"]) if sp_diff >= 0 else int(away_o["price"])
                            win_prob = norm.cdf(abs(sp_diff) / NBA_STD_DEV_SPREAD)
                            edge = win_prob - american_to_implied_prob(juice)
                            if edge >= MIN_EDGE_TO_POST:
                                ev_pct = edge * 100.0
                                opp_def_rk = opp_ranks.get(opp_team, {}).get("PTS", 15)
                                base_text = (
                                    f"🏀 NBA: {away} vs {home}\n"
                                    f"{time_ct} - Spread: {pick_team} {pick_line:+.1f} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV\n"
                                    f"📊 Fair Line: {pick_team} {(-model_margin if pick_team == home else model_margin):+.1f} | Opp Def: {opp_team} #{opp_def_rk} PPG"
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
                                proj_p_tot = model_total * scale
                                diff = proj_p_tot - line
                                pick_dir = "Over" if diff >= 0 else "Under"
                                juice = int(over_o["price"]) if diff >= 0 else int(under_o["price"])
                                win_prob = norm.cdf(abs(diff) / std_dev)
                                edge = win_prob - american_to_implied_prob(juice)
                                if edge >= MIN_EDGE_TO_POST:
                                    ev_pct = edge * 100.0
                                    base_text = (
                                        f"🏀 NBA: {away} vs {home}\n"
                                        f"{time_ct} - {period_label}: {line} {pick_dir} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV\n"
                                        f"📊 Proj: {proj_p_tot:.1f} | Pace: {exp_pace:.1f}"
                                    )
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
                                norm_p = normalize_name(p_name)
                                # Skip if player is on injury report or missing from live database
                                if norm_p in injured_players or norm_p not in player_db or "Over" not in sides or "Under" not in sides:
                                    continue
                                bet_key = f"{match_key}_{norm_p}_{stat_code}"
                                if bet_key in already_posted:
                                    continue

                                line = float(sides["Over"]["point"])
                                base_avg = player_db[norm_p][stat_code]

                                # Matchup-specific multiplier & defensive rank insight
                                if stat_code == "PTS":
                                    matchup_mult = ((h_stats["def"] + a_stats["def"]) / 2.0) / avg_off
                                    combined_def_rk = int(round((h_ranks["PTS"] + a_ranks["PTS"]) / 2.0))
                                    opp_stat_val = (h_stats["def"] + a_stats["def"]) / 2.0
                                elif stat_code == "REB":
                                    matchup_mult = ((h_stats["opp_reb"] + a_stats["opp_reb"]) / 2.0) / avg_opp_reb
                                    combined_def_rk = int(round((h_ranks["REB"] + a_ranks["REB"]) / 2.0))
                                    opp_stat_val = (h_stats["opp_reb"] + a_stats["opp_reb"]) / 2.0
                                else:
                                    matchup_mult = ((h_stats["opp_ast"] + a_stats["opp_ast"]) / 2.0) / avg_opp_ast
                                    combined_def_rk = int(round((h_ranks["AST"] + a_ranks["AST"]) / 2.0))
                                    opp_stat_val = (h_stats["opp_ast"] + a_stats["opp_ast"]) / 2.0

                                pace_mult = exp_pace / avg_pace
                                lambda_proj = base_avg * (0.75 * matchup_mult + 0.25 * pace_mult)

                                prob_under = poisson.cdf(int(np.floor(line)), lambda_proj)
                                prob_over = 1.0 - prob_under

                                if prob_over >= prob_under:
                                    pick_dir, win_prob, juice = "Over", prob_over, int(sides["Over"]["price"])
                                else:
                                    pick_dir, win_prob, juice = "Under", prob_under, int(sides["Under"]["price"])

                                edge = win_prob - american_to_implied_prob(juice)
                                if edge >= MIN_EDGE_TO_POST:
                                    ev_pct = edge * 100.0
                                    matchup_badge = format_matchup_tag(combined_def_rk, pick_dir)
                                    base_text = (
                                        f"🏀 NBA Prop: {p_name} ({away} vs {home})\n"
                                        f"{time_ct} - {stat_code}: {line} {pick_dir} ({juice:+d}) | 🔥 +{ev_pct:.1f}% EV\n"
                                        f"📊 Avg: {base_avg:.1f} (Proj: {lambda_proj:.1f}) | Matchup Def: #{combined_def_rk} ({opp_stat_val:.1f} {stat_code}/g - {matchup_badge})"
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

        h_name = normalize_team_name(home_c["team"]["displayName"])
        a_name = normalize_team_name(away_c["team"]["displayName"])

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
                        p_disp = normalize_name(ath["athlete"]["displayName"])
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
            elif mkt == "PLAYER_PROP" and is_final:
                norm_bet_p = normalize_name(bet["player"])
                if norm_bet_p in player_stats_map:
                    actual_stat = player_stats_map[norm_bet_p][bet["stat"]]
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
    print("🤖 LIVE NBA STARTERS + ROLE PLAYERS + OPPONENT RANKINGS BOT IS LIVE!")
    last_scan = 0
    while True:
        now_ts = time.time()
        if (now_ts - last_scan) >= FANDUEL_SCAN_SECONDS:
            scan_all_fanduel_markets()
            last_scan = now_ts
        check_scores_and_grade_all_markets()
        time.sleep(SCORE_CHECK_SECONDS)
