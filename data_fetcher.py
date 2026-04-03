import pandas as pd
import numpy as np
import requests
import json
import io
import zipfile
import os
from datetime import datetime

# Cricsheet Daily Matches (Last 7 Days) - Typically updated within 24h
RECENT_JSON_URL = "https://cricsheet.org/downloads/recently_played_7_json.zip"

# Local cache for live data
CACHE_FILE = "live_2026_cache.csv"

def fetch_live_data():
    """
    Downloads the last 7 days of match data from Cricsheet, extracts IPL 2026 matches,
    and returns them as a pandas DataFrame.
    """
    try:
        print("Checking for live IPL 2026 matches from Cricsheet...")
        
        # 1. Download Zip
        response = requests.get(RECENT_JSON_URL, timeout=15)
        if response.status_code != 200:
            print(f"Failed to download Cricsheet zip: {response.status_code}")
            return load_cached_data()

        # 2. Open Zip and filter for IPL 2026
        z = zipfile.ZipFile(io.BytesIO(response.content))
        match_records = []
        
        for filename in z.namelist():
            if filename.endswith(".json") and filename != "README.txt":
                with z.open(filename) as f:
                    data = json.load(f)
                    
                    # Filter for IPL 2026
                    info = data.get("info", {})
                    event_name = info.get("event", {}).get("name", "")
                    competition = info.get("competition", "")
                    dates = info.get("dates", [""])[0]
                    
                    is_ipl = "Indian Premier League" in competition or "Indian Premier League" in event_name
                    is_2026 = "2026" in str(info.get("season", "")) or "2026" in str(dates)
                    
                    if is_ipl and is_2026:
                        print(f"  Processing Match: {info.get('teams', [])} on {dates}")
                        records = parse_cricsheet_json(data, filename)
                        match_records.extend(records)

        if not match_records:
            print("No 2026 IPL matches found in the last 7 days of data.")
            return load_cached_data()

        df_live = pd.DataFrame(match_records)
        save_to_cache(df_live)
        
        print(f"Successfully updated with {len(df_live)} deliveries from IPL 2026.")
        return df_live
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return load_cached_data()

def parse_cricsheet_json(data, filename):
    """
    Parses a single Cricsheet JSON into delivery-level records.
    """
    records = []
    info = data.get("info", {})
    match_id = filename.split(".")[0]
    season = "2026"
    start_date = info.get("dates", [""])[0]
    venue = info.get("venue", "")
    
    for inning_idx, inning in enumerate(data.get("innings", [])):
        innings_num = inning_idx + 1
        for over_data in inning.get("overs", []):
            over_num = over_data.get("over", 0)
            
            # Determine Phase
            phase = "Powerplay" if over_num < 6 else ("Middle" if over_num < 15 else "Death")
            
            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                ball_num = ball_idx + 1 
                
                # Extract Runs & Extras
                runs = delivery.get("runs", {})
                runs_off_bat = runs.get("batter", 0)
                total_extras = runs.get("extras", 0)
                
                # Detailed Extras
                ext_detail = delivery.get("extras", {})
                wides = ext_detail.get("wides", 0)
                noballs = ext_detail.get("noballs", 0)
                byes = ext_detail.get("byes", 0)
                legbyes = ext_detail.get("legbyes", 0)
                penalty = ext_detail.get("penalty", 0)
                
                # Extract Wicket
                wickets = delivery.get("wickets", [])
                wicket_type = "0"
                player_dismissed = ""
                if wickets:
                    wicket_type = wickets[0].get("kind", "0")
                    player_dismissed = wickets[0].get("player_out", "")

                records.append({
                    "match_id": match_id,
                    "over": over_num,
                    "ball": ball_num,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "runs_off_bat": runs_off_bat,
                    "wicket_type": wicket_type,
                    "phase": phase
                })
    return records

def load_cached_data():
    if os.path.exists(CACHE_FILE):
        try:
            return pd.read_csv(CACHE_FILE)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_to_cache(df):
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)

if __name__ == "__main__":
    df = fetch_live_data()
    print(df.head())
