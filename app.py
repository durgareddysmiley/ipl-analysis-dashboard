import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# --- 1. Initialize State ---
if "page" not in st.session_state:
    st.session_state.page = "analytics"

# --- Page Config ---
st.set_page_config(page_title="IPL Data Dashboard", layout="wide")

st.title("IPL Cricket Analysis Dashboard")
st.markdown("Explore batting & bowling matchups, phase analysis, and strategic insights using a unified IPL dataset.")

from data_fetcher import fetch_live_data

# --- Data Loading ---
def load_data(force_refresh=False):
    # Load historical data
    df_h = pd.read_csv("cleaned_ipl_data.csv")
    
    # Fetch live data (IPL 2026)
    # If force_refresh is True, we tell Streamlit to re-run the inner fetch
    if force_refresh:
        st.cache_data.clear()
        
    df_l = fetch_live_data()
    
    if not df_l.empty:
        # Merge datasets
        df_merged = pd.concat([df_h, df_l], ignore_index=True)
        return df_merged
        
    return df_h

# Load data
with st.spinner("Syncing Latest IPL 2026 Data..."):
    df = load_data()

# Ensure ball_num exists
if 'ball_num' not in df.columns:
    if 'ball' in df.columns:
        df['ball_num'] = ((df['ball'] - df['ball'].astype(int)) * 10).round().astype(int)
    else:
        df['ball_num'] = 1

# --- Validations & Setup ---
if df.empty:
    st.error("No data found! Please make sure `prepare_data.py` has run successfully.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.markdown("### **Data Sync**")
if st.sidebar.button("🔄 Refresh Latest Matches"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### **Analytics Filters**")

# Responsive Function
def reset_to_analytics():
    st.session_state.page = "analytics"
    st.session_state.nav_select = "Player Performance Analytics"

player_type = st.sidebar.radio("Select Player Role", ["Batsman", "Bowler"], on_change=reset_to_analytics)

# Global player lists
players_list_bat = sorted(df['batter'].dropna().unique())
players_list_bowl = sorted(df['bowler'].dropna().unique())

if player_type == "Batsman":
    selected_player = st.sidebar.selectbox("Select Batsman", players_list_bat, on_change=reset_to_analytics)
    player_data = df[df["batter"] == selected_player]
else:
    selected_player = st.sidebar.selectbox("Select Bowler", players_list_bowl, on_change=reset_to_analytics)
    player_data = df[df["bowler"] == selected_player]

st.sidebar.markdown("---")
st.sidebar.markdown("## **Strategic decision engine**")

# Main Navigation
nav_choice = st.sidebar.selectbox(
    "Navigation & Strategic Decisions",
    [
        "Player Performance Analytics",
        "Decision 1 — Bowler Selection",
        "Decision 2 — Batsman Selection",
        "Decision 3 — Bowler Phase Management",
        "Decision 4 — Player Weakness Identification",
        "Decision 5 — Batsman Phase Management"
    ],
    key="nav_select"
)

# Sync state
if "Decision" in nav_choice:
    st.session_state.page = "strategy"
    decision = nav_choice
else:
    st.session_state.page = "analytics"
    decision = None

st.sidebar.markdown("---")
st.sidebar.markdown("### **Tournament Leaderboard**")

st.sidebar.markdown("**Top Batsman Leaderboard**")
top_run_getters = df.groupby("batter")["runs_off_bat"].sum().sort_values(ascending=False).head(5)
st.sidebar.dataframe(top_run_getters.reset_index().rename(columns={"batter": "Batsman", "runs_off_bat": "Runs"}), hide_index=True)

st.sidebar.markdown("**Top Bowler Leaderboard**")
bowler_agg = df.groupby("bowler").agg(
    total_runs_conceded=("runs_off_bat", "sum"),
    total_balls_bowled=("runs_off_bat", "count"),
    total_wickets=("wicket_type", lambda x: (x != '0').sum())
)
valid_bowlers = bowler_agg[bowler_agg["total_balls_bowled"] >= 60].copy()
valid_bowlers["economy"] = valid_bowlers["total_runs_conceded"] / (valid_bowlers["total_balls_bowled"] / 6)
best_bowlers = valid_bowlers.sort_values(by="total_wickets", ascending=False).head(5)

# --- 2. Advanced Player Metadata ---
PLAYER_METADATA = {
    # Batsmen Handedness
    "V Kohli": "RHB", "RG Sharma": "RHB", "MS Dhoni": "RHB", "S Dhawan": "LHB", "DA Warner": "LHB",
    "KL Rahul": "RHB", "SK Raina": "LHB", "AB de Villiers": "RHB", "CH Gayle": "LHB", "RV Uthappa": "RHB",
    "R Ravindra": "LHB", "S Dubey": "LHB", "RA Jadeja": "LHB", "Ishan Kishan": "LHB", "Q de Kock": "LHB",
    "YBK Jaiswal": "LHB", "TM Head": "LHB", "Abhishek Sharma": "LHB", "N Pooran": "LHB", "SO Hetmyer": "LHB",
    "KK Nair": "RHB", "SA Yadav": "RHB", "JC Buttler": "RHB", "SV Samson": "RHB", "RR Pant": "LHB",
    # Bowler Types (Handedness + Style)
    "JJ Bumrah": "RA Fast", "Mohammed Siraj": "RA Fast", "B Kumar": "RA Medium", "TA Boult": "LA Fast-Medium",
    "YS Chahal": "Leg-break", "Kuldeep Yadav": "LA Chinaman", "R Ashwin": "Off-break", "SP Narine": "Off-break",
    "Rashid Khan": "Leg-break", "M Pathirana": "RA Fast", "Mustafizur Rahman": "LA Fast-Medium", "Harshit Rana": "RA Fast",
}

def get_player_attribute(player, attr_type="bat"):
    """Infers player attributes or returns from metadata."""
    if player in PLAYER_METADATA:
        return PLAYER_METADATA[player]
    
    # Simple inference logic
    spin_keywords = ["chahal", "kuldeep", "ashwin", "jadeja", "tahir", "imran", "narine", "rashid", "piyush", "bishnoi", "varun", "krunal", "axar", "swapnil", "dagar"]
    if any(k in player.lower() for k in spin_keywords):
        return "Spin"
    return "Pace"

# --- Strategic Helper Functions ---

@st.cache_data
def get_bowler_selection_data(df, batsman, phase, remaining_overs=None):
    """Calculates best bowlers vs a specific batsman in a given phase with context."""
    filtered = df[(df["batter"] == batsman) & (df["phase"] == phase)]
    if filtered.empty:
        return None
    
    bowler_stats = filtered.groupby("bowler").agg(
        runs=("runs_off_bat", "sum"),
        balls=("runs_off_bat", "count"),
        wickets=("wicket_type", lambda x: (x != '0').sum()),
        dot_balls=("runs_off_bat", lambda x: (x == 0).sum())
    ).reset_index()
    
    # Minimum sample size for ranking
    bowler_stats = bowler_stats[bowler_stats["balls"] >= 6]
    
    if bowler_stats.empty:
        return None
        
    bowler_stats["economy"] = (bowler_stats["runs"] / (bowler_stats["balls"] / 6))
    bowler_stats["dot_pct"] = (bowler_stats["dot_balls"] / bowler_stats["balls"]) * 100
    
    # Advanced Scoring: Emphasis on Wickets in Death, Economy in Powerplay
    if phase == "Powerplay":
        bowler_stats["score"] = (-bowler_stats["economy"] * 1.5 + bowler_stats["wickets"] * 2 + bowler_stats["dot_pct"] * 0.1)
    elif phase == "Death":
        bowler_stats["score"] = (-bowler_stats["economy"] + bowler_stats["wickets"] * 5 + bowler_stats["dot_pct"] * 0.05)
    else:
        bowler_stats["score"] = (-bowler_stats["economy"] + bowler_stats["wickets"] * 3 + bowler_stats["dot_pct"] * 0.1)

    # Adjust for remaining overs if provided
    if remaining_overs:
        bowler_stats["rem_ov"] = bowler_stats["bowler"].map(lambda x: remaining_overs.get(x, 0))
        # Prioritize bowlers who still have overs left
        bowler_stats["score"] = bowler_stats.apply(lambda row: row["score"] if row["rem_ov"] > 0 else row["score"] - 50, axis=1)

    return bowler_stats.sort_values("score", ascending=False)

@st.cache_data
def get_batsman_selection_data(df, phase, bowler="Any", bowl_type="All", rrr=None):
    """Suggests best batsmen for a phase and bowling type with context."""
    filtered = df[df["phase"] == phase].copy()
    if bowler != "Any":
        filtered = filtered[filtered["bowler"] == bowler]

    spin_keywords = ["chahal", "kuldeep", "ashwin", "jadeja", "tahir", "imran", "narine", "rashid", "piyush", "bishnoi", "varun"]
    if bowl_type == "Spin":
        filtered = filtered[filtered["bowler"].str.lower().str.contains('|'.join(spin_keywords), na=False)]
    elif bowl_type == "Pace":
        filtered = filtered[~filtered["bowler"].str.lower().str.contains('|'.join(spin_keywords), na=False)]

    if filtered.empty:
        return None, None

    bat_stats = filtered.groupby("batter").agg(
        runs=("runs_off_bat", "sum"), 
        balls=("runs_off_bat", "count"), 
        boundaries=("runs_off_bat", lambda x: ((x == 4) | (x == 6)).sum()), 
        wickets=("wicket_type", lambda x: (x != '0').sum())
    ).reset_index()
    
    bat_stats = bat_stats[bat_stats["balls"] >= 10]
    if bat_stats.empty:
        return None, None

    bat_stats["strike_rate"] = (bat_stats["runs"] / bat_stats["balls"]) * 100
    bat_stats["boundary_pct"] = (bat_stats["boundaries"] / bat_stats["balls"]) * 100
    bat_stats["score"] = bat_stats["strike_rate"] + bat_stats["boundary_pct"] - (bat_stats["wickets"] * 5)
    
    top_players = bat_stats.sort_values("score", ascending=False)
    avoid_players = bat_stats.sort_values("score", ascending=True)
    return top_players, avoid_players

@st.cache_data
def get_bowler_phase_stats(df, bowler):
    """Analyzes bowler performance across phases."""
    filtered_bowl = df[df["bowler"] == bowler]
    if filtered_bowl.empty:
        return None
    
    phase_stats = filtered_bowl.groupby("phase").agg(
        runs=("runs_off_bat", "sum"), 
        balls=("runs_off_bat", "count"), 
        wickets=("wicket_type", lambda x: (x != '0').sum()), 
        dot_balls=("runs_off_bat", lambda x: (x == 0).sum())
    ).reset_index()
    
    if phase_stats.empty:
        return None
        
    phase_stats["economy"] = phase_stats["runs"] / (phase_stats["balls"] / 6)
    phase_stats["dot_pct"] = (phase_stats["dot_balls"] / phase_stats["balls"]) * 100
    return phase_stats

@st.cache_data
def get_batsman_phase_stats(df, batsman):
    """Analyzes batsman performance across phases."""
    filtered_bat = df[df["batter"] == batsman]
    if filtered_bat.empty:
        return None
        
    phase_stats = filtered_bat.groupby("phase").agg(
        runs=("runs_off_bat", "sum"), 
        balls=("runs_off_bat", "count"), 
        boundaries=("runs_off_bat", lambda x: ((x == 4) | (x == 6)).sum())
    ).reset_index()
    
    if phase_stats.empty:
        return None
        
    phase_stats["strike_rate"] = (phase_stats["runs"] / phase_stats["balls"]) * 100
    return phase_stats

@st.cache_data
def analyze_player_context(df, player, player_type, opponent_team=None):
    """Deep analysis of player vs specific team/types."""
    if player_type == "Batsman":
        filtered = df[df["batter"] == player]
        
        # Split by opponent team if provided
        if opponent_team:
            opp_squad = TEAM_SQUADS.get(opponent_team, [])
            opp_df = filtered[filtered["bowler"].isin(opp_squad)]
            if not opp_df.empty:
                filtered = opp_df

        spin_df = filtered[filtered["bowler"].apply(lambda x: "Spin" in get_player_attribute(x, "bowl"))]
        pace_df = filtered[~filtered["bowler"].apply(lambda x: "Spin" in get_player_attribute(x, "bowl"))]

        def compute_stats(data):
            if len(data) == 0: return {"SR": 0, "Avg": 0, "Dot%": 0, "Bnd%": 0}
            runs, balls, wickets = data["runs_off_bat"].sum(), len(data), (data["wicket_type"] != '0').sum()
            dots = (data["runs_off_bat"] == 0).sum()
            boundaries = ((data["runs_off_bat"] == 4) | (data["runs_off_bat"] == 6)).sum()
            return {
                "SR": (runs / balls * 100) if balls > 0 else 0,
                "Avg": runs / max(wickets, 1),
                "Dot%": (dots / balls * 100) if balls > 0 else 0,
                "Bnd%": (boundaries / balls * 100) if balls > 0 else 0
            }

        return pd.DataFrame([compute_stats(spin_df), compute_stats(pace_df)], index=["vs Spin", "vs Pace"])
    
    else:
        # Bowler analysis vs RHB/LHB
        filtered = df[df["bowler"] == player]
        lhb_list = [p for p, hand in PLAYER_METADATA.items() if hand == "LHB"]
        
        lhb_df = filtered[filtered["batter"].isin(lhb_list)]
        rhb_df = filtered[~filtered["batter"].isin(lhb_list)]
        
        def compute_bowl_stats(data):
            if len(data) == 0: return {"Eco": 0, "Wkts": 0, "Dot%": 0}
            runs, balls, wickets = data["runs_off_bat"].sum(), len(data), (data["wicket_type"] != '0').sum()
            dots = (data["runs_off_bat"] == 0).sum()
            return {
                "Eco": (runs / (balls / 6)) if balls > 0 else 0,
                "Wkts": wickets,
                "Dot%": (dots / balls * 100) if balls > 0 else 0
            }
            
        return pd.DataFrame([compute_bowl_stats(rhb_df), compute_bowl_stats(lhb_df)], index=["vs RHB", "vs LHB"])

@st.cache_data
def get_player_deployment_strategy(df, player, player_role):
    """Analyzes best deployment for a player."""
    if player_role == "Bowler":
        stats = get_bowler_phase_stats(df, player)
        if stats is not None:
            best_phase = stats.loc[stats["economy"].idxmin()]["phase"]
            # Best/Worst Matchups
            matchups = df[df["bowler"] == player].groupby("batter").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "count"), wickets=("wicket_type", lambda x: (x != '0').sum())).reset_index()
            matchups = matchups[matchups["balls"] >= 12]
            strong = matchups.sort_values(["wickets", "runs"], ascending=[False, True]).head(3)
            weak = matchups.sort_values(["runs", "wickets"], ascending=[False, True]).head(3)
            return {"best_phase": best_phase, "strong": strong, "weak": weak}
    else:
        stats = get_batsman_phase_stats(df, player)
        if stats is not None:
            best_phase = stats.loc[stats["strike_rate"].idxmax()]["phase"]
            matchups = df[df["batter"] == player].groupby("bowler").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "count"), wickets=("wicket_type", lambda x: (x != '0').sum())).reset_index()
            matchups = matchups[matchups["balls"] >= 12]
            strong = matchups.sort_values(["runs", "wickets"], ascending=[False, True]).head(3)
            weak = matchups.sort_values(["wickets", "runs"], ascending=[False, True]).head(3)
            return {"best_phase": best_phase, "strong": strong, "weak": weak}
    return None

# --- IPL Team Squads (Updated for 2026 Context) ---
TEAM_SQUADS = {
    "Royal Challengers Bengaluru": ["V Kohli", "F du Plessis", "GJ Maxwell", "RM Patidar", "KD Karthik", "Mohammed Siraj", "Yash Dayal", "KV Sharma", "C Green", "WG Jacks", "Swapnil Singh", "LH Ferguson"],
    "Chennai Super Kings": ["RD Gaikwad", "R Ravindra", "AM Rahane", "DJ Mitchell", "S Dubey", "MS Dhoni", "RA Jadeja", "M Pathirana", "T Deshpande", "DL Chahar", "Mustafizur Rahman", "SN Thakur"],
    "Mumbai Indians": ["RG Sharma", "Ishan Kishan", "N Tilak Varma", "HH Pandya", "JJ Bumrah", "G Coetzee", "Piyush Chawla", "TH David", "Mohammad Nabi", "NU Thushara", "L Wood", "Akash Madhwal"],
    "Kolkata Knight Riders": ["PD Salt", "SP Narine", "A Raghuvanshi", "SS Iyer", "VR Iyer", "RK Singh", "AD Russell", "MA Starc", "Harshit Rana", "CV Varun", "Vaibhav Arora", "Ramandeep Singh"],
    "Rajasthan Royals": ["YBK Jaiswal", "JC Buttler", "SV Samson", "R Parag", "DH Jurel", "SO Hetmyer", "R Ashwin", "TA Boult", "Avesh Khan", "YS Chahal", "Sandeep Sharma", "K Maharaj"],
    "Sunrisers Hyderabad": ["TM Head", "Abhishek Sharma", "AK Markram", "H Klaasen", "Nithish Kumar Reddy", "Abdul Samad", "Shahbaz Ahmed", "PJ Cummins", "B Kumar", "JD Unadkat", "T Natarajan", "Mayank Agarwal"],
    "Delhi Capitals": ["Prithvi Shaw", "JF Fraser-McGurk", "Abishek Porel", "Shai Hope", "RR Pant", "T Stubbs", "AR Patel", "Kuldeep Yadav", "Mukesh Kumar", "I Sharma", "KK Ahmed", "Anrich Nortje"],
    "Lucknow Super Giants": ["KL Rahul", "Q de Kock", "MP Stoinis", "DJ Hooda", "N Pooran", "AY Badoni", "KH Pandya", "Ravi Bishnoi", "Mohsin Khan", "Yash Thakur", "Naveen-ul-Haq", "Mayank Yadav"],
    "Gujarat Titans": ["Shubman Gill", "WP Saha", "B Sai Sudharsan", "DA Miller", "Azmatullah Omarzai", "R Tewatia", "Rashid Khan", "R Sai Kishore", "Umesh Yadav", "Sandeep Warrier", "Noor Ahmad", "Mohit Sharma"],
    "Punjab Kings": ["JM Bairstow", "P Simran Singh", "RR Rossouw", "SM Curran", "JM Sharma", "Shashank Singh", "Ashutosh Sharma", "Harpreet Brar", "HV Patel", "K Rabada", "Arshdeep Singh", "Rahul Chahar"]
}

# --- Modes ---

def team_builder_mode(df):
    """Contains the existing Global Strategy and Analytics logic."""
    # Data Validation for Analytics View
    if st.session_state.page == "analytics":
        if player_data.empty:
            st.warning(f"No data available for {selected_player}.")
            return

    # ==========================================
    # MAIN CONTENT VIEWS
    # ==========================================
    if st.session_state.page == "analytics":
        st.title("Player Performance Analytics")
        # BATSMAN VIEW
        if player_type == "Batsman":
            st.header(f"Batting Performance: {selected_player}")

            col1, col2, col3, col4 = st.columns(4)
            total_runs = player_data["runs_off_bat"].sum()
            total_balls = len(player_data)
            strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0

            col1.metric("Total Runs", int(total_runs))
            col2.metric("Balls Faced", int(total_balls))
            col3.metric("Strike Rate", f"{strike_rate:.2f}")

            # Phase Analysis
            st.subheader("Runs by Match Phase")
            runs_by_phase = player_data.groupby("phase")["runs_off_bat"].sum().reset_index()
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.barplot(x="phase", y="runs_off_bat", data=runs_by_phase, ax=ax1, palette="viridis", order=["Powerplay", "Middle", "Death"])
            ax1.set_xlabel("Match Phase")
            ax1.set_ylabel("Total Runs")
            ax1.set_title(f"Runs Scored by {selected_player} in Each Phase")
            st.pyplot(fig1)

            # Batsmen vs Favorite Bowler
            st.subheader("Batsmen vs Favorite Bowler")
            st.markdown("Bowlers who have conceded the most runs per over against this batsman (top 5).")
            bowler_matchups = player_data.groupby("bowler").agg(
                runs=("runs_off_bat", "sum"),
                balls=("runs_off_bat", "count"),
                wickets=("wicket_type", lambda x: (x != '0').sum())
            ).reset_index()

            # Update logic for "Favorite Bowlers" -> most runs per over (economy against them)
            fav_matchups = bowler_matchups[bowler_matchups["balls"] >= 6].copy()
            fav_matchups["economy_against"] = (fav_matchups["runs"] / (fav_matchups["balls"] / 6))
            fav_top = fav_matchups.sort_values(by="economy_against", ascending=False).head(5)

            if not fav_top.empty:
                st.dataframe(fav_top[["bowler", "runs", "balls", "economy_against"]].rename(
                    columns={"bowler": "Bowler", "runs": "Runs", "balls": "Balls", "economy_against": "Runs Per Over"}
                ).style.format({"Runs Per Over": "{:.2f}"}), use_container_width=True)
            else:
                st.info("Not enough data to show favorite bowlers.")

            # Batsman struggled by
            st.subheader("Batsman struggled by")
            st.markdown("Bowlers who have dismissed this batsman or kept things tight.")
            struggle_matchups = bowler_matchups.sort_values(by=["wickets", "runs"], ascending=[False, True]).head(5)
            if not struggle_matchups.empty:
                st.dataframe(struggle_matchups.rename(
                    columns={"bowler": "Bowler", "runs": "Runs", "balls": "Balls", "wickets": "Times Out"}
                ), use_container_width=True)
            else:
                st.info("No struggle data found.")


        # BOWLER VIEW
        else:
            st.header(f"Bowling Performance: {selected_player}")

            col1, col2, col3, col4 = st.columns(4)
            total_runs_conceded = player_data["runs_off_bat"].sum()
            total_balls_bowled = len(player_data)
            total_wickets = (player_data["wicket_type"] != '0').sum()
            economy_rate = (total_runs_conceded / (total_balls_bowled / 6)) if total_balls_bowled > 0 else 0

            col1.metric("Total Wickets", int(total_wickets))
            col2.metric("Balls Bowled", int(total_balls_bowled))
            col3.metric("Runs Conceded", int(total_runs_conceded))
            col4.metric("Economy Rate", f"{economy_rate:.2f}")

            # Phase Analysis (Bowler)
            st.subheader("Bowling Economy & Wickets by Match Phase")

            phase_stats = player_data.groupby("phase").agg(
                runs=("runs_off_bat", "sum"),
                balls=("runs_off_bat", "count"),
                wickets=("wicket_type", lambda x: (x != '0').sum())
            ).reset_index()

            if not phase_stats.empty:
                phase_stats["economy"] = phase_stats["runs"] / (phase_stats["balls"] / 6)

                fig_bowl, ax_bowl = plt.subplots(1, 2, figsize=(12, 4))
                sns.barplot(x="phase", y="economy", data=phase_stats, ax=ax_bowl[0], palette="coolwarm_r", order=["Powerplay", "Middle", "Death"])
                ax_bowl[0].set_title(f"Economy Rate by Phase — {selected_player}")
                ax_bowl[0].set_ylabel("Economy Rate")

                sns.barplot(x="phase", y="wickets", data=phase_stats, ax=ax_bowl[1], palette="magma", order=["Powerplay", "Middle", "Death"])
                ax_bowl[1].set_title(f"Wickets by Phase — {selected_player}")
                ax_bowl[1].set_ylabel("Total Wickets")
                st.pyplot(fig_bowl)

            # Bowler vs Favorite Batsmen (Top 5 by Wickets)
            st.subheader("Bowler vs Favorite Batsmen")
            batsman_matchups = player_data.groupby("batter").agg(
                runs_conceded=("runs_off_bat", "sum"),
                balls_bowled=("runs_off_bat", "count"),
                dismissals=("wicket_type", lambda x: (x != '0').sum())
            ).reset_index()

            top_bat_matchups = batsman_matchups.sort_values(by="dismissals", ascending=False).head(5)
            if not top_bat_matchups.empty:
                top_bat_matchups["batsman_strike_rate"] = (top_bat_matchups["runs_conceded"] / top_bat_matchups["balls_bowled"]) * 100
                st.dataframe(top_bat_matchups.style.format({"batsman_strike_rate": "{:.2f}"}), use_container_width=True)
            else:
                st.info("Not enough data to show significant matchups.")

            # Dominated by Batsman (Top 5 by Runs)
            st.subheader("Dominated by Batsman")
            top_runs_matchups = batsman_matchups.sort_values(by="runs_conceded", ascending=False).head(5)
            if not top_runs_matchups.empty:
                top_runs_matchups["batsman_strike_rate"] = (top_runs_matchups["runs_conceded"] / top_runs_matchups["balls_bowled"]) * 100
                st.dataframe(top_runs_matchups.style.format({"batsman_strike_rate": "{:.2f}"}), use_container_width=True)
            else:
                st.info("Not enough data to show significant matchups.")


    # ==========================================
    # STRATEGIC DECISION ENGINE VIEW
    # ==========================================
    elif st.session_state.page == "strategy":
        st.title("Strategic Decision Engine")
        st.markdown("Data-driven answers to the key tactical questions teams face during match planning.")

        if decision == "Decision 1 — Bowler Selection":
            st.markdown("### Which bowler should bowl against a specific batsman in a given phase?")
            col1, col2 = st.columns(2)
            with col1: d1_batsman = st.selectbox("Select Batsman", sorted(df['batter'].dropna().unique()), key="d1_bat")
            with col2: d1_phase = st.selectbox("Select Match Phase", ["Powerplay", "Middle", "Death"], key="d1_phase")

            if st.button("Find Best Bowler", key="d1_btn"):
                top_25, avoid_25 = get_bowler_selection_data(df, d1_batsman, d1_phase)
                if top_25 is None:
                    st.warning("Not enough head-to-head deliveries to rank bowlers.")
                else:
                    st.success(f"### **Top Suggested Bowlers vs {d1_batsman} ({d1_phase})**")
                    st.dataframe(top_25.head(10)[["bowler", "economy", "wickets", "dot_pct"]].rename(columns={"bowler": "Bowler", "economy": "Economy", "wickets": "Wickets", "dot_pct": "Dot Ball %"}).style.format({"Economy": "{:.2f}", "Dot Ball %": "{:.1f}"}), use_container_width=True)
                    
                    fig_d1, ax_d1 = plt.subplots(figsize=(10, 5))
                    sns.barplot(x="bowler", y="economy", data=top_25.head(10), ax=ax_d1, palette="YlOrRd")
                    ax_d1.set_title(f"Economy of Top 10 Suggested Bowlers vs {d1_batsman}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_d1)

        elif decision == "Decision 2 — Batsman Selection":
            st.markdown("### Which batsman to recommend based on specific bowler, bowling type, and match phase?")
            col1, col2, col3 = st.columns(3)
            with col1: d2_bowler = st.selectbox("Opponent Bowler (Optional)", ["Any"] + players_list_bowl, key="d2_bowler")
            with col2: d2_bowl_type = st.selectbox("Opposition Bowling Type", ["All", "Spin", "Pace"], key="d2_type")
            with col3: d2_phase = st.selectbox("Match Phase", ["Powerplay", "Middle", "Death"], key="d2_phase")

            if st.button("Suggest Batsmen", key="d2_btn"):
                top_25, avoid_20 = get_batsman_selection_data(df, d2_phase, d2_bowler, d2_bowl_type)
                if top_25 is None:
                    st.warning("Not enough data for this specific combination.")
                else:
                    st.success(f"### **Top Suggested Batsmen for {d2_phase}**")
                    st.dataframe(top_25.head(10)[["batter", "runs", "balls", "strike_rate", "boundary_pct"]].rename(columns={"batter": "Batsman", "runs": "Runs", "balls": "Balls", "strike_rate": "Strike Rate", "boundary_pct": "Boundary %"}).style.format({"Strike Rate": "{:.2f}", "Boundary %": "{:.1f}"}), use_container_width=True)

        elif decision == "Decision 3 — Bowler Phase Management":
            st.markdown("### How should a bowler be deployed across match phases?")
            d3_bowler = st.selectbox("Select Bowler", sorted(df['bowler'].dropna().unique()), key="d3_bowl")
            phase_stats_bowl = get_bowler_phase_stats(df, d3_bowler)
            if phase_stats_bowl is not None:
                best_phase_row_bowl = phase_stats_bowl.loc[phase_stats_bowl["economy"].idxmin()]
                st.success(f"**Bowler Recommendation: Deploy {d3_bowler} in {best_phase_row_bowl['phase']}**")
                fig_d3_bowl, axes_bowl = plt.subplots(1, 3, figsize=(14, 4))
                phase_order = ["Powerplay", "Middle", "Death"]
                for ax, col, title, color in zip(axes_bowl, ["economy", "wickets", "dot_pct"], ["Economy Rate", "Wickets", "Dot Ball %"], ["coolwarm_r", "magma", "Blues_d"]):
                    data_plot = phase_stats_bowl.set_index("phase").reindex(phase_order).fillna(0).reset_index()
                    sns.barplot(x="phase", y=col, data=data_plot, ax=ax, palette=color)
                    ax.set_title(f"Bowler {title}")
                st.pyplot(fig_d3_bowl)

        elif decision == "Decision 5 — Batsman Phase Management":
            st.markdown("### How should a batsman be deployed across match phases?")
            d5_batsman = st.selectbox("Select Batsman", players_list_bat, key="d5_bat_select")
            phase_stats_bat = get_batsman_phase_stats(df, d5_batsman)
            if phase_stats_bat is not None:
                best_phase_row_bat = phase_stats_bat.loc[phase_stats_bat["strike_rate"].idxmax()]
                st.success(f"**Batsman Recommendation: Deploy {d5_batsman} in {best_phase_row_bat['phase']}**")
                fig_d5_bat, ax_bat = plt.subplots(figsize=(8, 4))
                phase_order = ["Powerplay", "Middle", "Death"]
                sns.barplot(x="phase", y="strike_rate", data=phase_stats_bat.set_index("phase").reindex(phase_order).fillna(0).reset_index(), ax=ax_bat, palette="viridis")
                ax_bat.set_title(f"Strike Rate by Phase")
                st.pyplot(fig_d5_bat)

        elif decision == "Decision 4 — Player Weakness Identification":
            st.markdown("### Identify a player's weakness.")
            d4_player_type = st.radio("Analyse Player Type", ["Batsman", "Bowler"], horizontal=True, key="d4_type_radio")
            if d4_player_type == "Batsman":
                d4_player = st.selectbox("Select Batsman", sorted(df['batter'].dropna().unique()), key="d4_bat")
                if st.button("Identify Weaknesses", key="d4_btn"):
                    comparison = compute_player_weakness(df, d4_player, "Batsman")
                    st.dataframe(comparison.style.format("{:.2f}"), use_container_width=True)
            else:
                d4_player = st.selectbox("Select Bowler", sorted(df['bowler'].dropna().unique()), key="d4_bowl")
                if st.button("Identify Strengths & Weaknesses", key="d4_bowl_btn"):
                    batsman_stats = compute_player_weakness(df, d4_player, "Bowler")
                    st.dataframe(batsman_stats.head(10), use_container_width=True)

def match_decision_mode(df):
    """Upgraded Match Decision Mode with full team context and rankings."""
    st.title("🏹 Captain's Match Decision Engine")
    st.markdown("Real-time tactical suggestions for match-winning decisions.")

    # 1. Team Context Selection
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        my_team = st.selectbox("My Team (Your Squad)", list(TEAM_SQUADS.keys()), key="md_my_team")
        my_squad = TEAM_SQUADS[my_team]
    with col_t2:
        opp_team = st.selectbox("Opponent Team", [t for t in TEAM_SQUADS.keys() if t != my_team], key="md_opp_team")
        opp_squad = TEAM_SQUADS[opp_team]

    # 0. Performance Optimization: Pre-filter main DF for match-relevant data
    match_relevant_players = set(my_squad) | set(opp_squad)
    df_match = df[df["batter"].isin(match_relevant_players) | df["bowler"].isin(match_relevant_players)].copy()

    st.divider()

    # 1. Decision Selection & Match Context
    col_inp, col_res = st.columns([1, 2])
    
    with col_inp:
        st.subheader("Match Scenario")
        decision_type = st.radio(
            "Decision Engine Segment",
            ["Bowling Decision", "Batting Decision", "Player Context Analysis", "Deployment Strategy"],
            key="md_dec_area"
        )
        
        phase = st.selectbox("Current Match Phase", ["Powerplay", "Middle", "Death"], key="md_phase")
        
        if decision_type == "Bowling Decision":
            st.info("Suggesting optimal bowler from your XI for the next over.")
            opp_bat = st.selectbox("Current Opponent Batsman", sorted(list(set(df_match['batter'].dropna().unique()) & set(opp_squad))), key="md_opp_bat")
            
            st.write("**Bowler Quotas (Overs Left)**")
            # Filter squad players who have ever bowled in our dataset
            potential_bowlers = sorted(list(set(my_squad) & set(df_match['bowler'].dropna().unique())))
            rem_overs = {}
            for p in potential_bowlers:
                rem_overs[p] = st.slider(f"{p}", 0, 4, 4, key=f"md_rem_{p}")
                     
        elif decision_type == "Batting Decision":
            st.info("Suggesting batting order/next player based on matchups.")
            opp_bowl = st.selectbox("Current Opponent Bowler", sorted(list(set(df_match['bowler'].dropna().unique()) & set(opp_squad))), key="md_opp_bowl")
            wickets = st.number_input("Wickets Fallen", 0, 9, 0, key="md_wickets")
            rrr = st.number_input("Required Run Rate", 0.0, 36.0, 8.0, key="md_rrr")
            
        elif decision_type == "Player Context Analysis":
            combined_squads = sorted(list(set(my_squad) | set(opp_squad)))
            target_player = st.selectbox("Select Player for Detailed Intel", combined_squads, key="md_ana_p")

    with col_res:
        st.subheader("Strategic Suggestions")
        
        if decision_type == "Bowling Decision":
            # Ranked list vs Opponent Batsman
            ranked = get_bowler_selection_data(df, opp_bat, phase, remaining_overs=rem_overs)
            if ranked is not None:
                # Filter by My Squad
                ranked_squad = ranked[ranked["bowler"].isin(my_squad)]
                if not ranked_squad.empty:
                    st.success(f"**Full Ranked Bowlers vs {opp_bat} in {phase}**")
                    for i, row in enumerate(ranked_squad.itertuples()):
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            c1.markdown(f"#### {i+1}. {row.bowler}")
                            c2.metric("Score", f"{row.score:.1f}")
                            
                            st.write(f"**Matchup Data:** Eco: {row.economy:.2f}, Wkts: {int(row.wickets)}, Dots: {row.dot_pct:.1f}%")
                            
                            # Reasoning
                            if row.score > 60:
                                st.markdown("✅ **High Fit:** Strong historical control over this batsman.")
                            elif row.score < 30:
                                st.markdown("⚠️ **Caution:** Often targeted by this batsman in this phase.")
                            else:
                                st.markdown("ℹ️ **Neutral:** Standard performance metrics.")
                else:
                    st.warning(f"No head-to-head data for {my_team} bowlers vs {opp_bat} in {phase}.")
            else:
                st.warning(f"No historical data found for {opp_bat} in {phase}.")

        elif decision_type == "Batting Decision":
            ranked = get_batsman_selection_data(df, phase, bowler=opp_bowl, rrr=rrr)
            if ranked is not None:
                ranked_squad = ranked[ranked["batter"].isin(my_squad)]
                if not ranked_squad.empty:
                    st.success(f"**Full Ranked Batsmen vs {opp_bowl} in {phase}**")
                    for i, row in enumerate(ranked_squad.itertuples()):
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            c1.markdown(f"#### {i+1}. {row.batter}")
                            c2.metric("SR Fit", f"{row.strike_rate:.1f}")
                            
                            st.write(f"**Phase Performance:** SR: {row.strike_rate:.2f}, Avg: {row.average:.1f}")
                            
                            if rrr > 10 and row.strike_rate > 150:
                                st.markdown("🚀 **Aggressor:** Best for quick runs to maintain RRR.")
                            elif rrr <= 8 and row.average > 30:
                                st.markdown("🧱 **Anchor:** Reliable option to finish the game.")
                else:
                    st.warning(f"No data for {my_team} batsmen vs {opp_bowl} in {phase}.")
            else:
                st.warning(f"No historical records for {opp_bowl} in this phase.")

        elif decision_type == "Player Context Analysis":
            role = "Bowler" if target_player in players_list_bowl else "Batsman"
            st.write(f"### Intel for {target_player} ({role})")
            
            # Context-based analysis
            analysis = analyze_player_context(df_match, target_player, role, opponent_team=opp_team if target_player in my_squad else my_team)
            st.write("**Performance Splits**")
            st.dataframe(analysis.style.format("{:.2f}"), use_container_width=True)
            
            # Deployment Intel
            strat = get_player_deployment_strategy(df_match, target_player, role)
            if strat:
                st.info(f"**Optimal Operating Phase:** {strat['best_phase']}")
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown("**Strong Matchups**")
                    label = "batter" if role == "Bowler" else "bowler"
                    st.dataframe(strat["strong"][[label, "runs"]].rename(columns={label: "Player", "runs": "Runs"}), hide_index=True)
                with s_col2:
                    st.markdown("**Weak Matchups**")
                    st.dataframe(strat["weak"][[label, "runs"]].rename(columns={label: "Player", "runs": "Runs"}), hide_index=True)

        elif decision_type == "Deployment Strategy":
            st.success(f"**Operational Plan for {my_team}**")
            for player in my_squad:
                role = "Bowler" if player in players_list_bowl else "Batsman"
                strat = get_player_deployment_strategy(df, player, role)
                if strat:
                    with st.expander(f"{player} — {strat['best_phase']} Expert"):
                        st.write(f"Use primarily in **{strat['best_phase']}**.")
                        if role == "Bowler":
                            st.write(f"Top Targets: {', '.join(strat['strong']['batter'].tolist())[:30]}...")
                        else:
                            st.write(f"Best against: {', '.join(strat['strong']['bowler'].tolist())[:30]}...")

# --- Mode Switcher ---
app_mode = st.sidebar.radio("App Mode", ["Team Builder Mode", "Match Decision Mode"], key="app_mode_select")

if app_mode == "Team Builder Mode":
    team_builder_mode(df)
else:
    match_decision_mode(df)
