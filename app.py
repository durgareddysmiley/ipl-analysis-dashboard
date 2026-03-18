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

# --- Data Loading ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_ipl_data.csv")

# Load data
with st.spinner("Loading Consolidated IPL Data..."):
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
st.sidebar.dataframe(best_bowlers.reset_index()[["bowler", "total_wickets"]].rename(columns={"bowler": "Bowler", "total_wickets": "Wickets"}), hide_index=True)

# Data Validation for Analytics View
if st.session_state.page == "analytics":
    if player_data.empty:
        st.warning(f"No data available for {selected_player}.")
        st.stop()

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
        top_bat_matchups["batsman_strike_rate"] = (top_bat_matchups["runs_conceded"] / top_bat_matchups["balls_bowled"]) * 100

        if not top_bat_matchups.empty:
            st.dataframe(top_bat_matchups.style.format({"batsman_strike_rate": "{:.2f}"}), use_container_width=True)
        else:
            st.info("Not enough data to show significant matchups.")

        # Dominated by Batsman (Top 5 by Runs)
        st.subheader("Dominated by Batsman")
        top_runs_matchups = batsman_matchups.sort_values(by="runs_conceded", ascending=False).head(5)
        top_runs_matchups["batsman_strike_rate"] = (top_runs_matchups["runs_conceded"] / top_runs_matchups["balls_bowled"]) * 100

        if not top_runs_matchups.empty:
            st.dataframe(top_runs_matchups.style.format({"batsman_strike_rate": "{:.2f}"}), use_container_width=True)
        else:
            st.info("Not enough data to show significant matchups.")


# ==========================================
# STRATEGIC DECISION ENGINE VIEW
# ==========================================
elif st.session_state.page == "strategy":
    st.title("Strategic Decision Engine")
    st.markdown("Data-driven answers to the key tactical questions teams face during match planning.")

    # Nest all decision logic inside here to prevent NameError and redundant views
    # -----------------------------------------------
    # Decision 1 — Bowler Selection
    # -----------------------------------------------
    if decision == "Decision 1 — Bowler Selection":
        st.markdown("### Which bowler should bowl against a specific batsman in a given phase?")

        col1, col2 = st.columns(2)
        with col1:
            d1_batsman = st.selectbox("Select Batsman", sorted(df['batter'].dropna().unique()), key="d1_bat")
        with col2:
            d1_phase = st.selectbox("Select Match Phase", ["Powerplay", "Middle", "Death"], key="d1_phase")

        if st.button("Find Best Bowler", key="d1_btn"):
            filtered = df[(df["batter"] == d1_batsman) & (df["phase"] == d1_phase)]
            if filtered.empty:
                st.warning("Not enough data for this combination.")
            else:
                bowler_stats = filtered.groupby("bowler").agg(
                    runs=("runs_off_bat", "sum"),
                    balls=("runs_off_bat", "count"),
                    wickets=("wicket_type", lambda x: (x != '0').sum()),
                    dot_balls=("runs_off_bat", lambda x: (x == 0).sum())
                ).reset_index()
                bowler_stats = bowler_stats[bowler_stats["balls"] >= 6]

                if bowler_stats.empty:
                    st.warning("Not enough head-to-head deliveries to rank bowlers (need at least 1 over per bowler).")
                else:
                    bowler_stats["economy"] = (bowler_stats["runs"] / (bowler_stats["balls"] / 6))
                    bowler_stats["dot_pct"] = (bowler_stats["dot_balls"] / bowler_stats["balls"]) * 100
                    bowler_stats["score"] = (-bowler_stats["economy"] + bowler_stats["wickets"] * 3 + bowler_stats["dot_pct"] * 0.1)
                    
                    top_25 = bowler_stats.sort_values("score", ascending=False).head(25)
                    avoid_25 = bowler_stats.sort_values("score", ascending=True).head(25)

                    st.success(f"### **Top 25 Suggested Bowlers vs {d1_batsman} ({d1_phase})**")
                    st.dataframe(top_25[["bowler", "economy", "wickets", "dot_pct"]].rename(columns={"bowler": "Bowler", "economy": "Economy", "wickets": "Wickets", "dot_pct": "Dot Ball %"}).style.format({"Economy": "{:.2f}", "Dot Ball %": "{:.1f}"}), use_container_width=True)
                    
                    st.markdown("### ⚠️ **Bowlers to Avoid vs {d1_batsman}**")
                    st.dataframe(avoid_25[["bowler", "economy", "wickets", "dot_pct"]].rename(columns={"bowler": "Bowler", "economy": "Economy", "wickets": "Wickets", "dot_pct": "Dot Ball %"}).style.format({"Economy": "{:.2f}", "Dot Ball %": "{:.1f}"}), use_container_width=True)

                    fig_d1, ax_d1 = plt.subplots(figsize=(10, 5))
                    sns.barplot(x="bowler", y="economy", data=top_25.head(10), ax=ax_d1, palette="YlOrRd")
                    ax_d1.set_title(f"Economy of Top 10 Suggested Bowlers vs {d1_batsman} ({d1_phase})")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_d1)

    # -----------------------------------------------
    # Decision 2 — Batsman Selection
    # -----------------------------------------------
    elif decision == "Decision 2 — Batsman Selection":
        st.markdown("### Which batsman to recommend based on specific bowler, bowling type, and match phase?")

        col1, col2, col3 = st.columns(3)
        with col1:
            d2_bowler = st.selectbox("Opponent Bowler (Optional)", ["Any"] + players_list_bowl, key="d2_bowler")
        with col2:
            d2_bowl_type = st.selectbox("Opposition Bowling Type", ["All", "Spin", "Pace"], key="d2_type")
        with col3:
            d2_phase = st.selectbox("Match Phase", ["Powerplay", "Middle", "Death"], key="d2_phase")

        if st.button("Suggest Batsmen", key="d2_btn"):
            filtered = df[df["phase"] == d2_phase].copy()
            if d2_bowler != "Any":
                filtered = filtered[filtered["bowler"] == d2_bowler]

            spin_keywords = ["chahal", "kuldeep", "ashwin", "jadeja", "tahir", "imran", "narine", "rashid", "piyush", "bishnoi", "varun"]
            if d2_bowl_type == "Spin":
                filtered = filtered[filtered["bowler"].str.lower().str.contains('|'.join(spin_keywords), na=False)]
            elif d2_bowl_type == "Pace":
                filtered = filtered[~filtered["bowler"].str.lower().str.contains('|'.join(spin_keywords), na=False)]

            if filtered.empty:
                st.warning("Not enough data for this specific combination.")
            else:
                bat_stats = filtered.groupby("batter").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "count"), boundaries=("runs_off_bat", lambda x: ((x == 4) | (x == 6)).sum()), wickets=("wicket_type", lambda x: (x != '0').sum())).reset_index()
                bat_stats = bat_stats[bat_stats["balls"] >= 10]
                bat_stats["strike_rate"] = (bat_stats["runs"] / bat_stats["balls"]) * 100
                bat_stats["boundary_pct"] = (bat_stats["boundaries"] / bat_stats["balls"]) * 100
                bat_stats["score"] = bat_stats["strike_rate"] + bat_stats["boundary_pct"] - (bat_stats["wickets"] * 5)
                
                top_25 = bat_stats.sort_values("score", ascending=False).head(25)
                avoid_20 = bat_stats.sort_values("score", ascending=True).head(20)

                st.success(f"### **Top 25 Suggested Batsmen for {d2_phase} vs {d2_bowl_type}**")
                st.dataframe(top_25[["batter", "runs", "balls", "strike_rate", "boundary_pct"]].rename(columns={"batter": "Batsman", "runs": "Runs", "balls": "Balls", "strike_rate": "Strike Rate", "boundary_pct": "Boundary %"}).style.format({"Strike Rate": "{:.2f}", "Boundary %": "{:.1f}"}), use_container_width=True)
                
                st.markdown("### ⚠️ **Batsmen to Avoid for {d2_phase}**")
                st.dataframe(avoid_20[["batter", "runs", "balls", "strike_rate", "boundary_pct"]].rename(columns={"batter": "Batsman", "runs": "Runs", "balls": "Balls", "strike_rate": "Strike Rate", "boundary_pct": "Boundary %"}).style.format({"Strike Rate": "{:.2f}", "Boundary %": "{:.1f}"}), use_container_width=True)

                fig_d2, ax_d2 = plt.subplots(figsize=(10, 5))
                sns.barplot(x="batter", y="strike_rate", data=top_25.head(10), ax=ax_d2, palette="viridis")
                ax_d2.set_title(f"Strike Rate of Top 10 Suggested Batsmen")
                plt.xticks(rotation=45)
                st.pyplot(fig_d2)

    # -----------------------------------------------
    # Decision 3 — Bowler Phase Management
    # -----------------------------------------------
    elif decision == "Decision 3 — Bowler Phase Management":
        st.markdown("### How should a bowler be deployed across match phases?")
        d3_bowler = st.selectbox("Select Bowler", sorted(df['bowler'].dropna().unique()), key="d3_bowl")

        # Analyse immediately or via button
        filtered_bowl = df[df["bowler"] == d3_bowler]
        if not filtered_bowl.empty:
            phase_stats_bowl = filtered_bowl.groupby("phase").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "count"), wickets=("wicket_type", lambda x: (x != '0').sum()), dot_balls=("runs_off_bat", lambda x: (x == 0).sum())).reset_index()
            phase_stats_bowl["economy"] = phase_stats_bowl["runs"] / (phase_stats_bowl["balls"] / 6)
            phase_stats_bowl["dot_pct"] = (phase_stats_bowl["dot_balls"] / phase_stats_bowl["balls"]) * 100
            
            # Ensure index safety
            if not phase_stats_bowl.empty:
                best_phase_row_bowl = phase_stats_bowl.loc[phase_stats_bowl["economy"].idxmin()]
                best_phase_bowl = best_phase_row_bowl["phase"]
                st.success(f"**Bowler Recommendation: Deploy {d3_bowler} in {best_phase_bowl}**  \nEconomy: {best_phase_row_bowl['economy']:.2f} | Wickets: {int(best_phase_row_bowl['wickets'])} | Dot Ball %: {best_phase_row_bowl['dot_pct']:.1f}%")

                fig_d3_bowl, axes_bowl = plt.subplots(1, 3, figsize=(14, 4))
                phase_order = ["Powerplay", "Middle", "Death"]
                for ax, col, title, color in zip(axes_bowl, ["economy", "wickets", "dot_pct"], ["Economy Rate", "Wickets", "Dot Ball %"], ["coolwarm_r", "magma", "Blues_d"]):
                    data_plot = phase_stats_bowl.set_index("phase").reindex(phase_order).fillna(0).reset_index()
                    sns.barplot(x="phase", y=col, data=data_plot, ax=ax, palette=color)
                    ax.set_title(f"Bowler {title}")
                st.pyplot(fig_d3_bowl)
            else:
                st.warning("Not enough phase data for this bowler.")

    # -----------------------------------------------
    # Decision 5 — Batsman Phase Management
    # -----------------------------------------------
    elif decision == "Decision 5 — Batsman Phase Management":
        st.markdown("### How should a batsman be deployed across match phases?")
        d5_batsman = st.selectbox("Select Batsman", players_list_bat, key="d5_bat_select")
        
        filtered_bat = df[df["batter"] == d5_batsman]
        if not filtered_bat.empty:
            phase_stats_bat = filtered_bat.groupby("phase").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "count"), boundaries=("runs_off_bat", lambda x: ((x == 4) | (x == 6)).sum())).reset_index()
            phase_stats_bat["strike_rate"] = (phase_stats_bat["runs"] / phase_stats_bat["balls"]) * 100
            
            if not phase_stats_bat.empty:
                best_phase_row_bat = phase_stats_bat.loc[phase_stats_bat["strike_rate"].idxmax()]
                best_phase_bat = best_phase_row_bat["phase"]
                st.success(f"**Batsman Recommendation: Deploy {d5_batsman} in {best_phase_bat}**  \nStrike Rate: {best_phase_row_bat['strike_rate']:.2f} | Runs: {int(best_phase_row_bat['runs'])}")

                fig_d5_bat, ax_bat = plt.subplots(figsize=(8, 4))
                phase_order = ["Powerplay", "Middle", "Death"]
                sns.barplot(x="phase", y="strike_rate", data=phase_stats_bat.set_index("phase").reindex(phase_order).fillna(0).reset_index(), ax=ax_bat, palette="viridis")
                ax_bat.set_title(f"Strike Rate by Phase — {d5_batsman}")
                st.pyplot(fig_d5_bat)
            else:
                st.warning("Not enough phase data for this batsman.")

    # -----------------------------------------------
    # Decision 4 — Player Weakness Identification
    # -----------------------------------------------
    elif decision == "Decision 4 — Player Weakness Identification":
        st.markdown("### Identify a player's weakness: pace vs spin, dot ball rates, and more.")
        d4_player_type = st.radio("Analyse Player Type", ["Batsman", "Bowler"], horizontal=True, key="d4_type")

        if d4_player_type == "Batsman":
            d4_player = st.selectbox("Select Batsman", sorted(df['batter'].dropna().unique()), key="d4_bat")
            if st.button("Identify Weaknesses", key="d4_btn"):
                filtered = df[df["batter"] == d4_player]
                spin_keywords = ["chahal", "kuldeep", "ashwin", "jadeja", "tahir", "imran", "narine", "rashid", "piyush", "bishnoi", "varun"]
                spin_df = filtered[filtered["bowler"].str.lower().str.contains('|'.join(spin_keywords), na=False)]
                pace_df = filtered[~filtered["bowler"].str.lower().str.contains('|'.join(spin_keywords), na=False)]

                def compute_stats(data):
                    if len(data) == 0: return {"Strike Rate": 0, "Dot Ball %": 0, "Boundary %": 0, "Average": 0}
                    runs, balls, wickets = data["runs_off_bat"].sum(), len(data), (data["wicket_type"] != '0').sum()
                    dots = (data["runs_off_bat"] == 0).sum()
                    boundaries = ((data["runs_off_bat"] == 4) | (data["runs_off_bat"] == 6)).sum()
                    strike_rate = (runs / balls * 100) if balls > 0 else 0
                    dot_pct = (dots / balls * 100) if balls > 0 else 0
                    boundary_pct = (boundaries / balls * 100) if balls > 0 else 0
                    average = runs / max(wickets, 1)
                    return {"Strike Rate": strike_rate, "Dot Ball %": dot_pct, "Boundary %": boundary_pct, "Average": average}

                spin_stats, pace_stats = compute_stats(spin_df), compute_stats(pace_df)
                comparison = pd.DataFrame([spin_stats, pace_stats], index=["vs Spin", "vs Pace"])
                st.dataframe(comparison.style.format("{:.2f}"), use_container_width=True)
                
                if not comparison.empty:
                    if spin_stats["Strike Rate"] < pace_stats["Strike Rate"]: st.info(f"**Insight:** {d4_player} is slower against spin.")
                    else: st.info(f"**Insight:** {d4_player} is more vulnerable to pace.")
                fig_d4, ax_d4 = plt.subplots(figsize=(8, 4))
                comparison[["Strike Rate", "Dot Ball %", "Boundary %"]].plot(kind='bar', ax=ax_d4, colormap='Set2', edgecolor='black')
                ax_d4.set_title(f"Pace vs Spin Breakdown — {d4_player}")
                st.pyplot(fig_d4)

        else:
            d4_player = st.selectbox("Select Bowler", sorted(df['bowler'].dropna().unique()), key="d4_bowl")
            if st.button("Identify Strengths & Weaknesses", key="d4_bowl_btn"):
                filtered = df[df["bowler"] == d4_player]
                batsman_stats = filtered.groupby("batter").agg(runs=("runs_off_bat", "sum"), balls=("runs_off_bat", "count"), dismissals=("wicket_type", lambda x: (x != '0').sum())).reset_index()
                batsman_stats = batsman_stats[batsman_stats["balls"] >= 6]
                batsman_stats["strike_rate_against"] = (batsman_stats["runs"] / batsman_stats["balls"]) * 100
                top_threat = batsman_stats.sort_values("runs", ascending=False).head(3)
                best_matchup = batsman_stats.sort_values("dismissals", ascending=False).head(3)
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("**Most Threatened By:**"); st.dataframe(top_threat[["batter", "runs", "strike_rate_against"]], use_container_width=True)
                with col_r:
                    st.markdown("**Best Matchups:**"); st.dataframe(best_matchup[["batter", "dismissals", "strike_rate_against"]], use_container_width=True)
