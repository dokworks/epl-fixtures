import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import re
from typing import Any, Dict, List
import pytz
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Premier League Fixtures Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
STATS_URL = "https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/competitions/8/teams/stats/leaderboard"
MATCHES_URL = "https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.premierleague.com",
    "Referer": "https://www.premierleague.com/fixtures",
    "Accept": "application/json",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Cache functions to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def http_get(url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make HTTP GET request with retry logic"""
    for i in range(5):
        try:
            r = requests.get(url, params=params or {}, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** i, 8))
                continue
            try:
                body = r.json()
            except Exception:
                body = r.text[:500]
            st.error(f"HTTP {r.status_code} {url} params={params} body={body}")
            return {}
        except Exception as e:
            if i == 4:  # Last retry
                st.error(f"Failed to fetch data: {str(e)}")
                return {}
            time.sleep(min(2 ** i, 8))
    return {}

def norm_name(s: str) -> str:
    """Normalize team names for matching"""
    if not s:
        return ""
    _name_rx = re.compile(r"[^a-z0-9]+")
    return _name_rx.sub("", s.lower())

@st.cache_data(ttl=3600)
def fetch_stats(season: str = "2024", limit: int = 40) -> pd.DataFrame:
    """Fetch team statistics"""
    params = {"_sort": "total_shots:desc", "season": season, "_limit": str(limit)}
    payload = http_get(STATS_URL, params)
    
    if not payload:
        return pd.DataFrame()
    
    rows = []
    for row in payload.get("data", []):
        tm = row.get("teamMetadata", {}) or {}
        s = row.get("stats", {}) or {}
        team_name = tm.get("name")
        sot = float(s.get("shotsOnTargetIncGoals", 0) or 0)
        cin = float(s.get("shotsOnConcededInsideBox", 0) or 0)
        cout = float(s.get("shotsOnConcededOutsideBox", 0) or 0)
        denom = sot + cin + cout
        metric = (sot / denom) if denom > 0 else np.nan
        rows.append({
            "team": team_name,
            "team_norm": norm_name(team_name),
            "metric": metric,
            "metric_%": None if pd.isna(metric) else round(metric * 100, 2),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["team_norm"]).drop_duplicates(subset=["team_norm"]).reset_index(drop=True)
    return df

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_matches(season: str = "2025", page_size: int = 50) -> pd.DataFrame:
    """Fetch match fixtures"""
    params = {"competition": "8", "season": season, "_limit": str(page_size)}
    items: List[Dict[str, Any]] = []
    
    while True:
        payload = http_get(MATCHES_URL, params)
        if not payload:
            break
        items.extend(payload.get("data", []) or [])
        nxt = payload.get("pagination", {}).get("_next")
        if not nxt:
            break
        params["_next"] = nxt

    rows = []
    for m in items:
        home = m.get("homeTeam", {}) or {}
        away = m.get("awayTeam", {}) or {}
        rows.append({
            "matchId": m.get("matchId"),
            "matchWeek": m.get("matchWeek"),
            "period": m.get("period"),
            "kickoff": m.get("kickoff"),
            "ground": m.get("ground"),
            "Home Team": home.get("name"),
            "Away Team": away.get("name"),
            "home_norm": norm_name(home.get("name")),
            "away_norm": norm_name(away.get("name")),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df["kickoff_dt"] = pd.to_datetime(df["kickoff"], errors="coerce", utc=True)
        df = df.sort_values(["kickoff_dt", "matchWeek", "matchId"], na_position="last").reset_index(drop=True)
    return df

def convert_to_melbourne_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert kickoff times to Melbourne timezone"""
    if df.empty or 'kickoff' not in df.columns:
        return df
    
    melbourne_tz = pytz.timezone('Australia/Melbourne')
    df = df.copy()
    df['kickoff_utc'] = pd.to_datetime(df['kickoff'], errors='coerce', utc=True)
    
    # Apply 1-hour correction as determined in the notebook
    df['kickoff_utc_corrected'] = df['kickoff_utc'] - timedelta(hours=1)
    df['kickoff_melbourne'] = df['kickoff_utc_corrected'].dt.tz_convert(melbourne_tz)
    df['kickoff_formatted'] = df['kickoff_melbourne'].dt.strftime('%Y-%m-%d %H:%M %Z')
    
    return df

def build_schedule_with_metrics(stats_df: pd.DataFrame, matches_df: pd.DataFrame, prematch_only: bool = True) -> pd.DataFrame:
    """Build schedule with team metrics and xGDiff calculation"""
    if matches_df.empty:
        return pd.DataFrame()
    
    if prematch_only and "period" in matches_df.columns:
        matches_df = matches_df[matches_df["period"] == "PreMatch"].copy()

    # Map normalized team name -> metric/percent
    metric_by_name = dict(zip(stats_df["team_norm"], stats_df["metric"]))
    pct_by_name = dict(zip(stats_df["team_norm"], stats_df["metric_%"]))

    out = matches_df.copy()
    out["Home Team %"] = out["home_norm"].map(pct_by_name)
    out["Away Team %"] = out["away_norm"].map(pct_by_name)
    out["home_metric"] = out["home_norm"].map(metric_by_name).astype(float)
    out["away_metric"] = out["away_norm"].map(metric_by_name).astype(float)

    # xGDiff calculation
    out["xGDiff"] = (out["home_metric"] * 5.0) + (out["away_metric"] * -4.6)

    # Round values
    out["Home Team %"] = out["Home Team %"].round(2)
    out["Away Team %"] = out["Away Team %"].round(2)
    out["xGDiff"] = out["xGDiff"].round(3)

    return out

def main():
    # Header
    st.title("⚽ Premier League Fixtures Analysis")
    st.markdown("Analysis of upcoming Premier League fixtures with team statistics and Expected Goal Difference (xGDiff)")

    # Sidebar controls
    st.sidebar.header("Settings")
    stats_season = st.sidebar.selectbox("Stats Season", ["2025", "2024"], index=0)
    fixtures_season = st.sidebar.selectbox("Fixtures Season", ["2025", "2024"], index=0)
    prematch_only = st.sidebar.checkbox("PreMatch fixtures only", value=True)
    match_limit = st.sidebar.number_input("Match limit", min_value=1, max_value=100, value=20)
    
    # Add refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner("Fetching team statistics..."):
        stats_df = fetch_stats(season=stats_season, limit=40)
    
    if stats_df.empty:
        st.error("Failed to fetch team statistics")
        return

    with st.spinner("Fetching fixtures..."):
        matches_df = fetch_matches(season=fixtures_season, page_size=50)
    
    if matches_df.empty:
        st.error("Failed to fetch fixtures")
        return

    # Apply match limit
    if match_limit and not matches_df.empty:
        matches_df = matches_df.head(match_limit)

    # Convert to Melbourne time
    matches_df = convert_to_melbourne_time(matches_df)

    # Build schedule
    schedule = build_schedule_with_metrics(stats_df, matches_df, prematch_only=prematch_only)

    if schedule.empty:
        st.warning("No matches found with the current filters")
        return

    # Update schedule with Melbourne time
    if 'kickoff_formatted' in matches_df.columns:
        kickoff_mapping = dict(zip(matches_df.index, matches_df['kickoff_formatted']))
        if len(schedule) <= len(matches_df):
            schedule['Kickoff Melbourne'] = schedule.index.map(kickoff_mapping)
            schedule = schedule.drop('kickoff', axis=1, errors='ignore')
            schedule = schedule.rename(columns={'Kickoff Melbourne': 'Kickoff'})

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Teams Loaded", len(stats_df))
    with col2:
        st.metric("Total Fixtures", len(matches_df))
    with col3:
        st.metric("Fixtures Shown", len(schedule))
    with col4:
        missing_count = schedule[["Home Team %", "Away Team %"]].isna().any(axis=1).sum()
        st.metric("Missing Stats", missing_count)

    # Main table
    st.header("Fixtures with Analysis")
    
    # Prepare display columns
    display_cols = ["Home Team", "Home Team %", "Away Team", "Away Team %", "xGDiff", "matchWeek", "Kickoff", "ground"]
    display_schedule = schedule[display_cols].rename(columns={"matchWeek": "Week", "ground": "Ground"})
    
    # Style the dataframe
    def style_xgdiff(val):
        if pd.isna(val):
            return ''
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}; font-weight: bold'

    styled_schedule = display_schedule.style.applymap(style_xgdiff, subset=['xGDiff'])
    st.dataframe(styled_schedule, use_container_width=True)

    # Missing stats warning
    missing = schedule[schedule[["Home Team %", "Away Team %"]].isna().any(axis=1)]
    if not missing.empty:
        st.warning(f"⚠️ {len(missing)} fixtures have missing team statistics")
        with st.expander("View fixtures with missing stats"):
            st.dataframe(missing[["Home Team", "Away Team", "matchWeek", "Kickoff", "ground"]], use_container_width=True)

    # Download option
    csv = display_schedule.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"premier_league_fixtures_{fixtures_season}.csv",
        mime="text/csv"
    )

    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.markdown("(Home Team % × 5) + (Away Team % × -4.6)")
    st.sidebar.markdown("**Times:** Melbourne timezone with 1-hour API correction")
    
    # Show current Melbourne time
    melbourne_tz = pytz.timezone('Australia/Melbourne')
    current_time = datetime.now(melbourne_tz)
    st.sidebar.markdown(f"**Current Melbourne Time:** {current_time.strftime('%Y-%m-%d %H:%M %Z')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the logs or contact support if this persists.")
        st.stop()
