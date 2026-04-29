"""
Streamlit dashboard for P2-ETF-EMD-HYBRID results.
Displays:
- Top picks for each universe (Global & Shrinking consensus)
- Selected models per IMF for the top picks
- Next US trading day
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from huggingface_hub import HfApi, HfFileSystem
import config
from us_calendar import next_trading_day

# Page config
st.set_page_config(page_title="EMD-Hybrid ETF Forecast", layout="wide")
st.title("📈 P2-ETF-EMD-HYBRID")
st.caption("Absolute return forecast via CEEMDAN + SVR/MLP/LightGBM")

# Load latest result from HF
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_latest_result():
    fs = HfFileSystem(token=config.HF_TOKEN)
    repo = config.HF_OUTPUT_REPO
    try:
        files = fs.ls(f"datasets/{repo}")
        json_files = [f for f in files if f.endswith('.json')]
        if not json_files:
            return None
        latest = max(json_files)  # lexicographic works for YYYY-MM-DD
        with fs.open(latest, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None

result = load_latest_result()
if result is None:
    st.warning("No forecast results found. Run trainer.py first.")
    st.stop()

# Sidebar info
st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {result['run_date']}")
next_trade = next_trading_day()
st.sidebar.write(f"**Next trading day:** {next_trade}")
st.sidebar.write("**Method:** CEEMDAN → IMFs → best of SVR/MLP/LGBM per IMF → sum")

# Display universes
universes = result['universes']
for universe_name, modes in universes.items():
    st.header(f"🌍 {universe_name}")
    cols = st.columns(2)
    
    # Global mode
    if 'global' in modes:
        global_data = modes['global']
        top3 = global_data['top_picks']
        with cols[0]:
            st.subheader("Global Model (2008–today)")
            df_global = pd.DataFrame(top3)
            df_global['predicted_return'] = df_global['predicted_return'].apply(lambda x: f"{x:.6f}")
            st.dataframe(df_global, hide_index=True)
            # Show model selection for the top pick (if available in the JSON)
            # The JSON stores 'selected_models' per ticker under 'all_scores'
            all_scores = global_data.get('all_scores', [])
            if all_scores:
                top_ticker = top3[0]['ticker']
                for item in all_scores:
                    if item['ticker'] == top_ticker and 'selected_models' in item:
                        models = item['selected_models']
                        st.caption(f"Top pick model mix: {models}")
                        break
    
    # Shrinking mode
    if 'shrinking' in modes:
        shrink = modes['shrinking']
        with cols[1]:
            st.subheader("Shrinking Windows (3‑year consensus)")
            st.metric("Consensus pick", shrink['consensus_ticker'], f"{shrink['conviction']:.0f}% conviction")
            st.write(f"Based on {shrink['num_windows']} windows")
            if st.checkbox(f"Show window details for {universe_name}"):
                windows_df = pd.DataFrame(shrink['windows'])
                st.dataframe(windows_df)
    
    st.divider()

# Footer
st.caption("Data source: P2SAMAPA/fi-etf-macro-signal-master-data | Results stored in: P2SAMAPA/p2-etf-emd-hybrid-results")
