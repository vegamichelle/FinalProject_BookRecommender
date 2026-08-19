"""Model-monitoring dashboard (runs on its own EC2 instance).

Connects to the SAME persistent store as the backend (DynamoDB in prod) and
reads the recommendation logs directly — never via the backend API — as
required by the spec.

Visualises:
  * Recommendation latency over time
  * Distribution of recommended genres (target / popularity drift)
  * Request volume over time
  * Cache-hit rate
  * Live relevance (helpful-rate) from user feedback
"""
import os
import sys
from collections import Counter

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import db  # noqa: E402

st.set_page_config(page_title="Book Recommender Monitoring", page_icon="📊", layout="wide")
st.title("📊 Book Recommender — Model Monitoring")

if st.sidebar.button("Refresh now"):
    st.rerun()


@st.cache_data(ttl=15)
def load_logs() -> pd.DataFrame:
    rows = db.fetch_logs()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


df = load_logs()
if df.empty:
    st.info("No recommendation requests logged yet. Use the app first.")
    st.stop()

# --- KPI row ---------------------------------------------------------------
total = len(df)
avg_latency = df["latency_ms"].mean()
cache_rate = df["cache_hit"].mean() * 100 if "cache_hit" in df else 0.0
fb = df[df["feedback"].notna()] if "feedback" in df else pd.DataFrame()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total requests", f"{total:,}")
k2.metric("Avg latency (ms)", f"{avg_latency:.1f}")
k3.metric("Cache-hit rate", f"{cache_rate:.0f}%")
k4.metric("Feedback collected", f"{len(fb):,}")

st.divider()

# --- Latency over time -----------------------------------------------------
st.subheader("Recommendation latency over time")
st.line_chart(df.set_index("timestamp")["latency_ms"])

col_a, col_b = st.columns(2)

# --- Recommended-genre distribution (target drift) -------------------------
with col_a:
    st.subheader("Recommended genre distribution")
    genres: Counter = Counter()
    for row in df.get("recommended_genres", []):
        if isinstance(row, list):
            genres.update(row)
    if genres:
        gdf = pd.Series(dict(genres)).sort_values(ascending=False)
        st.bar_chart(gdf)
    else:
        st.caption("No genre data yet.")

# --- Request volume --------------------------------------------------------
with col_b:
    st.subheader("Request volume (hourly)")
    st.bar_chart(df.set_index("timestamp").resample("1h").size())

st.divider()

# --- Recommendation source mix ---------------------------------------------
st.subheader("Recommendation source mix")
if "source" in df:
    st.bar_chart(df["source"].value_counts())

st.divider()

# --- Live relevance from feedback ------------------------------------------
st.subheader("Live relevance (from user feedback)")
if fb.empty:
    st.info("No feedback yet — use 👍 / 👎 in the app to populate this.")
else:
    helpful_rate = fb["feedback"].mean() * 100
    c1, c2 = st.columns(2)
    c1.metric("Helpful rate", f"{helpful_rate:.0f}%")
    c2.metric("Feedback samples", f"{len(fb):,}")
    st.caption("Helpful (1) vs not-helpful (0) over time")
    st.line_chart(fb.set_index("timestamp")["feedback"])

with st.expander("Raw recent logs"):
    st.dataframe(df.tail(50).sort_values("timestamp", ascending=False))
