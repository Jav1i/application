# -----------------------------------------------------------------------------
# 1) IMPORTS - Installing Dependencies & Import Libraries
# -----------------------------------------------------------------------------

# Streamlit: for web UI
import streamlit as st

# pandas & numpy: for data manipulation & numerical operations
import pandas as pd
import numpy as np

# requests: to call external HTTP APIs (CryptoPanic)
import requests

# CoinGeckoAPI: to fetch crypto price data
from pycoingecko import CoinGeckoAPI

# VADER sentiment analyzer: for quick text sentiment scoring
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# PRAW: Python Reddit API Wrapper, to fetch Reddit posts
import praw

# -----------------------------------------------------------------------------
# 2) CONFIGURATION & CREDENTIALS
# -----------------------------------------------------------------------------
# This section is meant to be configured with specific API credentials depending on the user...

# CryptoPanic API token:
CRYPTOPANIC_TOKEN = "3f262f77f585323d84624c5f0721bdc6a63f58e3"

# Reddit API credentials:
REDDIT_CLIENT_ID = "di0NEXnZ0rNLHEmwe2tyJw"
REDDIT_CLIENT_SECRET = "TP20TCSIRxAvpNC_X6lHP6GYdCpTow"
REDDIT_USER_AGENT = "crypto_panic_index_app_v1.0"


# -----------------------------------------------------------------------------
# 3) DATA FETCHING FUNCTIONS
# -----------------------------------------------------------------------------
def get_price_history(coin_id: str, days: int = 10000) -> pd.DataFrame:
    """
    Fetch OHLC (Open-High-Low-Close) data for a given coin from CoinGecko,
    compute daily returns and a 7-day rolling volatility.

    Args:
        coin_id: CoinGecko coin identifier (e.g., "bitcoin", "ethereum").
        days:    Number of past days to retrieve.

    Returns:
        DataFrame with:
          - close: closing price in USD
          - vol_7d: annualized 7-day rolling volatility
    """
    # Initialize the CoinGecko client
    cg = CoinGeckoAPI()

    # Fetch daily price data (returns [timestamp, price] pairs)
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    prices = data['prices']

    df = pd.DataFrame(prices, columns=['ts', 'price'])
    df['date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('date', inplace=True)

    # Compute daily returns
    df['ret'] = df['price'].pct_change()

    # Compute 7-day annualized volatility
    df['vol_7d'] = df['ret'].rolling(window=7).std() * np.sqrt(365)

    return df[['price', 'vol_7d']].dropna()


def fetch_cryptopanic() -> pd.DataFrame:
    """
    Fetch the latest news posts from CryptoPanic and assign a simple sentiment:
      - +1 for positive
      - -1 for negative
      -  0 for neutral/other

    Returns:
        DataFrame indexed by 'published_at' datetime with a 'sentiment' column.
        However, we have to remove the timezone info with [.tz_localize(None)] for the publishing time as we are trying to merge time series data
        where price_df has timezone-aware datetime values, but cp_df and reddit_df have timezone-naive datetime values (i.e. not time dependent).

    """
    # Endpoint for CryptoPanic
    url = "https://cryptopanic.com/api/v1/posts/"

    # Call the API
    res = requests.get(url, params={
        "auth_token": CRYPTOPANIC_TOKEN,
        "kind": "news",
        "public": "true"
    })
    data = res.json().get("results", [])

    # Transform into records
    records = []
    for item in data:
        # Determine sentiment from the 'domain' tag
        dom = item.get("domain", "").lower()
        if "positive" in dom:
            sentiment = 1
        elif "negative" in dom:
            sentiment = -1
        else:
            sentiment = 0

        # Parse publish time
        published = pd.to_datetime(item["published_at"]).tz_localize(None)  # (timezone-naive)

        records.append({
            "published_at": published,
            "sentiment": sentiment
        })

    # Create DataFrame, set index, sort by time
    df = pd.DataFrame(records)
    df.set_index("published_at", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_reddit(subreddit: str = "cryptocurrency", limit: int = 100) -> pd.DataFrame:
    """
    Fetch the top 'hot' posts from a subreddit and score each title
    for sentiment using VADER.

    Args:
        subreddit: which subreddit to pull posts from.
        limit:     how many posts to fetch.

    Returns:
        DataFrame indexed by 'created' datetime with a 'sentiment' column
        containing VADER compound scores (-1 to +1).
    """
    # Initialize PRAW Reddit client
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Collect sentiment records
    records = []
    for post in reddit.subreddit(subreddit).top(time_filter="all", limit=limit):
        score = analyzer.polarity_scores(post.title)["compound"]
        created = pd.to_datetime(post.created_utc, unit="s").tz_localize(None)  # (timezone-naive)
        records.append({
            "created": created,
            "sentiment": score
        })

    # Build DataFrame, set index, sort
    df = pd.DataFrame(records)
    df.set_index("created", inplace=True)
    df.sort_index(inplace=True)
    return df


# -----------------------------------------------------------------------------
# 4) PANIC INDEX CALCULATION
# -----------------------------------------------------------------------------
def build_panic_index(
        price_df: pd.DataFrame,
        reddit_df: pd.DataFrame,
        cp_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine the three signals (volatility, Reddit sentiment, CryptoPanic sentiment)
    into a single DataFrame and compute a normalized 'panic_index'.

    Steps:
      1) Resample sentiment series to daily averages.
      2) Align with price volatility dates.
      3) Normalize each series to [0,1].
      4) Composite index = 0.50*vol_norm + 0.30*reddit_norm + 0.20*cp_norm

    Returns:
        DataFrame with columns: ['vol', 'reddit', 'panic',
                                  'v_n', 'r_n', 'p_n', 'panic_index']
    """
    # Daily average Reddit sentiment
    rd_daily = reddit_df["sentiment"].resample("D").mean()

    # Daily average CryptoPanic sentiment
    cp_daily = cp_df["sentiment"].resample("D").mean()
    if cp_daily.isna().all():
        cp_daily = pd.Series(0.0, index=price_df.index)  # assume neutral sentiment

    # Build master DataFrame
    df = pd.DataFrame({
        "vol": price_df["vol_7d"].resample("D").last(),
        "reddit": rd_daily,
        "panic": cp_daily
    })

    # Fill any gaps by linear interpolation, then forward-fill
    df.interpolate(method="linear", inplace=True)
    df.fillna(method="ffill", inplace=True)

    # Define normalization helper
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val:
            return pd.Series(0.5, index=series.index)  # assign neutral value
        return (series - series.min()) / (series.max() - series.min())

    # Normalize each signal
    df["v_n"] = normalize(df["vol"])
    df["r_n"] = normalize(-df["reddit"])  # invert: more negative → higher panic
    df["p_n"] = normalize(df["panic"])

    # Compute composite panic index
    print("Normalized vol:\n", df["v_n"].describe())
    print("Normalized reddit:\n", df["r_n"].describe())
    print("Normalized cryptopanic:\n", df["p_n"].describe())

    df["panic_index"] = (
            0.50 * df["v_n"] +
            0.30 * df["r_n"] +
            0.20 * df["p_n"]
    )

    return df


# -----------------------------------------------------------------------------
# 5) STREAMLIT APP
# -----------------------------------------------------------------------------
def main():
    """
    Streamlit application entry point.
    Sets up the sidebar controls, fetches data, computes the panic index,
    and displays interactive charts.
    """
    st.set_page_config(page_title="Crypto Sentiment Dashboard", layout="wide")
    # Page title
    st.title("📈 Crypto Panic Index Dashboard")

    # Sidebar controls
    st.sidebar.header("Controls")
    coin = st.sidebar.selectbox(
        "Select cryptocurrency:",
        ["bitcoin", "ethereum", "dogecoin", "solana"]
    )
    days = st.sidebar.slider(
        "Select history window (days):",
        min_value=7,
        max_value=365,
        value=90
    )
    reddit_limit = st.sidebar.slider(
        "Number of Reddit posts:",
        min_value=50,
        max_value=300,
        value=150
    )

    # Informational text
    st.markdown("""
    This dashboard displays:
    - 7-day rolling volatility of the selected coin's USD price  
    - Average sentiment from Reddit (/r/cryptocurrency)  
    - Average sentiment from CryptoPanic news  
    - A composite **panic index** combining all three signals.
    """)

    # Fetch and compute—in showy loading spinners
    with st.spinner("Fetching price data from CoinGecko..."):
        price_df = get_price_history(coin, days=365)
    with st.spinner("Fetching Reddit sentiment..."):
        reddit_df = fetch_reddit(limit=reddit_limit)
    with st.spinner("Fetching CryptoPanic sentiment..."):
        cp_df = fetch_cryptopanic()
    with st.spinner("Calculating panic index..."):
        df = build_panic_index(price_df, reddit_df, cp_df)

        # Debug: check if the panic index has valid values
        print("Panic Index Preview:\n", df["panic_index"].dropna().tail())
        print("Total non-NaN panic_index values:", df["panic_index"].notna().sum())

    # Display raw signals
    st.subheader("Raw Signals Over Time")
    st.line_chart(df[["vol", "reddit", "panic"]])

    # Display composite index
    st.subheader("Composite Panic Index")
    st.line_chart(df["panic_index"])

    # Show data snapshot
    st.subheader("Latest Data Snapshot")
    st.dataframe(df.tail(5))

    # Footer note
    st.markdown("---")
    st.markdown("Created by Team 09.01, using Streamlit, CoinGecko, Reddit & CryptoPanic APIs.")


# Only run the app if this file is executed directly
if __name__ == "__main__":
    main()

# === Streamlit Dashboard UI ===

# crypto_sentiment_dashboard.py
"""
Streamlit dashboard to explore how crypto price volatility correlates 
with social-media sentiment (Reddit) and news sentiment (CryptoPanic).

Reuses data-fetching & panic-index logic from Crypto_panic_index.py:
  • get_price_history
  • fetch_reddit
  • fetch_cryptopanic
  • build_panic_index
"""

import streamlit as st



# Quick check that this file is actually loading
st.write("Dashboard script loaded from:", __file__)

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 2) Import your helper functions
# -----------------------------------------------------------------------------
try:
    pass
except ImportError:
    get_price_history = None
    fetch_reddit = None
    fetch_cryptopanic = None
    build_panic_index = None

# -----------------------------------------------------------------------------
# 3) Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Filters & Navigation")

MAX_DAYS = 365
today = dt.date.today()
end_date = st.sidebar.date_input(
    "End date",
    value=today,
    min_value=today - dt.timedelta(days=MAX_DAYS),
    max_value=today,
)
start_date = st.sidebar.date_input(
    "Start date",
    value=end_date - dt.timedelta(days=MAX_DAYS),
    min_value=end_date - dt.timedelta(days=MAX_DAYS),
    max_value=end_date,
)
if (end_date - start_date).days > MAX_DAYS:
    st.sidebar.warning("Free API only allows 365 days. Start date clamped.")
    start_date = end_date - dt.timedelta(days=MAX_DAYS)

reddit_limit = st.sidebar.slider("Max Reddit posts:", 0, 500, 150)
panic_limit = st.sidebar.slider("Max CryptoPanic articles:", 0, 500, 50)

COINS = ["bitcoin", "ethereum", "dogecoin", "solana"]
view = st.sidebar.radio("Select view", ["Overview"] + [c.title() for c in COINS])


# -----------------------------------------------------------------------------
# 4) Data-loading utility
# -----------------------------------------------------------------------------
def load_combined(coin: str) -> pd.DataFrame:
    days = (end_date - start_date).days

    # --- Price history (with 365-day retry) ----------------------------
    if callable(get_price_history):
        try:
            price_df = get_price_history(coin, days=days)
        except ValueError as e:
            if "exceeds the allowed time range" in str(e):
                st.warning(f"{coin.title()}: range >365d; loading last 365 days.")
                price_df = get_price_history(coin, days=MAX_DAYS)
            else:
                st.error(f"{coin.title()} price fetch error: {e}")
                return pd.DataFrame()
    else:
        st.error("get_price_history() not found.")
        return pd.DataFrame()

    # --- Reddit sentiment --------------------------------------------
    if callable(fetch_reddit):
        try:
            reddit_df = fetch_reddit(limit=reddit_limit)
        except Exception as e:
            st.error(f"{coin.title()} Reddit fetch error: {e}")
            return pd.DataFrame()
    else:
        st.error("fetch_reddit() not found.")
        return pd.DataFrame()

    # --- CryptoPanic sentiment (no limit arg) ------------------------
    if callable(fetch_cryptopanic):
        try:
            panic_df = fetch_cryptopanic()
        except Exception as e:
            st.error(f"{coin.title()} CryptoPanic fetch error: {e}")
            return pd.DataFrame()
    else:
        st.error("fetch_cryptopanic() not found.")
        return pd.DataFrame()

    # --- Composite panic index ----------------------------------------
    if callable(build_panic_index):
        combined = build_panic_index(price_df, reddit_df, panic_df)
    else:
        st.error("build_panic_index() not found.")
        return pd.DataFrame()

    return combined


# -----------------------------------------------------------------------------
# 5) Plot helpers
# -----------------------------------------------------------------------------
def plot_time_series(df: pd.DataFrame):
    """
    Render the three raw signals and panic index as native Streamlit line charts.
    """
    st.line_chart(df[["vol", "r_n", "p_n"]])
    st.line_chart(df["panic_index"])


def plot_correlation(df: pd.DataFrame, caption: str):
    """
    Small Matplotlib scatter for volatility vs normalized sentiment.
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.scatter(df["v_n"], df["r_n"], s=30, alpha=0.6, label="Reddit")
    ax.scatter(df["v_n"], df["p_n"], s=30, alpha=0.6, label="News")
    ax.set_title(caption)
    ax.set_xlabel("7d Vol (norm)")
    ax.set_ylabel("Sentiment (norm)")
    ax.legend(loc="upper left", fontsize="small")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color("white")
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# 6) Main rendering
# -----------------------------------------------------------------------------
if view == "Overview":
    st.title("World-wide Correlations Overview")
    st.markdown("All four coins in a grid layout:")

    # Create four equal columns
    cols = st.columns(4)
    for idx, coin in enumerate(COINS):
        df = load_combined(coin)
        col = cols[idx]
        with col:
            st.subheader(coin.title())
            if df.empty:
                st.warning("No data.")
                continue

            # 1) Small time-series charts
            plot_time_series(df)

            # 2) Small correlation scatter
            plot_correlation(df, f"{coin.title()} Worldwide")

else:
    # Detail view for a single coin
    coin = view.lower()
    st.title(f"🔍 {view} Analysis")
    st.markdown(
        f"- Date range: **{start_date} → {end_date}**  \n"
        f"- Max Reddit posts: **{reddit_limit}**  \n"
        f"- Max CryptoPanic articles: **{panic_limit}**"
    )

    df = load_combined(coin)
    if df.empty:
        st.error("No data available.")
    else:
        tab_w, tab_c, tab_t = st.tabs([
            "🌍 Worldwide", "🗺️ By Continent", "🏆 Top-5 Countries"
        ])

        with tab_w:
            st.subheader("Raw Signals Over Time")
            st.line_chart(df[["vol", "r_n", "p_n"]])
            st.subheader("Composite Panic Index")
            st.line_chart(df["panic_index"])
            plot_correlation(df, f"{view} Worldwide")

        with tab_c:
            continent = st.selectbox("Select Continent",
                                     ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])
            st.write(f"_Synthetic filter for_ **{continent}**")
            plot_correlation(df, f"{view} – {continent}")

        with tab_t:
            country = st.selectbox("Select Country",
                                   ["USA", "China", "India", "UK", "Germany"])
            st.write(f"_Synthetic filter for_ **{country}**")
            plot_correlation(df, f"{view} – {country}")
