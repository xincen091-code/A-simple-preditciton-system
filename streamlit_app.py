import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from datetime import timedelta
from pandas.tseries.offsets import BDay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from wordcloud import WordCloud
import os

# PAGE / THEME
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

PRIMARY = "#0B6EFD"     # blue
ACCENT  = "#198754"     # green
WARN    = "#FF6B6B"     # red
BG_SOFT = "#F5F8FC"

st.markdown(f"""
<style>
  .main {{ background:{BG_SOFT}; }}
  .block-container {{ padding-top: 0.75rem; }}
  .kpi-card {{
      padding: 12px 14px; background: white; border: 1px solid #e9eef5;
      border-radius: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  .section-title {{
      margin-top: 0.5rem; margin-bottom: 0.25rem;
      font-weight: 700; font-size: 1.1rem; color: #0f1a2a;
  }}
</style>
""", unsafe_allow_html=True)

st.title("News Headline Sentiment Analysis & Market Movement Prediction")

# DATA LOADING 
@st.cache_data(ttl=900) 
def load_data(version=None):
    import io, requests 
    RAW_BASE = "https://raw.githubusercontent.com/crndogan/stock-news-prediction/main/notebooks"
    LOCAL_BASE = "notebooks"
    def fetch_bytes(url, timeout=20):
        r = requests.get(url, headers={"User-Agent": "streamlit-app"}, timeout=timeout)
        r.raise_for_status()
        return r.content

    def read_csv_remote(name):
        url = f"{RAW_BASE}/{name}"
        try:
            data = fetch_bytes(url)
            return pd.read_csv(io.BytesIO(data), parse_dates=["date"])
        except Exception:
            # fallback to local if remote fails
            local_path = os.path.join(LOCAL_BASE, name)
            return pd.read_csv(local_path, parse_dates=["date"]) if os.path.exists(local_path) else pd.DataFrame()

    def read_excel_remote(name):
        url = f"{RAW_BASE}/{name}"
        try:
            data = fetch_bytes(url)
            return pd.read_excel(io.BytesIO(data), parse_dates=["date"], engine="openpyxl")
        except Exception:
            local_path = os.path.join(LOCAL_BASE, name)
            return pd.read_excel(local_path, parse_dates=["date"], engine="openpyxl") if os.path.exists(local_path) else pd.DataFrame()

    # ---- Load core files 
    tone     = read_excel_remote("stock_news_tone.xlsx")
    hist     = read_csv_remote("prediction_results.csv")
    prices   = read_csv_remote("sp500_cleaned.csv")
    tomorrow = read_csv_remote("tomorrow_prediction.csv")
    topics   = read_csv_remote("topic_modeling.csv")  # (csv in your repo)

    # ---- Wordcloud
    wordcloud_df = read_csv_remote("wordcloud.csv")
    if wordcloud_df.empty:  # fallback
        wordcloud_df = read_csv_remote("topic_up_down.csv")

    # Ensure required columns exist even if file was missing
    if wordcloud_df.empty:
        wordcloud_df = pd.DataFrame(columns=["date", "word", "label", "count"])

    # ---- Metrics 
    metrics = read_csv_remote("metrics.csv")  # optional, may be empty

    # ---- Normalize/clean 
    for df in (tone, hist, prices, tomorrow, topics, wordcloud_df, metrics):
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # daily alignment for joins/windowing (keep full ts in metrics too)
            if df is not metrics:
                df["date"] = df["date"].dt.normalize()

    if "Dominant_Topic" in topics.columns:
        topics["Dominant_Topic"] = topics["Dominant_Topic"].astype(str).str.strip()

    # Standardize wordcloud columns if needed
    for col in ["word", "label"]:
        if col not in wordcloud_df.columns:
            wordcloud_df[col] = pd.Series(dtype="object")
    if "count" not in wordcloud_df.columns:
        wordcloud_df["count"] = 1

    return tone, hist, prices, tomorrow, topics, wordcloud_df, metrics


def version_stamp():
    base = "notebooks"
    files = [
        f"{base}/stock_news_tone.xlsx",
        f"{base}/prediction_results.csv",
        f"{base}/sp500_cleaned.csv",
        f"{base}/tomorrow_prediction.csv",
        f"{base}/topic_modeling.csv",
        f"{base}/topic_up_down.csv",
        f"{base}/wordcloud.csv",
        f"{base}/metrics.csv",
    ]
    return tuple(os.path.getmtime(p) for p in files if os.path.exists(p))

# Top controls
left, right = st.columns([1,1])
with left:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()

tone_df, hist_df, sp500_df, tomorrow_df, topics_df, wordcloud_df, metrics_hist = load_data(version_stamp())

# BASIC DATES / STATUS
dfs = [tone_df, hist_df, sp500_df, tomorrow_df, topics_df]
today = max(df["date"].max() for df in dfs if not df.empty).normalize()
st.sidebar.info(f"📅 Latest data date: **{today.date()}**")

# SIDEBAR FILTERS
st.sidebar.markdown("### 🔍 Filters")
st.sidebar.caption("Use the filters to update charts and metrics in real time.")

topic_options = ["All"]
if not topics_df.empty and "Dominant_Topic" in topics_df.columns:
    topic_options += sorted([int(t) if str(t).isdigit() else t
                             for t in topics_df["Dominant_Topic"].dropna().unique().tolist()])

selected_topic = st.sidebar.selectbox("Filter by Topic", options=topic_options, index=0)
selected_sentiment = st.sidebar.slider("Filter by Compound Sentiment", -1.0, 1.0, value=(-1.0, 1.0))

# Historical date range
if not hist_df.empty:
    min_hist_date = hist_df["date"].min().date()
    max_hist_date = max(today.date(), hist_df["date"].max().date())
    default_date = np.clip(pd.to_datetime(today).date(), min_hist_date, max_hist_date)
    selected_date = st.sidebar.date_input("Show history up to", value=default_date,
                                          min_value=min_hist_date, max_value=max_hist_date)
else:
    selected_date = today.date()

# Use this anchor everywhere instead of "today"
anchor_date = pd.to_datetime(selected_date).normalize()

# NEXT TRADING DAY PREDICTION
st.markdown('<div class="section-title">Next Trading Day Prediction</div>', unsafe_allow_html=True)
next_td = anchor_date + BDay(1)
pred_row = tomorrow_df[tomorrow_df["date"] == next_td]

# Fallback: nearest prediction on/after anchor_date, else latest overall
if pred_row.empty and not tomorrow_df.empty:
    cand = tomorrow_df.loc[tomorrow_df["date"] >= anchor_date]
    pred_row = cand.iloc[[0]] if not cand.empty else tomorrow_df.loc[[tomorrow_df["date"].idxmax()]]

if not pred_row.empty:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted Movement", pred_row["predicted_movement"].iloc[0])
    with c2:
        st.metric("Confidence", f"{pred_row['confidence'].iloc[0]:.1%}")
else:
    st.warning("No prediction available for the selected window.")

# SENTIMENT SNAPSHOT 
st.markdown('<div class="section-title">Selected-Day Sentiment Snapshot</div>', unsafe_allow_html=True)
tone_up_to = tone_df[tone_df["date"] <= anchor_date]
if not tone_up_to.empty:
    latest_day = tone_up_to["date"].max()
    row = tone_up_to.loc[tone_up_to["date"] == latest_day].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi-card'><b>Compound</b><br><span style='font-size:1.4rem;'>{row['sent_compound']:.3f}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><b>Emotion: Positive</b><br><span style='font-size:1.4rem;'>{row['emo_positive']:.3f}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><b>Emotion: Negative</b><br><span style='font-size:1.4rem;'>{row['emo_negative']:.3f}</span></div>", unsafe_allow_html=True)
else:
    st.info("No sentiment data up to the selected date.")

# BUILD FILTERED HISTORY 
tone_filtered = tone_df[
    (tone_df["sent_compound"].between(*selected_sentiment)) &
    (tone_df["date"] <= anchor_date)
][["date"]].drop_duplicates()

if selected_topic != "All" and not topics_df.empty:
    topic_mask = (topics_df["Dominant_Topic"] == selected_topic) & (topics_df["date"] <= anchor_date)
    topic_dates = topics_df.loc[topic_mask, ["date"]].drop_duplicates()
    driver_dates = pd.merge(tone_filtered, topic_dates, on="date", how="inner")
else:
    driver_dates = tone_filtered

filtered_hist = (
    pd.merge(
        hist_df[hist_df["date"] <= anchor_date],
        driver_dates,
        on="date",
        how="inner"
    )
    .sort_values("date")
)

label_map = {"Up": 1, "Down": 0, 1:1, 0:0}
for col in ["actual_label", "predicted_label"]:
    if filtered_hist[col].dtype == "O":
        filtered_hist[col] = filtered_hist[col].str.strip()
filtered_hist["actual_numeric"] = filtered_hist["actual_label"].map(label_map)
filtered_hist["predicted_numeric"] = filtered_hist["predicted_label"].map(label_map)

# INTERACTIVE CHART: Actual vs Predicted
st.markdown('<div class="section-title">Actual vs Predicted Market Direction</div>', unsafe_allow_html=True)
if not filtered_hist.empty:
    chart_df = filtered_hist[["date", "actual_numeric", "predicted_numeric"]].melt(
        id_vars="date", var_name="Series", value_name="Value"
    ).replace({"actual_numeric":"Actual", "predicted_numeric":"Predicted"})

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Value:Q", title="Direction (0=Down, 1=Up)", scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("date:T", title="Date"), "Series:N", alt.Tooltip("Value:Q", title="Direction")]
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No rows match the current filters (topic, sentiment, date).")

# METRICS 
st.markdown('<div class="section-title">Classification Metrics</div>', unsafe_allow_html=True)

showed_from_csv = False
if not metrics_hist.empty and {"accuracy","f1_score","precision","recall","date"}.issubset(metrics_hist.columns):
    m_hist = metrics_hist[metrics_hist["date"] <= anchor_date]
    if not m_hist.empty:
        latest = m_hist.sort_values("date").iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy",  f"{float(latest['accuracy']):.2%}")
        k2.metric("F1 Score",  f"{float(latest['f1_score']):.2f}")
        k3.metric("Precision", f"{float(latest['precision']):.2f}")
        k4.metric("Recall",    f"{float(latest['recall']):.2f}")
        st.caption(f"Last updated: {pd.to_datetime(latest['date']).strftime('%Y-%m-%d %H:%M:%S')}")
        showed_from_csv = True

if not showed_from_csv:
    metrics_df = filtered_hist.dropna(subset=["actual_numeric", "predicted_numeric"])
    if not metrics_df.empty and metrics_df["actual_numeric"].nunique() == 2:
        y_true = metrics_df["actual_numeric"]
        y_pred = metrics_df["predicted_numeric"]
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec  = recall_score(y_true, y_pred)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{acc:.2%}")
        k2.metric("F1 Score", f"{f1:.2f}")
        k3.metric("Precision", f"{prec:.2f}")
        k4.metric("Recall", f"{rec:.2f}")
        st.caption("Showing metrics computed from current filters (metrics.csv not available).")
    else:
        st.info("Not enough class variation to compute metrics, and metrics.csv not found. Loosen filters or generate metrics.csv.")

# S&P 500 TABLE
st.markdown('<div class="section-title">Recent S&P 500 Market Close</div>', unsafe_allow_html=True)
last_7_days = anchor_date - timedelta(days=7)
sp_week = sp500_df[(sp500_df["date"] >= last_7_days) & (sp500_df["date"] <= anchor_date)].copy().sort_values("date", ascending=False)

if not sp_week.empty:
    if "close_price" in sp_week.columns:
        sp_week["prev_close"] = sp_week["close_price"].shift(-1)
        sp_week["Direction"] = np.where(sp_week["close_price"] >= sp_week["prev_close"], "Up", "Down")
        sp_week.drop(columns=["prev_close"], inplace=True)
        def color_dir(val):
            if val == "Up": return "background-color: #d9f2e5"
            if val == "Down": return "background-color: #fde0e0"
            return ""
        styled = sp_week.style.format(precision=2).apply(
            lambda s: [color_dir(v) if s.name == "Direction" else "" for v in s], axis=0
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(sp_week, use_container_width=True, hide_index=True)
else:
    st.info("No S&P 500 data in the selected 7-day window.")

# TOPICS FROM LAST 7 DAYS 
st.markdown('<div class="section-title">Topics from the Last 7 Days</div>', unsafe_allow_html=True)
topics_week = topics_df[(topics_df["date"] >= last_7_days) & (topics_df["date"] <= anchor_date)].sort_values("date", ascending=False)

if not topics_week.empty and {"Dominant_Topic","Topic_Keywords"}.issubset(topics_week.columns):
    for _, row in topics_week.iterrows():
        st.markdown(f"""
        <div style='padding:10px;margin-bottom:8px;background:#ffffff;border:1px solid #e9eef5;border-radius:12px;'>
            <b>{row['date'].date()}</b> — <span style="color:{PRIMARY}">Topic #{row['Dominant_Topic']}</span><br>
            <span style="opacity:0.9">{row['Topic_Keywords']}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No topic modeling data in the selected 7-day window.")

# WORDCLOUDS
st.markdown('<div class="section-title">Topic Trends WordCloud</div>', unsafe_allow_html=True)

def _pick(df, candidates):
    """Return the first existing column (case-insensitive) from candidates."""
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None

if not wordcloud_df.empty:
    wc_raw = wordcloud_df.copy()

    # ---- standardize column names ----
    word_col  = _pick(wc_raw, ["word", "token", "term", "keyword"])
    label_col = _pick(wc_raw, ["label", "direction", "trend", "class"])
    count_col = _pick(wc_raw, ["count", "freq", "frequency", "weight"])
    date_col  = _pick(wc_raw, ["date", "day", "dt"])

    if word_col is None or label_col is None:
        st.warning(
            "Wordcloud CSV is loaded but missing required columns. "
            f"Found: {list(wc_raw.columns)} — need word/token and label/direction."
        )
    else:
        wc = wc_raw[[c for c in [word_col, label_col, count_col, date_col] if c is not None]].copy()
        wc.rename(columns={word_col: "word", label_col: "label"}, inplace=True)
        if count_col: wc.rename(columns={count_col: "count"}, inplace=True)
        else: wc["count"] = 1

        wc["word"]  = wc["word"].astype(str).str.strip()
        wc["label"] = wc["label"].astype(str).str.strip().str.lower()

        # Map common synonyms to up/down
        up_alias   = {"up","increase","increasing","rise","rising","bull","bullish","positive","green","upward"}
        down_alias = {"down","decrease","decreasing","fall","falling","bear","bearish","negative","red","downward"}
        wc["label"] = np.where(wc["label"].isin(up_alias), "up",
                        np.where(wc["label"].isin(down_alias), "down", wc["label"]))

        # ---- date filter (7-day window); if empty, fall back to all-time ----
        filtered_wc = wc.copy()
        used_window = False
        if date_col:
            wc.rename(columns={date_col: "date"}, inplace=True)
            wc["date"] = pd.to_datetime(wc["date"], errors="coerce").dt.normalize()
            mask = (wc["date"] >= last_7_days) & (wc["date"] <= anchor_date)
            filtered_wc = wc[mask]
            used_window = True

        # If the window is empty (no up/down words), show all-time instead
        def _freqs(df):
            up_f   = df.loc[df["label"] == "up"].groupby("word")["count"].sum().to_dict()
            down_f = df.loc[df["label"] == "down"].groupby("word")["count"].sum().to_dict()
            return up_f, down_f

        up_freq, down_freq = _freqs(filtered_wc)
        window_empty = (len(up_freq) == 0 and len(down_freq) == 0)

        if window_empty:
            up_freq, down_freq = _freqs(wc)  # all-time fallback

        # ---- render clouds ----
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trending on Up Days")
            fig_up, ax_up = plt.subplots(figsize=(6, 4))
            wc_up = WordCloud(background_color="white", colormap="Greens") \
                    .generate_from_frequencies(up_freq if up_freq else {"NoData": 1})
            ax_up.imshow(wc_up, interpolation="bilinear"); ax_up.axis("off")
            st.pyplot(fig_up)
        with col2:
            st.subheader("Trending on Down Days")
            fig_down, ax_down = plt.subplots(figsize=(6, 4))
            wc_down = WordCloud(background_color="white", colormap="Reds") \
                    .generate_from_frequencies(down_freq if down_freq else {"NoData": 1})
            ax_down.imshow(wc_down, interpolation="bilinear"); ax_down.axis("off")
            st.pyplot(fig_down)

        # ---- compact debug so you can verify quickly ----
        def _range(df):
            if "date" in df and not df["date"].isna().all() and len(df) > 0:
                return f"{df['date'].min().date()} → {df['date'].max().date()}"
            return "n/a"
        st.caption(
            f"Wordcloud rows total: {len(wc)} • in 7‑day window: {len(filtered_wc) if used_window else 'n/a'} • "
            f"labels total: {wc['label'].value_counts().to_dict()} • "
            f"window labels: {filtered_wc['label'].value_counts().to_dict() if used_window else 'n/a'} • "
            f"date range: {_range(wc)}"
            + (" • (No words in window → showing all‑time)" if window_empty and used_window else "")
        )
else:
    st.warning("No wordcloud data loaded. Expected notebooks/wordcloud.csv (or topic_up_down.csv).")



# FOOTER
st.markdown(
    "<hr style='margin-top: 2rem; margin-bottom: 0.5rem;'>"
    "<div style='text-align: center;'>"
    "View the full project on GitHub: "
    "<a href='https://github.com/crndogan/stock-news-prediction/tree/main' target='_blank'>"
    "crndogan/stock-news-prediction</a>"
    "</div>",
    unsafe_allow_html=True
)
