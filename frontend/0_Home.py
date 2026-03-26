import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Cars & Bids Price Intelligence", page_icon="🚗", layout="wide")

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #EDE8DF;
    border: 1px solid #C4A882;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label {
    color: #8B5E3C !important;
    font-weight: 600;
}
hr { border-color: #C4A882 !important; }
.nav-link {
    display: inline-block;
    background-color: #fff;
    border: 1px solid #C4A882;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #8B5E3C !important;
    text-decoration: none;
    margin-top: 16px;
    margin-right: 8px;
}
.nav-link:hover {
    background-color: #C4A882;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    # Prefer cleaned_data_no_encoding (full dataset, pre-price-floor) — falls back to dashboard_data
    for path in [
        "../data/frontend_data/cleaned_data_no_encoding.csv",
        "data/frontend_data/cleaned_data_no_encoding.csv",
        "../data/frontend_data/dashboard_data.csv",
        "data/frontend_data/dashboard_data.csv",
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()

# Derive most recent auction date for data currency caveat
if 'Auction_Date' in df.columns:
    latest_date = pd.to_datetime(df['Auction_Date'], errors='coerce').max()
    data_through = latest_date.strftime('%B %Y') if pd.notna(latest_date) else None
elif 'auction_year' in df.columns and 'auction_month' in df.columns:
    last = df.sort_values(['auction_year', 'auction_month']).iloc[-1]
    data_through = pd.Timestamp(year=int(last['auction_year']), month=int(last['auction_month']), day=1).strftime('%B %Y')
else:
    data_through = None

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🚗 Cars & Bids Price Intelligence")
st.markdown(
    "An independent machine learning platform built entirely on Cars & Bids auction data. "
    "This tool combines a predictive pricing engine with deep market analysis across "
    "**{:,} auctions** and **${:.0f}M in recorded sales** — offering buyers, sellers, and the platform "
    "itself an objective lens on what cars are actually worth.".format(len(df), df['Sold_Price'].sum() / 1e6)
)

if data_through:
    st.caption(
        f"Data current through **{data_through}**. Auction results and model predictions do not reflect "
        "listings added after this date. All figures are derived from publicly available auction records "
        "and may not capture every transaction or reflect real-time platform data. "
        "For auctions that did not meet the seller's reserve, the highest recorded bid was treated as the sale price."
    )

st.divider()

# ── HERO METRICS ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

total_auctions = len(df)
total_gmv = df['Sold_Price'].sum()
median_price = df['Sold_Price'].median()
makes_count = df['Make'].nunique()

oldest_row = df.loc[df['Year'].idxmin()]
oldest_label = f"{int(oldest_row['Year'])} {oldest_row['Make']}"

with m1:
    st.metric("Auctions Analyzed", f"{total_auctions:,}")
with m2:
    st.metric("Total Platform GMV", f"${total_gmv / 1e6:.0f}M+")
with m3:
    st.metric("Median Sale Price", f"${median_price:,.0f}")
with m4:
    st.metric("Unique Makes", f"{makes_count}")
with m5:
    st.markdown(f"""
    <div style="background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
        <div style="font-size:14px; color:#8B5E3C; font-weight:600; margin:0 0 4px 0;">Oldest Car</div>
        <div style="font-size:2.25rem; font-weight:600; color:#31333F; line-height:1.2; margin:0;">{int(oldest_row['Year'])}</div>
        <div style="font-size:14px; color:#6b7280; margin:4px 0 0 0;">{oldest_row['Make']} {oldest_row['Model']}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── NAVIGATION CARDS ─────────────────────────────────────────────────────────
st.markdown("#### Explore the Platform")
nav1, nav2 = st.columns(2)

with nav1:
    st.markdown("""
    <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:12px; padding:28px 32px; height:100%;">
        <p style="font-size:1.2rem; font-weight:700; color:#1a1a1a; margin:0 0 10px 0;">🔮 Price Intelligence</p>
        <p style="font-size:0.92rem; color:#444; margin:0 0 4px 0;">
            Enter any vehicle's specifications and auction listing text to receive a machine learning price estimate,
            benchmarked against historical comps from comparable sales. Built for buyers evaluating active listings
            and sellers setting realistic expectations.
        </p>
        <a href="/Predictor" target="_self" class="nav-link">Price Predictor</a>
        <a href="/Prediction_Analysis" target="_self" class="nav-link">Prediction Analysis</a>
    </div>
    """, unsafe_allow_html=True)

with nav2:
    st.markdown("""
    <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:12px; padding:28px 32px; height:100%;">
        <p style="font-size:1.2rem; font-weight:700; color:#1a1a1a; margin:0 0 10px 0;">📊 Market Intelligence</p>
        <p style="font-size:0.92rem; color:#444; margin:0 0 4px 0;">
            Exploratory analysis of the full auction dataset — price trends by make and model, seasonal patterns,
            volume growth, and NLP-derived insights on how listing language and condition descriptors
            correlate with final sale price.
        </p>
        <a href="/Market_Dashboard" target="_self" class="nav-link">Market Dashboard</a>
        <a href="/NLP_Insights" target="_self" class="nav-link">NLP Insights</a>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── CHARTS ────────────────────────────────────────────────────────────────────
if not df.empty:
    # ── CHARTS + MARKET SIGNALS ──────────────────────────────────────────────
    col_vol, col_stat = st.columns([2, 1])

    with col_vol:
        st.markdown("#### Auction Volume by Year")
        st.caption("Annual listing volume reflects the platform's rapid growth since its 2020 launch.")

        vol = df.groupby('auction_year').size().reset_index(name='count')
        vol_chart = alt.Chart(vol).mark_bar(color='#8B5E3C').encode(
            x=alt.X('auction_year:O', title='Year'),
            y=alt.Y('count:Q', title='Auctions Listed'),
            tooltip=[
                alt.Tooltip('auction_year:O', title='Year'),
                alt.Tooltip('count:Q', format=',', title='Auctions')
            ]
        )
        st.altair_chart(vol_chart, use_container_width=True)

    with col_stat:
        st.markdown("#### Market Signals")
        st.caption("Behavioral patterns extracted from listing text and sale outcomes across the full dataset.")

        one_owner_premium = (
            df[df['one_owner_ind'] == 1]['Sold_Price'].median() /
            df[df['one_owner_ind'] == 0]['Sold_Price'].median() - 1
        ) * 100

        two_keys_premium = (
            df[df['2_keys_ind'] == 1]['Sold_Price'].median() /
            df[df['2_keys_ind'] == 0]['Sold_Price'].median() - 1
        ) * 100

        top_make = df['Make'].value_counts().index[0]
        top_make_count = df['Make'].value_counts().iloc[0]

        st.markdown(f"""
        <div style="display:flex; flex-direction:column; gap:12px; margin-top:8px;">
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">SINGLE-OWNER PREMIUM</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">+{one_owner_premium:.1f}%</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">median price vs. multi-owner cars</p>
            </div>
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">TWO KEYS PREMIUM</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">+{two_keys_premium:.1f}%</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">median price when both keys are present</p>
            </div>
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">MOST LISTED MAKE</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">{top_make}</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">{top_make_count:,} auctions on the platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

