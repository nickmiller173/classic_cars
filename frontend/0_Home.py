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
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    file_path = "../data/frontend_data/dashboard_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/dashboard_data.csv"
    return pd.read_csv(file_path)

df = load_data()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🚗 Cars & Bids Price Intelligence")
st.markdown(
    "An independent machine learning platform built entirely on Cars & Bids auction data. "
    "This tool combines a predictive pricing engine with deep market analysis across "
    "**{:,} auctions** and **${:.0f}M in recorded sales** — offering buyers, sellers, and the platform "
    "itself an objective lens on what cars are actually worth.".format(len(df), df['Sold_Price'].sum() / 1e6)
)

st.divider()

# ── HERO METRICS ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

total_auctions = len(df)
total_gmv = df['Sold_Price'].sum()
median_price = df['Sold_Price'].median()
makes_count = df['Make'].nunique()
models_count = df['Model'].nunique()

with m1:
    st.metric("Auctions Analyzed", f"{total_auctions:,}")
with m2:
    st.metric("Total Platform GMV", f"${total_gmv / 1e6:.0f}M+")
with m3:
    st.metric("Median Sale Price", f"${median_price:,.0f}")
with m4:
    st.metric("Unique Makes", f"{makes_count}")
with m5:
    st.metric("Data Coverage", "2020 – 2026")

st.divider()

# ── CHARTS ────────────────────────────────────────────────────────────────────
if not df.empty:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Average Sale Price Over Time")
        st.caption(
            "Monthly average hammer price across all makes and models. The platform saw significant "
            "appreciation through 2022–2023 driven by post-pandemic collector car demand, stabilizing "
            "in the $28–32k range through 2025–2026."
        )

        df['date'] = pd.to_datetime(
            df['auction_year'].astype(str) + '-' + df['auction_month'].astype(str).str.zfill(2) + '-01'
        )
        monthly = df.groupby('date')['Sold_Price'].mean().reset_index()
        monthly.columns = ['date', 'avg_price']

        area = alt.Chart(monthly).mark_area(
            color='#C4895A', opacity=0.15, line=False
        ).encode(
            x=alt.X('date:T', title=None),
            y=alt.Y('avg_price:Q', scale=alt.Scale(zero=False))
        )

        line = alt.Chart(monthly).mark_line(
            color='#C4895A', strokeWidth=2.5
        ).encode(
            x=alt.X('date:T', title=None),
            y=alt.Y('avg_price:Q', title='Avg Sale Price', axis=alt.Axis(format='$,.0f'), scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip('date:T', title='Month', format='%b %Y'),
                alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price')
            ]
        )

        st.altair_chart(area + line, use_container_width=True)

    with col_right:
        st.markdown("#### Top Makes by Average Sale Price")
        st.caption(
            "Restricted to makes with 50 or more auctions for statistical reliability. "
            "Reflects the full price distribution including condition, mileage, and spec variance within each brand."
        )

        make_stats = (
            df.groupby('Make')['Sold_Price']
            .agg(['mean', 'count'])
            .query('count >= 50')
            .sort_values('mean', ascending=False)
            .head(12)
            .reset_index()
        )
        make_stats.columns = ['Make', 'avg_price', 'count']

        bar = alt.Chart(make_stats).mark_bar(color='#C4895A').encode(
            x=alt.X('avg_price:Q', title='Average Sale Price', axis=alt.Axis(format='$,.0f')),
            y=alt.Y('Make:N', sort='-x', title=None),
            tooltip=[
                'Make',
                alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Sale Price'),
                alt.Tooltip('count:Q', format=',', title='Auctions')
            ]
        )

        st.altair_chart(bar, use_container_width=True)

    st.divider()

    # ── SECOND ROW: volume + insight stat ────────────────────────────────────
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

        top_make = df['Make'].value_counts().index[0]
        top_make_count = df['Make'].value_counts().iloc[0]

        sport_seat_premium = (
            df[df['has_sport_seats'] == 1]['Sold_Price'].median() /
            df[df['has_sport_seats'] == 0]['Sold_Price'].median() - 1
        ) * 100

        new_tire_premium = (
            df[df['has_new_tires'] == 1]['Sold_Price'].median() /
            df[df['has_new_tires'] == 0]['Sold_Price'].median() - 1
        ) * 100

        st.markdown(f"""
        <div style="display:flex; flex-direction:column; gap:12px; margin-top:8px;">
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">SINGLE-OWNER PREMIUM</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">+{one_owner_premium:.1f}%</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">median price vs. multi-owner cars</p>
            </div>
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">SPORT SEATS PREMIUM</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">+{sport_seat_premium:.1f}%</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">median price vs. standard interior</p>
            </div>
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">MOST LISTED MAKE</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">{top_make}</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">{top_make_count:,} auctions on the platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── NAVIGATION CARDS ─────────────────────────────────────────────────────────
st.markdown("#### Explore the Platform")
nav1, nav2 = st.columns(2)

with nav1:
    st.markdown("""
    <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:12px; padding:28px 32px; height:100%;">
        <p style="font-size:1.4rem; font-weight:700; color:#1a1a1a; margin:0 0 10px 0;">🔮 Price Intelligence</p>
        <p style="font-size:0.92rem; color:#444; margin:0 0 16px 0;">
            Enter any vehicle's specifications and auction listing text to receive a machine learning price estimate,
            benchmarked against historical comps from comparable sales. Built for buyers evaluating active listings
            and sellers setting realistic expectations.
        </p>
        <p style="font-size:0.82rem; color:#8B5E3C; font-weight:600; margin:0;">→ Price Predictor &nbsp;|&nbsp; Prediction Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with nav2:
    st.markdown("""
    <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:12px; padding:28px 32px; height:100%;">
        <p style="font-size:1.4rem; font-weight:700; color:#1a1a1a; margin:0 0 10px 0;">📊 Market Intelligence</p>
        <p style="font-size:0.92rem; color:#444; margin:0 0 16px 0;">
            Exploratory analysis of the full auction dataset — price trends by make and model, seasonal patterns,
            volume growth, and NLP-derived insights on how listing language and condition descriptors
            correlate with final sale price.
        </p>
        <p style="font-size:0.82rem; color:#8B5E3C; font-weight:600; margin:0;">→ Market Dashboard &nbsp;|&nbsp; NLP Insights</p>
    </div>
    """, unsafe_allow_html=True)
