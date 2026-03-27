import streamlit as st
import pandas as pd
import altair as alt
import os

# page config and custom CSS
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


# data loading
@st.cache_data
def load_data():
    # cleaned_data_no_encoding is the full, unmodified dataset used for hero stats (total auctions, GMV, median price).
    # It is preferred here because dashboard_data has a price floor applied, which would skew aggregate figures.
    # Falls back to dashboard_data when running from a deploy environment where only that file is present.
    for path in [
        "../data/frontend_data/cleaned_data_no_encoding.csv",
        "data/frontend_data/cleaned_data_no_encoding.csv",
        "../data/frontend_data/dashboard_data.csv",
        "data/frontend_data/dashboard_data.csv",
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_dashboard_data():
    # dashboard_data is loaded separately so the market signals section (emissions premium, two-keys premium)
    # uses the same filtered dataset as the NLP Insights page — keeping the numbers consistent across pages.
    for path in [
        "../data/frontend_data/dashboard_data.csv",
        "data/frontend_data/dashboard_data.csv",
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()
df_dashboard = load_dashboard_data()

# Derive most recent auction date for data currency caveat
if 'Auction_Date' in df.columns:
    latest_date = pd.to_datetime(df['Auction_Date'], errors='coerce').max()
    data_through = latest_date.strftime('%B %Y') if pd.notna(latest_date) else None
elif 'auction_year' in df.columns and 'auction_month' in df.columns:
    last = df.sort_values(['auction_year', 'auction_month']).iloc[-1]
    data_through = pd.Timestamp(year=int(last['auction_year']), month=int(last['auction_month']), day=1).strftime('%B %Y')
else:
    data_through = None

# header and intro copy
st.title("🚗 Cars & Bids Price Intelligence")
st.markdown(
    "I built this platform because, like most of the Cars & Bids community, I love tracking the prices "
    "of rare enthusiast cars as well as finding out what quirks and features make them unique. My tool "
    "combines a predictive pricing engine with raw data analysis across **{:,} auctions** and "
    "**${:.0f}M in recorded sales** — offering any user an objective lens on what these cars are actually "
    "worth (and WHY). Whether you're tracking a vintage sports car or a modern daily driver, this is my attempt "
    "to turn raw auction results into fun and interesting insights."
    .format(len(df), df['Sold_Price'].sum() / 1e6)
)
if data_through:
    st.caption(
        f"Note - this data is only current through **{data_through}**. The auction results, model predictions, and data insights do not reflect "
        "listings added after this date. All of the figures are derived from publicly available auction records "
        "and may not capture every transaction or reflect real-time platform data. "
        "For auctions that did not meet the seller's reserve, the highest recorded bid was treated as the sale price."
    )

st.divider()

# hero metrics
m1, m2, m3, m4, m5 = st.columns(5)

total_auctions = len(df)
total_gmv = df['Sold_Price'].sum()
median_price = df['Sold_Price'].median()
makes_count = df['Make'].nunique()

oldest_row = df.loc[df['Year'].idxmin()]
oldest_label = f"{int(oldest_row['Year'])} {oldest_row['Make']}"

# metric_card renders a styled HTML card instead of st.metric because st.metric doesn't
# support a subtitle line, and the CSS injection for stMetric only styles the native widget.
def metric_card(label, value, subtitle=None):
    sub = f'<div style="font-size:14px; color:#6b7280; margin:4px 0 0 0;">{subtitle}</div>' if subtitle else ''
    return f"""
    <div style="background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
        <div style="font-size:14px; color:#8B5E3C; font-weight:400; margin:0 0 4px 0;">{label}</div>
        <div style="font-size:2.25rem; font-weight:700; color:#31333F; line-height:1.2; margin:0;">{value}</div>
        {sub}
    </div>
    """

with m1:
    st.markdown(metric_card("Auctions Analyzed", f"{total_auctions:,}"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Total Platform GMV", f"${total_gmv / 1e6:.0f}M+"), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card("Median Sale Price", f"${median_price:,.0f}"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card("Unique Makes", f"{makes_count}"), unsafe_allow_html=True)
with m5:
    st.markdown(metric_card("Oldest Car", f"{int(oldest_row['Year'])}", f"{oldest_row['Make']} {oldest_row['Model']}"), unsafe_allow_html=True)

st.divider()

# navigation cards
st.markdown("#### Explore the Platform")
nav1, nav2 = st.columns(2)

with nav1:
    st.markdown("""
    <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:12px; padding:28px 32px; height:100%;">
        <p style="font-size:1.2rem; font-weight:700; color:#1a1a1a; margin:0 0 10px 0;">🔮 Price Intelligence</p>
        <p style="font-size:0.92rem; color:#444; margin:0 0 4px 0;">
            Here you can enter any vehicle's specifications and auction listing text to receive a machine learning price estimate,
            benchmarked against historical comps from comparable sales. This is the main part of the platform for buyers evaluating
            active listings and sellers setting realistic expectations.
        </p>
        <a href="/Price_Predictor" target="_self" class="nav-link">Price Predictor</a>
        <a href="/Prediction_Analysis" target="_self" class="nav-link">Prediction Analysis</a>
    </div>
    """, unsafe_allow_html=True)

with nav2:
    st.markdown("""
    <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:12px; padding:28px 32px; height:100%;">
        <p style="font-size:1.2rem; font-weight:700; color:#1a1a1a; margin:0 0 10px 0;">📊 Market Intelligence</p>
        <p style="font-size:0.92rem; color:#444; margin:0 0 4px 0;">
            Here you can find exploratory analysis of the full auction dataset. Some examples are: price trends by make and model,
            seasonal patterns, volume growth, and NLP(Natural Language Processing) derived insights on how listing language and
            condition descriptors correlate with final sale price.
        </p>
        <a href="/Market_Trends" target="_self" class="nav-link">Market Trends</a>
        <a href="/Text_Analysis" target="_self" class="nav-link">Text Analysis</a>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# preview charts
if not df.empty:
    st.markdown("#### 📊 A Glimpse of the Data")
    st.caption("The charts below are a small sample of the market intelligence available across the platform. You can explore the full analysis by clicking on the links in the sections above or on the side bar menu")

    col_vol, col_stat = st.columns([2, 1])

    with col_vol:
        st.markdown("#### Auction Volume by Year")
        st.caption("Looking at the annual listing volume, you can see Cars & Bids rapid growth since its 2020 launch.")

        vol = df.groupby('auction_year').size().reset_index(name='count')
        vol_chart = alt.Chart(vol).mark_bar(color='#8B5E3C').encode(
            x=alt.X('auction_year:O', title='Year'),
            y=alt.Y('count:Q', title='Auctions Listed'),
            tooltip=[
                alt.Tooltip('auction_year:O', title='Year'),
                alt.Tooltip('count:Q', format=',', title='Auctions')
            ]
        )
        st.altair_chart(vol_chart.properties(height=550), use_container_width=True)

    with col_stat:
        st.markdown("#### Sample Insights")
        st.caption("Here are some example insights extracted from listing text and sale outcomes across the full dataset.")

        # Use df_dashboard (not df) so these percentages match the NLP Insights page, which also uses dashboard_data.
        _sig = df_dashboard if not df_dashboard.empty else df
        emissions_premium = (
            _sig[_sig['emissions_ind'] == 1]['Sold_Price'].mean() /
            _sig[_sig['emissions_ind'] == 0]['Sold_Price'].mean() - 1
        ) * 100

        two_keys_premium = (
            _sig[_sig['2_keys_ind'] == 1]['Sold_Price'].mean() /
            _sig[_sig['2_keys_ind'] == 0]['Sold_Price'].mean() - 1
        ) * 100

        top_make = df['Make'].value_counts().index[0]
        top_make_count = df['Make'].value_counts().iloc[0]

        st.markdown(f"""
        <div style="display:flex; flex-direction:column; gap:12px; margin-top:8px;">
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">EMISSIONS MENTIONED</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">{emissions_premium:+.1f}%</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">listings that mention passing emissions or smog checks vs. those that don't</p>
            </div>
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">TWO KEYS PREMIUM</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">+{two_keys_premium:.1f}%</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">listings where the seller confirms both original keys are included vs. those that don't</p>
            </div>
            <div style="background:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:16px 20px;">
                <p style="margin:0; font-size:0.8rem; color:#8B5E3C; font-weight:600;">MOST LISTED MAKE</p>
                <p style="margin:4px 0 0 0; font-size:1.6rem; font-weight:700; color:#1a1a1a;">{top_make}</p>
                <p style="margin:2px 0 0 0; font-size:0.78rem; color:#666;">{top_make_count:,} auctions on the platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("For the full technical details of this project — including the data pipeline, model architecture, AWS infrastructure, and design decisions — please see the [README](https://github.com/nickmiller173/classic_cars/blob/main/README.md).")

