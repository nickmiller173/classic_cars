import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Market Dashboard", page_icon="📈", layout="wide")
st.title("📈 Classic Car Market Trends")
st.markdown("Explore macroeconomic trends and brand performance across the auction platform.")

@st.cache_data
def load_full_data():
    file_path = "../data/frontend_data/dashboard_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/dashboard_data.csv"
    return pd.read_csv(file_path)

df = load_full_data()

if not df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Sale Price Over Time")
        st.caption("Tracks the average hammer price across all auctions by month — a simple way to see whether the classic car market is heating up or cooling down over time.")
        trend_df = df.groupby(['auction_year', 'auction_month'])['Sold_Price'].mean().reset_index()
        trend_df['Date'] = pd.to_datetime(trend_df['auction_year'].astype(str) + '-' + trend_df['auction_month'].astype(str) + '-01')
        trend_df = trend_df.sort_values('Date')

        line = alt.Chart(trend_df).mark_line(color='#00bfa5', point=True).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Sold_Price:Q', title='Average Sale Price ($)', scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
            tooltip=[alt.Tooltip('Date:T', format='%b %Y', title='Date'), alt.Tooltip('Sold_Price:Q', format='$,.0f', title='Avg Price')]
        )
        st.altair_chart(line, use_container_width=True)

    with col2:
        st.subheader("Top 10 Makes by Average Price")
        st.caption("Among makes with at least 50 sales, these are the ones commanding the highest prices at auction — not the most common makes, just the most valuable ones.")
        make_counts = df['Make'].value_counts()
        valid_makes = make_counts[make_counts >= 50].index

        top_makes_df = (
            df[df['Make'].isin(valid_makes)]
            .groupby('Make')['Sold_Price']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_makes_df.columns = ['Make', 'Avg_Price']

        bar = alt.Chart(top_makes_df).mark_bar(color='#00bfa5').encode(
            x=alt.X('Make:N', sort='-y', title='', axis=alt.Axis(labelAngle=-45, labelLimit=200, labelOverlap=False)),
            y=alt.Y('Avg_Price:Q', title='Average Sale Price ($)', scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
            tooltip=['Make', alt.Tooltip('Avg_Price:Q', format='$,.0f', title='Avg Price')]
        )
        st.altair_chart(bar, use_container_width=True)
