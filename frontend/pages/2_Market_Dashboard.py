import streamlit as st
import pandas as pd
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
        # Group by year and month
        trend_df = df.groupby(['auction_year', 'auction_month'])['Sold_Price'].mean().reset_index()
        # Create a datetime column for plotting
        trend_df['Date'] = pd.to_datetime(trend_df['auction_year'].astype(str) + '-' + trend_df['auction_month'].astype(str) + '-01')
        trend_df = trend_df.sort_values('Date')
        
        st.line_chart(trend_df.set_index('Date')['Sold_Price'])
        
    with col2:
        st.subheader("Top 10 Makes by Average Price")
        # Filter for makes with at least 50 sales to remove extreme outliers
        make_counts = df['Make'].value_counts()
        valid_makes = make_counts[make_counts >= 50].index
        
        top_makes = df[df['Make'].isin(valid_makes)].groupby('Make')['Sold_Price'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_makes)