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
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Market Overview",
        "Price by Make & Model",
        "Sales Volume",
        "Seasonal Trends",
        "Model Year Sweet Spot"
    ])

    # --- TAB 0: MARKET OVERVIEW (original charts) ---
    with tab0:
        st.subheader("Platform-Wide Market Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Average Sale Price Over Time")
            st.caption("Monthly average sale price across all makes and models — shows the overall trajectory of the market and whether the platform's prices are trending up or down.")

            price_over_time = df.groupby(['auction_year', 'auction_month'])['Sold_Price'].mean().reset_index()
            price_over_time['Date'] = pd.to_datetime(
                price_over_time['auction_year'].astype(str) + '-' + price_over_time['auction_month'].astype(str) + '-01'
            )
            price_over_time = price_over_time.sort_values('Date')

            line_overview = alt.Chart(price_over_time).mark_line(color='#00bfa5', point=True).encode(
                x=alt.X('Date:T', title=''),
                y=alt.Y('Sold_Price:Q', title='Average Sale Price ($)',
                        scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
                tooltip=[
                    alt.Tooltip('Date:T', format='%b %Y', title='Month'),
                    alt.Tooltip('Sold_Price:Q', format='$,.0f', title='Avg Price')
                ]
            )
            st.altair_chart(line_overview, use_container_width=True)

        with col2:
            st.write("#### Top 10 Makes by Average Sale Price")
            st.caption("The ten makes with the highest average hammer price, filtered to brands with at least 50 sales — removes small-sample outliers so only well-represented brands appear.")

            make_avg = df.groupby('Make')['Sold_Price'].agg(
                avg_price='mean', sales_count='count'
            ).reset_index()
            make_avg = make_avg[make_avg['sales_count'] >= 50].nlargest(10, 'avg_price')

            bar_makes = alt.Chart(make_avg).mark_bar(color='#00bfa5').encode(
                x=alt.X('Make:N', sort='-y', title='', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                y=alt.Y('avg_price:Q', title='Average Sale Price ($)',
                        scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
                tooltip=[
                    alt.Tooltip('Make:N', title='Make'),
                    alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price'),
                    alt.Tooltip('sales_count:Q', title='Total Sales')
                ]
            )
            st.altair_chart(bar_makes, use_container_width=True)

    # --- TAB 1: MAKE/MODEL PRICE OVER TIME ---
    with tab1:
        st.subheader("Price Trend by Make & Model")
        st.caption("Select a specific make and model to see how average sale prices have moved over time — great for spotting cars that are appreciating or cooling off.")

        col1, col2 = st.columns(2)
        with col1:
            makes = sorted(df['Make'].dropna().unique())
            selected_make = st.selectbox("Make", makes)
        with col2:
            models = sorted(df[df['Make'] == selected_make]['Model'].dropna().unique())
            selected_model = st.selectbox("Model", models)

        filtered = df[(df['Make'] == selected_make) & (df['Model'] == selected_model)]
        trend = filtered.groupby(['auction_year', 'auction_month'])['Sold_Price'].agg(
            avg_price='mean', sales_count='count'
        ).reset_index()
        trend['Date'] = pd.to_datetime(
            trend['auction_year'].astype(str) + '-' + trend['auction_month'].astype(str) + '-01'
        )
        trend = trend.sort_values('Date')
        # Filter out months with only 1 sale — single outlier auctions create very noisy lines
        trend = trend[trend['sales_count'] >= 2]

        if not trend.empty:
            line = alt.Chart(trend).mark_line(color='#00bfa5', point=True).encode(
                x=alt.X('Date:T', title=''),
                y=alt.Y('avg_price:Q', title='Average Sale Price ($)',
                        scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
                tooltip=[
                    alt.Tooltip('Date:T', format='%b %Y', title='Month'),
                    alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price'),
                    alt.Tooltip('sales_count:Q', title='Sales That Month')
                ]
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("Not enough monthly data to draw a trend for this model. Try a more commonly sold make/model.")

    # --- TAB 2: SALES VOLUME OVER TIME ---
    with tab2:
        st.subheader("Sales Volume Over Time")
        st.caption("How many cars sold per month across the whole platform — a direct read on whether the site is growing and which seasons see the most listing activity.")

        volume = df.groupby(['auction_year', 'auction_month']).size().reset_index(name='sales_count')
        volume['Date'] = pd.to_datetime(
            volume['auction_year'].astype(str) + '-' + volume['auction_month'].astype(str) + '-01'
        )
        volume = volume.sort_values('Date')

        bar = alt.Chart(volume).mark_bar(color='#00bfa5').encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('sales_count:Q', title='Number of Sales'),
            tooltip=[
                alt.Tooltip('Date:T', format='%b %Y', title='Month'),
                alt.Tooltip('sales_count:Q', title='Sales')
            ]
        )
        st.altair_chart(bar, use_container_width=True)

    # --- TAB 3: SEASONAL PRICE PATTERNS ---
    with tab3:
        st.subheader("Seasonal Price Patterns")
        st.caption("Average sale price by calendar month, collapsed across all years — shows whether buyers consistently pay more in certain seasons regardless of what year it is.")

        month_labels = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        seasonal = df.groupby('auction_month')['Sold_Price'].mean().reset_index()
        seasonal.columns = ['month_num', 'avg_price']
        seasonal['Month'] = seasonal['month_num'].map(month_labels)

        bar_seasonal = alt.Chart(seasonal).mark_bar(color='#00bfa5').encode(
            x=alt.X('Month:N', sort=list(month_labels.values()), title='',
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('avg_price:Q', title='Average Sale Price ($)',
                    scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
            tooltip=[
                alt.Tooltip('Month:N', title='Month'),
                alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price')
            ]
        )
        st.altair_chart(bar_seasonal, use_container_width=True)

    # --- TAB 4: MODEL YEAR SWEET SPOT ---
    with tab4:
        st.subheader("Model Year Sweet Spot")
        st.caption("Pick a make and see which model years command the highest average prices — useful for spotting which vintages buyers are consistently willing to pay a premium for.")

        selected_make_t4 = st.selectbox("Make", sorted(df['Make'].dropna().unique()), key='tab4_make')

        make_df = df[df['Make'] == selected_make_t4]
        year_avg = make_df.groupby('Year')['Sold_Price'].agg(
            avg_price='mean', sales_count='count'
        ).reset_index()
        # Require at least 3 sales per model year to avoid single-auction noise
        year_avg = year_avg[year_avg['sales_count'] >= 3]

        if not year_avg.empty:
            bar_year = alt.Chart(year_avg).mark_bar(color='#00bfa5').encode(
                x=alt.X('Year:O', title='Model Year', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('avg_price:Q', title='Average Sale Price ($)',
                        scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
                tooltip=[
                    alt.Tooltip('Year:O', title='Model Year'),
                    alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price'),
                    alt.Tooltip('sales_count:Q', title='Number of Sales')
                ]
            )
            st.altair_chart(bar_year, use_container_width=True)
        else:
            st.info("Not enough data across model years for this make.")
