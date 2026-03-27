import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Auction Market Trends", page_icon="📈", layout="wide")

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #EDE8DF;
    border: 1px solid #C4A882;
    border-radius: 10px;
    padding: 16px 20px;
}
.stTabs [aria-selected="true"] {
    color: #8B5E3C !important;
    border-bottom-color: #8B5E3C !important;
}
hr { border-color: #C4A882 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Auction Market Trends")
st.markdown("Explore macroeconomic trends and brand performance across the auction platform.")

# data loading
@st.cache_data
def load_full_data():
    file_path = "../data/frontend_data/dashboard_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/dashboard_data.csv"
    return pd.read_csv(file_path)

df = load_full_data()

# tab definitions
if not df.empty:
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Market Overview",
        "Price by Make & Model",
        "Sales Volume",
        "Seasonal Trends",
        "Model Year Sweet Spot"
    ])

    with tab0:
        st.subheader("Platform-Wide Market Overview")

        st.write("#### Average Sale Price Over Time")
        st.caption("Monthly average sale price across all makes and models — shows the overall trajectory of the market and whether the platform's prices are trending up or down.")

        price_over_time = df.groupby(['auction_year', 'auction_month'])['Sold_Price'].mean().reset_index()
        price_over_time['Date'] = pd.to_datetime(
            price_over_time['auction_year'].astype(str) + '-' + price_over_time['auction_month'].astype(str) + '-01'
        )
        price_over_time = price_over_time.sort_values('Date')

        line_overview = alt.Chart(price_over_time).mark_line(color='#C4895A', point=True).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Sold_Price:Q', title='Average Sale Price ($)',
                    scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
            tooltip=[
                alt.Tooltip('Date:T', format='%b %Y', title='Month'),
                alt.Tooltip('Sold_Price:Q', format='$,.0f', title='Avg Price')
            ]
        )
        st.altair_chart(line_overview, use_container_width=True)

        # average price by make (top and bottom 10)
        make_avg = df.groupby('Make')['Sold_Price'].agg(
            avg_price='mean', sales_count='count'
        ).reset_index()
        make_avg_filtered = make_avg[make_avg['sales_count'] >= 50]

        st.write("#### Top 10 Makes by Average Sale Price")
        st.caption("The ten makes with the highest average hammer price, filtered to brands with at least 50 sales — removes small-sample outliers so only well-represented brands appear.")

        top_makes = make_avg_filtered.nlargest(10, 'avg_price')
        bar_top = alt.Chart(top_makes).mark_bar(color='#C4895A').encode(
            x=alt.X('Make:N', sort='-y', title='', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
            y=alt.Y('avg_price:Q', title='Average Sale Price ($)',
                    scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
            tooltip=[
                alt.Tooltip('Make:N', title='Make'),
                alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price'),
                alt.Tooltip('sales_count:Q', title='Total Sales')
            ]
        )
        st.altair_chart(bar_top, use_container_width=True)

        st.write("#### Bottom 10 Makes by Average Sale Price")
        st.caption("The ten makes with the lowest average hammer price among well-represented brands — useful for spotting the more accessible end of the market.")

        bottom_makes = make_avg_filtered.nsmallest(10, 'avg_price')
        bar_bottom = alt.Chart(bottom_makes).mark_bar(color='#8B3A3A').encode(
            x=alt.X('Make:N', sort='y', title='', axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
            y=alt.Y('avg_price:Q', title='Average Sale Price ($)',
                    scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
            tooltip=[
                alt.Tooltip('Make:N', title='Make'),
                alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price'),
                alt.Tooltip('sales_count:Q', title='Total Sales')
            ]
        )
        st.altair_chart(bar_bottom, use_container_width=True)

    with tab1:
        st.subheader("Price Trend by Make & Model")
        st.caption(
            "Select a make and model to see how average sale prices have moved over time. Use the optional model year "
            "filter to isolate a specific vintage — useful for separating, say, a 1990s air-cooled 911 from a modern one. "
            "Note that filtering to a single model year can produce a sparse or empty chart for less commonly sold cars, "
            "since months with fewer than 2 sales of that exact vintage are excluded to avoid misleading single-point spikes."
        )

        # Pre-compute valid combos: only show make/model pairs with at least 3 months having ≥2 sales.
        # The ≥3 qualifying months threshold prevents thin datasets from generating erratic single-point
        # "trend" lines — a model needs a meaningful sales history to produce a readable chart.
        month_counts = df.groupby(['Make', 'Model', 'auction_year', 'auction_month']).size().reset_index(name='n')
        valid_trend_combos = (
            month_counts[month_counts['n'] >= 2]
            .groupby(['Make', 'Model'])
            .size()
            .reset_index(name='qualifying_months')
        )
        valid_trend_combos = valid_trend_combos[valid_trend_combos['qualifying_months'] >= 3]

        col1, col2, col3 = st.columns(3)
        with col1:
            valid_makes_t1 = sorted(valid_trend_combos['Make'].unique())
            default_make_idx = valid_makes_t1.index('BMW') if 'BMW' in valid_makes_t1 else 0
            selected_make = st.selectbox("Make", valid_makes_t1, index=default_make_idx)
        with col2:
            valid_models_t1 = sorted(valid_trend_combos[valid_trend_combos['Make'] == selected_make]['Model'].unique())
            default_model_idx = valid_models_t1.index('3 Series') if '3 Series' in valid_models_t1 else 0
            selected_model = st.selectbox("Model", valid_models_t1, index=default_model_idx)
        with col3:
            # Optional model year filter — defaults to All Years to ensure the chart always has data.
            available_years = sorted(df[(df['Make'] == selected_make) & (df['Model'] == selected_model)]['Year'].dropna().unique().tolist(), reverse=True)
            year_options = ['All Years'] + [int(y) for y in available_years]
            selected_year = st.selectbox("Model Year (optional)", year_options)

        filtered = df[(df['Make'] == selected_make) & (df['Model'] == selected_model)]
        if selected_year != 'All Years':
            filtered = filtered[filtered['Year'] == selected_year]

        trend = filtered.groupby(['auction_year', 'auction_month'])['Sold_Price'].agg(
            avg_price='mean', sales_count='count'
        ).reset_index()
        trend['Date'] = pd.to_datetime(
            trend['auction_year'].astype(str) + '-' + trend['auction_month'].astype(str) + '-01'
        )
        trend = trend.sort_values('Date')
        # A single sale in a month means its price IS the average — one unusual result can
        # create a large spike or dip that misleads the reader about the real trend.
        trend = trend[trend['sales_count'] >= 2]

        if not trend.empty:
            line = alt.Chart(trend).mark_line(color='#C4895A', point=True).encode(
                x=alt.X('Date:T', title='', axis=alt.Axis(format='%b %Y', labelAngle=-45)),
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
            st.info("Not enough monthly data to draw a trend for this selection. Try selecting 'All Years' or a more commonly sold model year.")

    with tab2:
        st.subheader("Sales Volume Over Time")
        st.caption("How many cars sold per month across the whole platform — a direct read on whether the site is growing and which seasons see the most listing activity.")

        volume = df.groupby(['auction_year', 'auction_month']).size().reset_index(name='sales_count')
        volume['Date'] = pd.to_datetime(
            volume['auction_year'].astype(str) + '-' + volume['auction_month'].astype(str) + '-01'
        )
        volume = volume.sort_values('Date')

        bar = alt.Chart(volume).mark_bar(color='#C4895A').encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('sales_count:Q', title='Number of Sales'),
            tooltip=[
                alt.Tooltip('Date:T', format='%b %Y', title='Month'),
                alt.Tooltip('sales_count:Q', title='Sales')
            ]
        )
        st.altair_chart(bar, use_container_width=True)

    with tab3:
        st.subheader("Price Heatmap by Month & Year")
        st.caption("Each cell shows the average sale price for that month and year — darker teal means higher prices. Reading across a row shows seasonal swings within a year; reading down a column shows whether a particular month trends up or down over time.")

        month_labels = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }

        heatmap_data = df.groupby(['auction_year', 'auction_month'])['Sold_Price'].mean().reset_index()
        heatmap_data.columns = ['Year', 'month_num', 'avg_price']
        heatmap_data['Month'] = heatmap_data['month_num'].map(month_labels)

        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('Month:N', sort=list(month_labels.values()), title='',
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Year:O', title='', sort='descending'),
            color=alt.Color('avg_price:Q', title='Avg Price ($)',
                            scale=alt.Scale(scheme='oranges'),
                            legend=alt.Legend(format='$,.0f')),
            tooltip=[
                alt.Tooltip('Year:O', title='Year'),
                alt.Tooltip('Month:N', title='Month'),
                alt.Tooltip('avg_price:Q', format='$,.0f', title='Avg Price')
            ]
        ).properties(height=300)

        st.altair_chart(heatmap, use_container_width=True)

    with tab4:
        st.subheader("Model Year Sweet Spot")
        st.caption("Pick a make and model to see which production years command the highest average prices — useful for pinpointing the exact vintage buyers are willing to pay a premium for.")

        # Pre-compute valid combos: require at least 2 model years with ≥3 sales each.
        # A single qualifying model year produces a one-bar chart with nothing to compare against,
        # so the ≥2 year threshold ensures the sweet spot chart always shows a meaningful comparison.
        year_counts = df.groupby(['Make', 'Model', 'Year']).size().reset_index(name='n')
        valid_combos = (
            year_counts[year_counts['n'] >= 3]
            .groupby(['Make', 'Model'])
            .size()
            .reset_index(name='qualifying_years')
        )
        valid_combos = valid_combos[valid_combos['qualifying_years'] >= 2]

        col1, col2 = st.columns(2)
        with col1:
            valid_makes = sorted(valid_combos['Make'].unique())
            default_make_t4 = valid_makes.index('BMW') if 'BMW' in valid_makes else 0
            selected_make_t4 = st.selectbox("Make", valid_makes, index=default_make_t4, key='tab4_make')
        with col2:
            valid_models_t4 = sorted(valid_combos[valid_combos['Make'] == selected_make_t4]['Model'].unique())
            default_model_t4 = valid_models_t4.index('3 Series') if '3 Series' in valid_models_t4 else 0
            selected_model_t4 = st.selectbox("Model", valid_models_t4, index=default_model_t4, key='tab4_model')

        # aggregate by model year
        make_df = df[(df['Make'] == selected_make_t4) & (df['Model'] == selected_model_t4)]
        year_avg = make_df.groupby('Year')['Sold_Price'].agg(
            avg_price='mean', sales_count='count'
        ).reset_index()
        # Require at least 3 sales per model year to avoid single-auction noise
        year_avg = year_avg[year_avg['sales_count'] >= 3]

        if not year_avg.empty:
            bar_year = alt.Chart(year_avg).mark_bar(color='#C4895A').encode(
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
