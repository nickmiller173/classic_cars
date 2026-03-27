import altair as alt
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NLP Insights", page_icon="📝", layout="wide")

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

st.title("📝 Advanced Text Insights")
st.markdown("By applying Natural Language Processing to thousands of unstructured auction descriptions, I was able to extract hidden themes that may drive vehicle valuations on Cars & Bids.")

st.info(
        "A general note: these figures show **correlation, not causation.** Each bar reflects the average price difference between "
        "listings that mention a feature and those that don't. It does not mean the feature itself drives the price. "
        "A negative bar for 'Single Owner History', for example, doesn't mean single ownership hurts value. It "
        "likely means that single-owner cars on this platform tend to be older with lower market prices than "
        "the multi-owner modern performance cars that dominate the high end of the price range."
    )

# data loading
@st.cache_data
def load_data(filename):
    file_path = f"../data/frontend_data/{filename}"
    if not os.path.exists(file_path):
        file_path = f"data/frontend_data/{filename}"
        
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

df_dashboard = load_data("dashboard_data.csv")
df_brands = load_data("nlp_brands.csv")
df_archetypes = load_data("nlp_archetypes.csv")
df_effort = load_data("nlp_effort_scores.csv")
df_buzzwords = load_data("nlp_buzzwords.csv")

# tab definitions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "General Keywords", 
    "Aftermarket Brands", 
    "Listing Archetypes", 
    "Listing Detail",
    "Auction Buzzwords"
])

with tab1:
    st.subheader("General Keyword & Condition Flags")
    
    if not df_dashboard.empty:
        # Expanded dictionary to include all engineered text features from functions.py
        nlp_features = {
            '2_keys_ind': 'Included 2 Keys',
            'owners_manual_ind': "Included Owner's Manual",
            'is_dry_climate_car': 'Dry Climate / Rust Free',
            'is_project_car': 'Project Car / Not Running',
            'has_new_tires': 'New Tires / Fresh Rubber',
            'has_sport_seats': 'Sport Seats (Recaro, Bucket, etc.)',
            'emissions_ind': 'Emissions Mentioned',
            'one_owner_ind': 'Single Owner History',
            'carfax_ind': 'Carfax Mentioned',
            'recent_major_service': 'Recent Major Service (Timing Belt, etc.)'
        }
        
        # compute per-feature price premiums
        results = []
        for col, readable_name in nlp_features.items():
            if col in df_dashboard.columns:
                has_feature = df_dashboard[df_dashboard[col] == 1]['Sold_Price'].mean()
                does_not_have = df_dashboard[df_dashboard[col] == 0]['Sold_Price'].mean()
                
                # does_not_have > 0 guards against division by zero for rare binary flags where
                # every listing in the dataset has the feature (making the "without" group empty).
                if pd.notna(has_feature) and pd.notna(does_not_have) and does_not_have > 0:
                    percentage_premium = ((has_feature - does_not_have) / does_not_have) * 100
                    
                    results.append({
                        "Feature": readable_name,
                        "Avg Price (With)": has_feature,
                        "Avg Price (Without)": does_not_have,
                        "Premium (%)": percentage_premium
                    })
                
        impact_df = pd.DataFrame(results).sort_values(by="Premium (%)", ascending=False)
        
        st.write("### Resale Value Impact")
        st.caption("Each bar shows the average price difference between listings that mention a feature vs. those that don't — not that the feature itself causes a higher or lower price. A car described as a 'project car' sells for less because it's a project car, not because of the words used. Treat this as a signal of what types of cars use each phrase, not a recipe for writing listings.")
        
        import altair as alt
        
        # Upgraded to Altair to match the other tabs: handles angled labels and conditional colors
        bars = alt.Chart(impact_df).mark_bar().encode(
            x=alt.X('Feature:N', sort='-y', title="", axis=alt.Axis(labelAngle=-45, labelLimit=300, labelOverlap=False)),
            y=alt.Y('Premium (%):Q', title="Price Premium vs Baseline (%)", scale=alt.Scale(zero=False)),
            # Dynamically color the bars: Teal for positive value, Red for penalty
            color=alt.condition(
                alt.datum['Premium (%)'] > 0,
                alt.value('#C4895A'),  # Teal
                alt.value('#8B3A3A')   # Red
            ),
            tooltip=['Feature', alt.Tooltip('Premium (%):Q', format='.1f', title='Impact (%)')]
        )
        
        st.altair_chart(bars, use_container_width=True)
        
        # Optional: Keep the dataframe below for exact numerical reference
        with st.expander("View Raw Impact Data"):
            st.dataframe(
                impact_df, 
                use_container_width=True,
                column_config={
                    "Avg Price (With)": st.column_config.NumberColumn(format="$%.0f"),
                    "Avg Price (Without)": st.column_config.NumberColumn(format="$%.0f"),
                    "Premium (%)": st.column_config.NumberColumn(format="%.1f%%")
                }
            )

with tab2:
    st.subheader("The ROI of Premium Aftermarket Brands")
    st.markdown("The following charts attempt to answer the question: Do buyers actually pay a premium for aftermarket parts? Here I try to show the financial impact of specific brands extracted from the raw **Modifications**, **Equipment** and **Other Items Included in Sale** text.")
    
    if not df_brands.empty:
        baseline_price = df_brands['Sold_Price'].mean()
        has_mods_price = df_brands[df_brands['Has_Premium_Mods'] == 1]['Sold_Price'].mean()
        no_mods_price = df_brands[df_brands['Has_Premium_Mods'] == 0]['Sold_Price'].mean()
        
        # Calculate percentage increase
        pct_increase = ((has_mods_price - no_mods_price) / no_mods_price) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Avg Price (Stock / No Premium Brands)", f"${no_mods_price:,.0f}")
        col2.metric("Avg Price (With Premium Brands)", f"${has_mods_price:,.0f}", f"+{pct_increase:.1f}%")
        
        st.divider()
        st.write("### Brand Exclusivity vs. Value Premium")
        st.caption("Each dot is an aftermarket brand mentioned in listings. The higher it sits, the bigger the price premium it tends to command; the further right, the more commonly it shows up. Note: the premium reflects the average price of cars that happen to have that brand — expensive brands may simply show up on expensive cars regardless of whether they add value.")
        
        # explode brand list and compute per-brand premium
        df_exploded = df_brands.dropna(subset=['Extracted_Brands_List'])
        df_exploded = df_exploded[df_exploded['Extracted_Brands_List'] != '']
        df_exploded['Brand'] = df_exploded['Extracted_Brands_List'].str.split(', ')
        df_exploded = df_exploded.explode('Brand')
        
        brand_impact = df_exploded.groupby('Brand').agg(
            Average_Sale_Price=('Sold_Price', 'mean'),
            Total_Mentions=('Sold_Price', 'count')
        ).reset_index()
        
        # Filter to brands with >2 mentions — a brand mentioned once or twice is likely a
        # data entry quirk rather than a meaningful market signal, and it inflates the scatter.
        brand_impact = brand_impact[brand_impact['Total_Mentions'] > 2]
        # Calculate the premium as a percentage for the Y-Axis
        brand_impact['Premium_Pct'] = ((brand_impact['Average_Sale_Price'] - baseline_price) / baseline_price) * 100
        
        import altair as alt
        
        # Build a labeled scatter plot
        scatter = alt.Chart(brand_impact).mark_circle(size=100).encode(
            x=alt.X('Total_Mentions:Q', title='Total Listing Mentions'),
            y=alt.Y('Premium_Pct:Q', title='Price Premium vs Baseline (%)'),
            color=alt.Color('Brand:N', scale=alt.Scale(scheme='browns'), legend=None),
            tooltip=['Brand', 'Total_Mentions', 'Premium_Pct']
        )
        
        text = scatter.mark_text(
            align='left', baseline='middle', dx=10
        ).encode(text='Brand:N')
        
        st.altair_chart(scatter + text, use_container_width=True)

with tab3:
    st.subheader("The Four Hidden Car Archetypes")
    st.markdown("The charts below show an attempt at clustering cars into categories using only rich text fields. Using Non-Negative Matrix Factorization (NMF), I blinded the algorithm to the car's **Make** and **Model**, forcing it to cluster vehicles purely based on their highlights, equipment, modifications, and seller notes")
    
    if not df_archetypes.empty:
        cluster_mapping = {
            0: "The Loaded Luxury Cruiser",
            1: "The Rad-Era Driver (90s/00s)",
            2: "The Track & Performance Build",
            3: "The Modern EV / Financed Flip"
        }
        df_archetypes['Archetype'] = df_archetypes['Archetype_Cluster'].map(cluster_mapping)
        
        # aggregate archetype stats for charts
        arch_summary = df_archetypes.groupby("Archetype").agg(
            Average_Price=('Sold_Price', 'mean'),
            Market_Share=('Sold_Price', 'count')
        ).reset_index()
        
        import altair as alt
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("#### Market Share Distribution")
            st.caption("How the market splits across the four listing archetypes — essentially, which type of car listing is most common on the platform.")
            pie = alt.Chart(arch_summary).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Market_Share", type="quantitative"),
                color=alt.Color(field="Archetype", type="nominal",
                            scale=alt.Scale(range=['#C4895A', '#8B5E3C', '#8B3A3A', '#D4B896']),
                            legend=alt.Legend(title="Archetypes", orient="bottom")),
                tooltip=['Archetype', 'Market_Share', 'Average_Price']
            )
            st.altair_chart(pie, use_container_width=True)
            
        with col2:
            st.write("#### Average Sold Price")
            st.caption("The average hammer price for each archetype — so you can see which type of listing tends to attract the most money at auction.")
            # Upgraded to Altair to sync colors with the pie chart and fix label clipping
            bar = alt.Chart(arch_summary).mark_bar().encode(
                x=alt.X("Archetype:N", title="", axis=alt.Axis(labelAngle=-45, labelLimit=300, labelOverlap=False)),
                y=alt.Y("Average_Price:Q", title="Average Price ($)", scale=alt.Scale(zero=False)),
                color=alt.Color("Archetype:N", scale=alt.Scale(range=['#C4895A', '#8B5E3C', '#8B3A3A', '#D4B896']), legend=None),
                tooltip=['Archetype', 'Average_Price']
            )
            st.altair_chart(bar, use_container_width=True)

with tab4:
    st.subheader("Head-to-Head comparison of listing text length")
    st.markdown("The following charts attempt to answer the question: Does writing a longer description always help? You can use the dropdowns below to compare how word count in different sections impacts the final sale price.")
    
    # Map the clean CSV columns back to readable dropdown options
    wc_columns = {
        'Highlights_WC': 'Highlights',
        'Known_Flaws_WC': 'Known Flaws',
        'Modifications_WC': 'Modifications',
        'Equipment_WC': 'Equipment',
        'Recent_Service_WC': 'Recent Service History',
        'Ownership_WC': 'Ownership History',
        'Other_Items_WC': 'Other Items Included in Sale',
        'Seller_Notes_WC': 'Seller Notes'
    }
    
    if not df_effort.empty and 'Highlights_WC' in df_effort.columns:
        
        import altair as alt
        
        # Updated to dynamically zoom by dropping the top 5% extreme outliers
        def build_scatter_trend(x_col, x_title, line_color):
            # Cap at the 95th percentile for both axes independently so that a single listing
            # with 2,000 words or a $500k sale doesn't compress the entire visible chart into
            # a corner — the trendline still fits all data but the view focuses on the main mass.
            x_cap = df_effort[x_col].quantile(0.95)
            y_cap = df_effort['Sold_Price'].quantile(0.95)

            # Create a localized dataframe that drops the extreme outliers
            zoomed_df = df_effort[(df_effort[x_col] <= x_cap) & (df_effort['Sold_Price'] <= y_cap)]
            
            scatter = alt.Chart(zoomed_df).mark_circle(opacity=0.3, size=50, color="#808495").encode(
                x=alt.X(f'{x_col}:Q', title=f"{x_title} Word Count"),
                y=alt.Y('Sold_Price:Q', title="Sold Price ($)")
            )
            
            trendline = scatter.transform_regression(
                x_col, 'Sold_Price'
            ).mark_line(size=5, color=line_color)
            
            return scatter + trendline

        st.caption("Each dot is a real listing — the trendline shows whether longer descriptions in that section correlate with higher or lower prices. A positive slope doesn't mean writing more words causes a higher sale price; it likely means sellers of more expensive cars tend to write more detail. Charts are zoomed to the 95th percentile to hide outliers.")
        col1, col2 = st.columns(2)
        
        with col1:
            section_1 = st.selectbox("Select First Section", options=list(wc_columns.keys()), format_func=lambda x: wc_columns[x], index=0)
            st.altair_chart(build_scatter_trend(section_1, wc_columns[section_1], '#C4895A'), use_container_width=True)
            
        with col2:
            section_2 = st.selectbox("Select Second Section", options=list(wc_columns.keys()), format_func=lambda x: wc_columns[x], index=1) 
            st.altair_chart(build_scatter_trend(section_2, wc_columns[section_2], '#8B3A3A'), use_container_width=True)

with tab5:
    st.subheader("The 'Auction Buzzword' Impact Analyzer")
    st.markdown("Which specific words or phrases extracted from the text are most associated with high or low sale prices?")

    st.caption("These values show the average price difference between listings containing each word and those that don't — they are correlations, not causes. A word like 'performance' appearing with a large negative value doesn't mean writing it hurts your sale; it more likely means cheaper sporty cars use that word more often. Use this to understand what language tends to appear in different price brackets, not as writing advice.")

    if not df_buzzwords.empty:
        # Split into top 15 premium and top 15 discount words
        top_premium = df_buzzwords[df_buzzwords['Impact_Value'] > 0].nlargest(15, 'Impact_Value')
        top_discount = df_buzzwords[df_buzzwords['Impact_Value'] < 0].nsmallest(15, 'Impact_Value')
        
        combined_buzzwords = pd.concat([top_premium, top_discount])

        # render horizontal bar chart
        buzz_bar = alt.Chart(combined_buzzwords).mark_bar().encode(
            x=alt.X('Impact_Value:Q', title="Avg Price Difference vs. Listings Without This Word ($)"),
            y=alt.Y('Word:N', sort='-x', title="",
                    axis=alt.Axis(labelLimit=300, labelOverlap=False)),
            color=alt.condition(
                alt.datum.Impact_Value > 0,
                alt.value('#C4895A'),  # Teal for Premium
                alt.value('#8B3A3A')   # Red for Discount
            ),
            tooltip=['Word', 'Impact_Value', 'Frequency']
        )
        
        st.altair_chart(buzz_bar, use_container_width=True)
    else:
        st.warning("Buzzword data not found. Export TF-IDF coefficients to nlp_buzzwords.csv.")