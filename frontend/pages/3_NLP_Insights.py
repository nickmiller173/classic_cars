import altair as alt
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NLP Insights", page_icon="📝", layout="wide")
st.title("📝 Advanced Text Insights")
st.markdown("By applying Natural Language Processing to thousands of unstructured auction descriptions, we can extract hidden themes that drive vehicle valuations on Cars & Bids.")

# --- 1. Data Loading ---
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

# --- 2. Setup Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "General Keywords", 
    "Aftermarket Brands", 
    "Listing Archetypes", 
    "Listing Detail"
])

# --- TAB 1: ORIGINAL KEYWORDS ---
with tab1:
    st.subheader("General Keyword & Condition Flags")
    if not df_dashboard.empty:
        nlp_features = {
            '2_keys_ind': 'Included 2 Keys',
            'is_dry_climate_car': 'Dry Climate / Rust Free',
            'has_sport_seats': 'Sport Seats (Recaro, etc.)',
            'recent_major_service': 'Recent Major Service (Timing Belt, etc.)',
            'is_project_car': 'Project Car / Not Running'
        }
        
        results = []
        for col, readable_name in nlp_features.items():
            if col in df_dashboard.columns:
                has_feature = df_dashboard[df_dashboard[col] == 1]['Sold_Price'].mean()
                does_not_have = df_dashboard[df_dashboard[col] == 0]['Sold_Price'].mean()
                
                # Calculate the percentage bump
                percentage_premium = ((has_feature - does_not_have) / does_not_have) * 100
                
                results.append({
                    "Feature": readable_name,
                    "Avg Price (With)": has_feature,
                    "Avg Price (Without)": does_not_have,
                    "Premium (%)": percentage_premium
                })
                
        impact_df = pd.DataFrame(results).sort_values(by="Premium (%)", ascending=False)
        
        st.write("### Resale Value Impact")
        st.caption("Shows the percentage increase in average sale price when this feature is mentioned in the listing.")
        st.bar_chart(impact_df.set_index('Feature')['Premium (%)'])

# --- TAB 2: AFTERMARKET BRANDS ---
with tab2:
    st.subheader("The ROI of Premium Aftermarket Brands")
    st.markdown("Do buyers actually pay a premium for high-end aftermarket parts? Let's look at the financial impact of specific brands extracted from the raw **Modifications** and **Equipment** text.")
    
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
        st.caption("Brands higher on the Y-Axis command a larger percentage premium over the baseline. Brands further to the right are more common.")
        
        df_exploded = df_brands.dropna(subset=['Extracted_Brands_List'])
        df_exploded = df_exploded[df_exploded['Extracted_Brands_List'] != '']
        df_exploded['Brand'] = df_exploded['Extracted_Brands_List'].str.split(', ')
        df_exploded = df_exploded.explode('Brand')
        
        brand_impact = df_exploded.groupby('Brand').agg(
            Average_Sale_Price=('Sold_Price', 'mean'),
            Total_Mentions=('Sold_Price', 'count')
        ).reset_index()
        
        brand_impact = brand_impact[brand_impact['Total_Mentions'] > 2]
        # Calculate the premium as a percentage for the Y-Axis
        brand_impact['Premium_Pct'] = ((brand_impact['Average_Sale_Price'] - baseline_price) / baseline_price) * 100
        
        import altair as alt
        
        # Build a labeled scatter plot
        scatter = alt.Chart(brand_impact).mark_circle(size=100).encode(
            x=alt.X('Total_Mentions:Q', title='Total Listing Mentions'),
            y=alt.Y('Premium_Pct:Q', title='Price Premium vs Baseline (%)'),
            color=alt.Color('Brand:N', legend=None),
            tooltip=['Brand', 'Total_Mentions', 'Premium_Pct']
        )
        
        text = scatter.mark_text(
            align='left', baseline='middle', dx=10
        ).encode(text='Brand:N')
        
        st.altair_chart(scatter + text, use_container_width=True)

# --- TAB 3: LISTING ARCHETYPES ---
with tab3:
    st.subheader("The Four Hidden Car Archetypes")
    st.markdown("Using Non-Negative Matrix Factorization (NMF), we blinded an algorithm to the car's **Make** and **Model**, forcing it to cluster vehicles purely based on their build sheet, modifications, and condition.")
    
    if not df_archetypes.empty:
        cluster_mapping = {
            0: "The Loaded Luxury Cruiser",
            1: "The Rad-Era Driver (90s/00s)",
            2: "The Track & Performance Build",
            3: "The Modern EV / Financed Flip"
        }
        df_archetypes['Archetype'] = df_archetypes['Archetype_Cluster'].map(cluster_mapping)
        
        arch_summary = df_archetypes.groupby("Archetype").agg(
            Average_Price=('Sold_Price', 'mean'),
            Market_Share=('Sold_Price', 'count')
        ).reset_index()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("#### Market Share Distribution")
            # Create a donut chart with Altair
            pie = alt.Chart(arch_summary).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Market_Share", type="quantitative"),
                color=alt.Color(field="Archetype", type="nominal", legend=alt.Legend(title="Archetypes", orient="bottom")),
                tooltip=['Archetype', 'Market_Share', 'Average_Price']
            )
            st.altair_chart(pie, use_container_width=True)
            
        with col2:
            st.write("#### Average Sold Price")
            st.bar_chart(arch_summary.set_index("Archetype")["Average_Price"])

# --- TAB 4: COMPREHENSIVENESS ---
with tab4:
    st.subheader("Does Effort Equal Dollars?")
    st.markdown("We calculated the **Sentence Count** of the description fields to gauge listing comprehensiveness. Do highly detailed listings inherently attract higher bids?")
    
    if not df_effort.empty:
        # Filter out extreme outliers using sentence count
        df_effort = df_effort[(df_effort['Sentence_Count'] > 2) & (df_effort['Sentence_Count'] < 100)]
        
        bins = [0, 30000, 80000, float('inf')]
        price_labels = ['Under $30k (Entry)', '$30k - $80k (Premium)', 'Over $80k (High-End)']
        df_effort['Market_Segment'] = pd.cut(df_effort['Sold_Price'], bins=bins, labels=price_labels)
        
        # Bucket by sentences
        detail_labels = ["1: Brief", "2: Standard Detail", "3: Highly Detailed"]
        df_effort['Detail_Level'] = pd.qcut(df_effort['Sentence_Count'], q=3, labels=detail_labels)
        
        pivot = df_effort.pivot_table(index='Market_Segment', columns='Detail_Level', values='Sold_Price', aggfunc='mean')
        
        st.write("### Average Sale Price by Market Segment")
        st.caption("By segmenting the market, we can see if a detailed description adds more value to a cheap project car or an expensive supercar.")
        st.bar_chart(pivot)
        
        st.divider()
        st.write("### The Raw Data (Sentence Count vs. Price)")
        st.scatter_chart(df_effort, x="Sentence_Count", y="Sold_Price", color="Market_Segment")