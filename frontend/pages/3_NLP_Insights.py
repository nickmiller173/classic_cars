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
                difference = has_feature - does_not_have
                
                results.append({
                    "Feature": readable_name,
                    "Avg Price (With)": has_feature,
                    "Avg Price (Without)": does_not_have,
                    "Value Premium": difference
                })
                
        impact_df = pd.DataFrame(results).sort_values(by="Value Premium", ascending=False)
        st.bar_chart(impact_df.set_index('Feature')['Value Premium'])
        st.dataframe(impact_df, use_container_width=True)

# --- TAB 2: AFTERMARKET BRANDS ---
with tab2:
    st.subheader("The ROI of Premium Aftermarket Brands")
    st.markdown("Do buyers actually pay a premium for high-end aftermarket parts? Let's look at the financial impact of specific brands extracted from the raw **Modifications** and **Equipment** text.")
    
    if not df_brands.empty:
        baseline_price = df_brands['Sold_Price'].mean()
        has_mods_price = df_brands[df_brands['Has_Premium_Mods'] == 1]['Sold_Price'].mean()
        no_mods_price = df_brands[df_brands['Has_Premium_Mods'] == 0]['Sold_Price'].mean()
        
        # High-level KPIs
        col1, col2 = st.columns(2)
        col1.metric("Avg Price (Stock / No Premium Brands)", f"${no_mods_price:,.0f}")
        col2.metric("Avg Price (With Premium Brands)", f"${has_mods_price:,.0f}", f"${has_mods_price - no_mods_price:,.0f}")
        
        st.divider()
        st.write("### Value Added by Specific Brand")
        st.caption("Shows the average price premium of a listing containing this brand versus the baseline average car price.")
        
        # Explode the comma-separated string into a format we can group
        df_exploded = df_brands.dropna(subset=['Extracted_Brands_List'])
        df_exploded = df_exploded[df_exploded['Extracted_Brands_List'] != '']
        df_exploded['Brand'] = df_exploded['Extracted_Brands_List'].str.split(', ')
        df_exploded = df_exploded.explode('Brand')
        
        brand_impact = df_exploded.groupby('Brand').agg(
            Average_Price=('Sold_Price', 'mean'),
            Mentions=('Sold_Price', 'count')
        ).reset_index()
        
        # Filter for statistically relevant brands (e.g., more than 2 mentions)
        brand_impact['Premium vs Baseline'] = brand_impact['Average_Price'] - baseline_price
        brand_impact = brand_impact[brand_impact['Mentions'] > 2].sort_values(by="Premium vs Baseline", ascending=False)
        
        st.bar_chart(brand_impact.set_index("Brand")["Premium vs Baseline"])

# --- TAB 3: LISTING ARCHETYPES ---
with tab3:
    st.subheader("The Four Hidden Car Archetypes")
    st.markdown("Using Non-Negative Matrix Factorization (NMF), we blinded an algorithm to the car's **Make** and **Model**, forcing it to cluster vehicles purely based on their build sheet, modifications, and condition.")
    
    if not df_archetypes.empty:
        # Map the clusters your Jupyter Notebook discovered
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Volume by Archetype")
            st.bar_chart(arch_summary.set_index("Archetype")["Market_Share"])
        with col2:
            st.write("#### Average Sold Price")
            st.bar_chart(arch_summary.set_index("Archetype")["Average_Price"])

# --- TAB 4: COMPREHENSIVENESS ---
with tab4:
    st.subheader("Does Effort Equal Dollars?")
    st.markdown("We calculated the total word count of the description fields to gauge listing comprehensiveness. Do highly detailed listings inherently attract higher bids?")
    
    if not df_effort.empty:
        # Filter out extreme outliers (e.g., blank descriptions)
        df_effort = df_effort[(df_effort['Total_Word_Count'] > 50) & (df_effort['Total_Word_Count'] < 1000)]
        
        # Bucket the data to make it readable
        labels = ["1: Short & Sweet", "2: Standard Detail", "3: Highly Detailed"]
        df_effort['Listing_Length_Tier'] = pd.qcut(df_effort['Total_Word_Count'], q=3, labels=labels)
        
        effort_summary = df_effort.groupby("Listing_Length_Tier")['Sold_Price'].mean()
        
        st.write("### Average Price by Description Length")
        st.bar_chart(effort_summary)
        
        st.divider()
        st.write("### The Raw Data (Word Count vs. Price)")
        st.scatter_chart(df_effort, x="Total_Word_Count", y="Sold_Price")