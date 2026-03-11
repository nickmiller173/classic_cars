import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NLP Insights", page_icon="📝", layout="wide")
st.title("📝 Text Feature Impact")
st.markdown("By taking the common text fields and applying Natural Langauge Processing techniques, common themes can be extracted. Here is how specific keywords and conditions impact the final sale price.")

@st.cache_data
def load_full_data():
    file_path = "../data/frontend_data/dashboard_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/dashboard_data.csv"
    return pd.read_csv(file_path)

df = load_full_data()

if not df.empty:
    # List of the specific NLP features you engineered
    nlp_features = {
        '2_keys_ind': 'Included 2 Keys',
        'is_dry_climate_car': 'Dry Climate / Rust Free',
        'has_sport_seats': 'Sport Seats (Recaro, etc.)',
        'recent_major_service': 'Recent Major Service (Timing Belt, etc.)',
        'is_project_car': 'Project Car / Not Running'
    }
    
    results = []
    baseline_avg = df['Sold_Price'].mean()
    
    for col, readable_name in nlp_features.items():
        if col in df.columns:
            has_feature = df[df[col] == 1]['Sold_Price'].mean()
            does_not_have = df[df[col] == 0]['Sold_Price'].mean()
            
            # Calculate the raw dollar difference
            difference = has_feature - does_not_have
            
            results.append({
                "Feature": readable_name,
                "Avg Price (With)": has_feature,
                "Avg Price (Without)": does_not_have,
                "Value Premium": difference
            })
            
    impact_df = pd.DataFrame(results).sort_values(by="Value Premium", ascending=False)
    
    st.subheader("Average Price Impact by Feature")
    # Display as a bar chart showing the premium/penalty
    st.bar_chart(impact_df.set_index('Feature')['Value Premium'])
    
    st.divider()
    
    # Show the raw numbers in a clean table
    st.subheader("Detailed Breakdown")
    st.dataframe(
        impact_df, 
        use_container_width=True,
        column_config={
            "Avg Price (With)": st.column_config.NumberColumn(format="$%.2f"),
            "Avg Price (Without)": st.column_config.NumberColumn(format="$%.2f"),
            "Value Premium": st.column_config.NumberColumn(format="$%.2f")
        }
    )