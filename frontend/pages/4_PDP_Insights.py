import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="PDP Insights", page_icon="📈", layout="wide")

@st.cache_data
def load_pdp_data():
    # Handle local vs deployed file path routing
    file_path = "../data/frontend_data/pdp_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/pdp_data.csv"
        
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df_pdp = load_pdp_data()

st.title("📈 Partial Dependency Insights")
st.markdown("Explore how individual vehicle attributes impact the model's estimated price, assuming all other variables remain constant.")
st.divider()

if not df_pdp.empty:
    features = sorted(df_pdp['Feature'].unique())
    selected_feature = st.selectbox("Select a Variable to Analyze:", features)
    
    # Filter the data based on the dropdown
    feature_data = df_pdp[df_pdp['Feature'] == selected_feature]
    
    # Plot using Plotly for interactivity
    fig = px.line(
        feature_data, 
        x='Feature_Value', 
        y='Predicted_Price',
        title=f"Impact of {selected_feature} on Sold Price",
        labels={
            'Feature_Value': selected_feature, 
            'Predicted_Price': 'Estimated Price ($)'
        },
        markers=True
    )
    
    # Format the Y-axis as currency
    fig.update_layout(yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("PDP data not found. Please run the PDP generation script to create the CSV.")