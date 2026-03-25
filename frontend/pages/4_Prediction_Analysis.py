import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Prediction Analysis", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stTabs [aria-selected="true"] {
    color: #8B5E3C !important;
    border-bottom-color: #8B5E3C !important;
}
hr { border-color: #C4A882 !important; }
.streamlit-expanderHeader {
    background-color: #EDE8DF;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_shap_importance():
    file_path = "../data/frontend_data/shap_importance.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/shap_importance.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_residual_data():
    file_path = "../data/frontend_data/residual_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/residual_data.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_pdp_data():
    file_path = "../data/frontend_data/pdp_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/pdp_data.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()


df_shap = load_shap_importance()
df_residuals = load_residual_data()
df_pdp = load_pdp_data()

st.title("📈 Prediction Analysis")

# --- RESIDUAL ANALYSIS ---
st.subheader("1. Prediction Accuracy (Residual Analysis)")
st.markdown(
    "Each point represents a car from the held-out test set, plotted by the model's estimate against the actual sale price. "
    "Points along the dashed diagonal indicate accurate predictions. Points above the line sold for more than expected — "
    "driven by auction dynamics, rarity, or condition factors not fully captured in the listing text. "
    "Points below the line sold for less. Color intensity reflects the magnitude of the error, with blue indicating "
    "underestimates and orange indicating overestimates."
)

if not df_residuals.empty:
    df_residuals['Error'] = df_residuals['Sold_Price'] - df_residuals['Predicted_Price']
    max_val = max(df_residuals['Predicted_Price'].max(), df_residuals['Sold_Price'].max())

    scatter = alt.Chart(df_residuals).mark_circle(opacity=0.4, size=50).encode(
        x=alt.X('Predicted_Price:Q', title='Model Estimated Value ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('Sold_Price:Q', title='Actual Hammer Price ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('Error:Q', scale=alt.Scale(domainMid=0, scheme='blueorange'), legend=None),
        tooltip=['Make', 'Model', 'Year',
                 alt.Tooltip('Sold_Price:Q', format='$,.0f', title='Actual'),
                 alt.Tooltip('Predicted_Price:Q', format='$,.0f', title='Predicted')]
    )

    diagonal = alt.Chart(
        pd.DataFrame({'x': [0, max_val], 'y': [0, max_val]})
    ).mark_line(color='black', strokeDash=[5, 5]).encode(x='x:Q', y='y:Q')

    st.altair_chart(scatter + diagonal, use_container_width=True)
else:
    st.warning("Residual data not found. Export test set predictions to residual_data.csv.")

st.divider()

# --- SHAP GLOBAL FEATURE IMPORTANCE ---
st.subheader("2. Feature Importance (SHAP Values)")
st.markdown(
    "Bar length reflects each feature's average absolute SHAP value across the test set — a model-agnostic measure of "
    "how much a given input shifts the predicted price, in log-dollar units. Longer bars indicate features the model "
    "consistently relies on to distinguish high- from low-value vehicles. This does not indicate directionality; "
    "a feature ranked highly may push prices up or down depending on its value. Features near the bottom contribute "
    "minimal signal and may be candidates for removal in future iterations."
)

if not df_shap.empty:
    fig_shap = alt.Chart(df_shap).mark_bar(color='#C4895A').encode(
        x=alt.X('mean_abs_shap:Q', title='Mean Absolute SHAP Value (log $)'),
        y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=400)),
        tooltip=['feature', alt.Tooltip('mean_abs_shap:Q', format='.4f', title='Importance')]
    )
    st.altair_chart(fig_shap, use_container_width=True)
else:
    st.warning("SHAP importance data not found. Run the SHAP cell in model_insights.ipynb and export shap_importance.csv.")

st.divider()

# --- PDP PLOTS ---
st.subheader("3. Marginal Price Effects (Partial Dependence)")
st.markdown(
    "Partial dependence plots isolate the relationship between a single feature and the predicted price by averaging "
    "out the influence of all other variables. Select a feature below to see how the model's output changes as that "
    "input varies across its observed range, holding all else equal. Steep slopes indicate high sensitivity; flat "
    "regions suggest the model treats that range as largely price-neutral. Note that interactions between features "
    "are not captured here — use SHAP values above for a more complete picture."
)

if not df_pdp.empty:
    features = sorted(df_pdp['Feature'].unique())
    selected_feature = st.selectbox("Select a Variable to Analyze:", features)

    feature_data = df_pdp[df_pdp['Feature'] == selected_feature]

    fig_pdp = alt.Chart(feature_data).mark_line(color='#C4895A', point=True).encode(
        x=alt.X('Feature_Value:Q', title=selected_feature),
        y=alt.Y('Predicted_Price:Q', title='Estimated Price ($)', scale=alt.Scale(zero=False), axis=alt.Axis(format='$,.0f')),
        tooltip=[alt.Tooltip('Feature_Value:Q', title=selected_feature),
                 alt.Tooltip('Predicted_Price:Q', format='$,.0f', title='Est. Price')]
    )
    st.altair_chart(fig_pdp, use_container_width=True)
else:
    st.warning("PDP data not found. Please run the PDP generation script.")
