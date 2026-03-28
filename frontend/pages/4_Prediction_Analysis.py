import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
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
[data-testid="stMetric"] {
    background-color: #EDE8DF;
    border: 1px solid #C4A882;
    border-radius: 10px;
    padding: 16px 20px;
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
st.markdown(
    "This page provides a transparent look at how the price predictor model works and how well it performs. "
    "The predictor you used on the previous page is an XGBoost machine learning model trained on thousands of "
    "historical Cars & Bids auction results. Rather than taking that estimate on faith, the charts below let you "
    "examine the model's accuracy, understand which inputs it relies on most heavily, and see how individual "
    "variables influence the predicted price. If you ever wonder why the predictor returned a certain number, "
    "this page is where to look."
)
st.divider()

# accuracy summary metrics
if not df_residuals.empty:
    df_residuals['Error'] = df_residuals['Sold_Price'] - df_residuals['Predicted_Price']
    df_residuals['Abs_Pct_Error'] = (df_residuals['Error'].abs() / df_residuals['Sold_Price']) * 100

    r2 = 1 - (df_residuals['Error'] ** 2).sum() / ((df_residuals['Sold_Price'] - df_residuals['Sold_Price'].mean()) ** 2).sum()
    rmse = np.sqrt((df_residuals['Error'] ** 2).mean())
    median_ape = df_residuals['Abs_Pct_Error'].median()

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.3f}", help="Proportion of price variance explained by the model. 1.0 is perfect; 0.0 means no better than guessing the mean.")
    m2.metric("RMSE", f"${rmse:,.0f}", help="Root Mean Squared Error — the average dollar magnitude of the model's mistakes, weighted toward larger errors.")
    m3.metric("Median Absolute % Error", f"{median_ape:.1f}%", help="The typical prediction is off by this percentage. Half of predictions fall within this band; half fall outside it.")

    st.divider()

# residual scatter
st.subheader("1. Prediction Accuracy (Residual Analysis)")
st.markdown(
    "Each point represents a car from the test data set, plotted by the model's estimate against the actual sale price. "
    "Points along the dashed diagonal indicate accurate predictions. Points above the line sold for more than expected, which could potentially be"
    "driven by auction dynamics, rarity, or condition factors not fully captured in the listing text. "
    "Points below the line sold for less than expected. Color intensity reflects the magnitude of the error, with blue indicating "
    "underestimates and orange indicating overestimates."
)

if not df_residuals.empty:
    max_val = max(df_residuals['Predicted_Price'].max(), df_residuals['Sold_Price'].max())

    scatter = alt.Chart(df_residuals).mark_circle(opacity=0.4, size=50).encode(
        x=alt.X('Predicted_Price:Q', title='Model Estimated Value ($)', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('Sold_Price:Q', title='Actual Hammer Price ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('Error:Q', scale=alt.Scale(domainMid=0, scheme='blueorange'), legend=None),
        tooltip=['Make', 'Model', 'Year',
                 alt.Tooltip('Sold_Price:Q', format='$,.0f', title='Actual'),
                 alt.Tooltip('Predicted_Price:Q', format='$,.0f', title='Predicted')]
    )

    # The diagonal is the "perfect prediction" line (predicted == actual). Points above it
    # sold for more than the model expected; points below sold for less.
    diagonal = alt.Chart(
        pd.DataFrame({'x': [0, max_val], 'y': [0, max_val]})
    ).mark_line(color='black', strokeDash=[5, 5]).encode(x='x:Q', y='y:Q')

    st.altair_chart(scatter + diagonal, use_container_width=True)
else:
    st.warning("Residual data not found. Export test set predictions to residual_data.csv.")

st.divider()

# residuals by make
st.subheader("2. Prediction Bias by Make")
st.markdown(
    "Each box shows the distribution of prediction errors (Actual − Predicted) for that make across the test set. "
    "A box centered above zero means the model tends to underestimate that brand — buyers paid more than the model expected. "
    "A box below zero means the model overestimates — it predicted higher than what cars actually sold for. "
    "Makes with fewer than 5 test set appearances are excluded."
)

if not df_residuals.empty:
    # Filter to makes with enough test-set samples for a meaningful error distribution.
    make_counts = df_residuals['Make'].value_counts()
    valid_makes_resid = make_counts[make_counts >= 5].index.tolist()
    df_resid_makes = df_residuals[df_residuals['Make'].isin(valid_makes_resid)]

    # Sort makes by median error so the chart reads from most-underestimated to most-overestimated.
    make_order = (
        df_resid_makes.groupby('Make')['Error']
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    box_makes = alt.Chart(df_resid_makes).mark_boxplot(
        color='#C4895A', outliers={'color': '#C4895A', 'opacity': 0.3, 'size': 15}
    ).encode(
        x=alt.X('Make:N', sort=make_order, title='', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Error:Q', title='Prediction Error — Actual minus Predicted ($)',
                axis=alt.Axis(format='$,.0f')),
        tooltip=[alt.Tooltip('Make:N', title='Make')]
    ).properties(height=400)

    # Zero line marks perfect prediction — boxes above it mean the model underestimated.
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black', strokeDash=[4, 4]
    ).encode(y='y:Q')

    st.altair_chart(box_makes + zero_line, use_container_width=True)
else:
    st.warning("Residual data not found.")

st.divider()

# error by price tier
st.subheader("3. Prediction Accuracy by Price Tier")
st.markdown(
    "Breaks the test set into price tiers and shows the median absolute percentage error within each band. "
    "Models typically struggle at the extremes — very cheap cars often have quirky auction dynamics, "
    "while very expensive cars are rare enough that the training data thins out. "
    "A lower bar means the model is more reliable in that price range."
)

if not df_residuals.empty:
    # Bin into intuitive price tiers based on the Cars & Bids price distribution.
    tier_bins = [0, 15_000, 30_000, 60_000, 100_000, float('inf')]
    tier_labels = ['Under $15K', '$15K–$30K', '$30K–$60K', '$60K–$100K', 'Over $100K']
    df_residuals['Price Tier'] = pd.cut(
        df_residuals['Sold_Price'], bins=tier_bins, labels=tier_labels
    )

    tier_summary = df_residuals.groupby('Price Tier', observed=True)['Abs_Pct_Error'].agg(
        median_mape='median', count='count'
    ).reset_index()

    bar_tier = alt.Chart(tier_summary).mark_bar(color='#C4895A').encode(
        x=alt.X('Price Tier:N', sort=tier_labels, title=''),
        y=alt.Y('median_mape:Q', title='Median Absolute % Error'),
        tooltip=[
            alt.Tooltip('Price Tier:N', title='Tier'),
            alt.Tooltip('median_mape:Q', format='.1f', title='Median Abs % Error'),
            alt.Tooltip('count:Q', title='Test Set Cars')
        ]
    ).properties(height=350)

    st.altair_chart(bar_tier, use_container_width=True)
else:
    st.warning("Residual data not found.")

st.divider()

# feature importance
st.subheader("4. Feature Importance (SHAP Values)")
st.markdown(
    "Bar length reflects each variables's average absolute SHAP value across the test set. SHAP is a model-agnostic measure of "
    "how much a given input shifts the predicted price. Longer bars indicate variables the model "
    "consistently relies on to distinguish high vs low value vehicles. This does not indicate directionality, so "
    "a feature ranked highly may push prices up or down depending on its value. Features near the bottom contribute "
    "minimally to the prediction and may be candidates for removal in future iterations."
)

if not df_shap.empty:
    # mean_abs_shap is the average of |SHAP value| across all test samples for each feature.
    # It measures how much that feature moves the prediction on average, regardless of direction
    # (a feature that sometimes adds $5k and sometimes subtracts $5k still has a large mean abs SHAP).
    fig_shap = alt.Chart(df_shap).mark_bar(color='#C4895A').encode(
        x=alt.X('mean_abs_shap:Q', title='Mean Absolute SHAP Value (log $)'),
        y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=400)),
        tooltip=['feature', alt.Tooltip('mean_abs_shap:Q', format='.4f', title='Importance')]
    )
    st.altair_chart(fig_shap, use_container_width=True)
else:
    st.warning("SHAP importance data not found. Run the SHAP cell in model_insights.ipynb and export shap_importance.csv.")

st.divider()

# partial dependence
st.subheader("5. Marginal Price Effects (Partial Dependence)")
st.markdown(
    "Partial dependence plots isolate the relationship between a single feature and the predicted price by averaging "
    "out the influence of all other variables. Select a feature below to see how the model's output changes as that "
    "input varies across its observed range, holding all else equal. Steep slopes indicate high sensitivity while flat "
    "regions suggest the model treats that range as largely price-neutral. Note that interactions between features "
    "are not captured here, you can use SHAP values above for a more complete picture."
)

if not df_pdp.empty:
    # PDP shows the marginal effect of one variable by averaging out all others —
    # it answers "if only this input changed, how would the predicted price move?"
    # SHAP above answers a different question: "how much did each variable actually
    # contribute to this specific prediction, accounting for all feature interactions?"
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
