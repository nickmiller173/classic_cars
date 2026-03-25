import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import os

# --- CONFIGURATION ---
API_URL = "https://r0fo8f5io3.execute-api.us-west-2.amazonaws.com/default/CarPriceApp"

st.set_page_config(page_title="carsandbids.com: Classic Car Price Predictor", page_icon="🚗", layout="wide")

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
.streamlit-expanderHeader {
    background-color: #EDE8DF;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_car_data():
    file_path = "../data/frontend_data/dropdown_options.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/dropdown_options.csv"

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Could not find the options data at {file_path}. Please check your file paths.")
        return pd.DataFrame()

df_cars = load_car_data()

@st.cache_data
def load_historical_averages():
    file_path = "../data/frontend_data/historical_averages_lookup.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/historical_averages_lookup.csv"

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df_history = load_historical_averages()

@st.cache_data
def load_pdp_data():
    file_path = "../data/frontend_data/pdp_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/pdp_data.csv"

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df_pdp = load_pdp_data()

@st.cache_data
def load_residual_data():
    file_path = "../data/frontend_data/residual_data.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/residual_data.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df_residuals = load_residual_data()

@st.cache_data
def load_shap_importance():
    file_path = "../data/frontend_data/shap_importance.csv"
    if not os.path.exists(file_path):
        file_path = "data/frontend_data/shap_importance.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df_shap = load_shap_importance()

# --- SESSION STATE DEFAULTS ---
# Simple inputs use key= directly. Cascading dropdowns are managed manually
# because their options list changes and Streamlit errors if a stored value
# isn't in the new list.
_defaults = {
    'pred_make': 'Porsche',
    'pred_model': '996 911',
    'pred_year': 2002,
    'pred_trim': 'unknown',
    'pred_mileage': 50000,
    'pred_drivetrain': 'Rear-wheel drive',
    'pred_body_style': 'Coupe',
    'pred_transmission': 'Manual',
    'pred_engine_cyl': '',
    'pred_displacement': 3.0,
    'pred_gears': 6,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- UI HEADER ---
st.title("carsandbids.com 🚗 Classic Car Price Estimator")
st.markdown("Instantly estimate the auction value of a classic car based on its specs and historical condition reports.")
st.divider()

# --- TABS SETUP ---
tab1, tab2 = st.tabs(["Price Predictor", "ML Model Insights"])

with tab1:
    # --- VEHICLE SPECS ---
    st.subheader("🛠️ 1. Vehicle Specifications")
    col1, col2, col3 = st.columns(3)

    with col1:
        if not df_cars.empty and 'Make' in df_cars.columns:
            makes = sorted(df_cars['Make'].dropna().unique().tolist())
            saved_make = st.session_state['pred_make']
            make_idx = makes.index(saved_make) if saved_make in makes else (makes.index('Porsche') if 'Porsche' in makes else 0)
            make = st.selectbox("Make", makes, index=make_idx)
            st.session_state['pred_make'] = make

            models = sorted(df_cars[df_cars['Make'] == make]['Model'].dropna().unique().tolist())
            saved_model = st.session_state['pred_model']
            model_idx = models.index(saved_model) if saved_model in models else 0
            model = st.selectbox("Model", models, index=model_idx)
            st.session_state['pred_model'] = model
        else:
            make = st.text_input("Make", value="Porsche")
            model = st.text_input("Model", value="996 911")

    # Instantiate spec_df early
    if not df_cars.empty:
        spec_df = df_cars[(df_cars['Make'] == make) & (df_cars['Model'] == model)]
    else:
        spec_df = pd.DataFrame()

    with col1:
        years = sorted(spec_df['Year'].dropna().unique().tolist(), reverse=True) if not spec_df.empty and 'Year' in spec_df.columns else []
        if years:
            years = [int(y) for y in years]
            saved_year = st.session_state['pred_year']
            year_idx = years.index(saved_year) if saved_year in years else 0
            year = st.selectbox("Year", years, index=year_idx)
        else:
            year = st.number_input("Year", min_value=1925, max_value=2025, value=st.session_state['pred_year'])
        st.session_state['pred_year'] = year

        # 1. CASCADE: Filter by Year
        if not spec_df.empty and 'Year' in spec_df.columns:
            spec_df = spec_df[spec_df['Year'] == year]

        trims = sorted(spec_df['trim_slug'].dropna().unique().tolist()) if not spec_df.empty and 'trim_slug' in spec_df.columns else []
        if not trims:
            trims = ['unknown']
        if 'unknown' not in trims:
            trims = ['unknown'] + trims
        saved_trim = st.session_state['pred_trim']
        trim_idx = trims.index(saved_trim) if saved_trim in trims else 0
        trim_slug = st.selectbox("Trim", trims, index=trim_idx, format_func=lambda x: 'Unknown / Base' if x == 'unknown' else x.replace('-', ' ').title())
        st.session_state['pred_trim'] = trim_slug

        mileage = st.number_input("Mileage", min_value=0, value=st.session_state['pred_mileage'], step=500, key='pred_mileage')
        state = st.text_input("State Registered (e.g. AZ, CA)", max_chars=2, key='pred_state', value=st.session_state.get('pred_state', 'AZ'))

    with col2:
        exterior_color = st.selectbox("Exterior Color", ['Black', 'White', 'Gray', 'Silver', 'Red', 'Blue', 'Green', 'Brown', 'Beige', 'Yellow', 'Orange', 'Purple', 'Other'], index=8, key='pred_exterior_color')
        interior_color = st.selectbox("Interior Color", ['Black', 'Beige', 'Gray', 'Brown', 'Red', 'White', 'Blue', 'Other'], key='pred_interior_color')
        title_status = st.selectbox("Title Status", ["Clean", "Rebuilt/Salvage", "Mileage Issue", "Buyback", "Alternate Doc", "Other", "Unknown"], key='pred_title_status')
        seller_type = st.selectbox("Seller Type", ["Private Party", "Dealer", "Other"], key='pred_seller_type')

        drivetrains = sorted(spec_df['Drivetrain'].dropna().unique().tolist()) if not spec_df.empty and 'Drivetrain' in spec_df.columns else []
        if not drivetrains: drivetrains = ["Rear-wheel drive", "4WD/AWD", "Front-wheel drive"]
        saved_dt = st.session_state['pred_drivetrain']
        dt_idx = drivetrains.index(saved_dt) if saved_dt in drivetrains else (drivetrains.index('Rear-wheel drive') if 'Rear-wheel drive' in drivetrains else 0)
        drivetrain = st.selectbox("Drivetrain", drivetrains, index=dt_idx)
        st.session_state['pred_drivetrain'] = drivetrain

        # 2. CASCADE: Filter by Drivetrain
        if not spec_df.empty and 'Drivetrain' in spec_df.columns:
            spec_df = spec_df[spec_df['Drivetrain'] == drivetrain]

    with col3:
        body_styles = sorted(spec_df['Body Style'].dropna().unique().tolist()) if not spec_df.empty and 'Body Style' in spec_df.columns else []
        if not body_styles: body_styles = ["Convertible", "Coupe", "Hatchback", "SUV/Crossover", "Sedan", "Truck", "Van/Minivan", "Wagon"]
        saved_bs = st.session_state['pred_body_style']
        bs_idx = body_styles.index(saved_bs) if saved_bs in body_styles else (body_styles.index('Coupe') if 'Coupe' in body_styles else 0)
        body_style = st.selectbox("Body Style", body_styles, index=bs_idx)
        st.session_state['pred_body_style'] = body_style

        # 3. CASCADE: Filter by Body Style
        if not spec_df.empty and 'Body Style' in spec_df.columns:
            spec_df = spec_df[spec_df['Body Style'] == body_style]

        transmissions = sorted(spec_df['Transmission_Type'].dropna().unique().tolist()) if not spec_df.empty and 'Transmission_Type' in spec_df.columns else []
        if not transmissions: transmissions = ["Automatic", "Manual", "Other"]
        saved_tx = st.session_state['pred_transmission']
        tx_idx = transmissions.index(saved_tx) if saved_tx in transmissions else (transmissions.index('Manual') if 'Manual' in transmissions else 0)
        transmission = st.selectbox("Transmission", transmissions, index=tx_idx)
        st.session_state['pred_transmission'] = transmission

        # 4. CASCADE: Filter by Transmission
        if not spec_df.empty and 'Transmission_Type' in spec_df.columns:
            spec_df = spec_df[spec_df['Transmission_Type'] == transmission]

        engine_cyls = sorted(spec_df['Engine_Cylinders'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns else []
        if not engine_cyls: engine_cyls = ["I4", "I6", "V6", "V8", "V10", "V12", "H4", "H6", "Electric", "Rotary", "Other", "Unknown"]
        saved_cyl = st.session_state['pred_engine_cyl']
        cyl_idx = engine_cyls.index(saved_cyl) if saved_cyl in engine_cyls else 0
        engine_cyl = st.selectbox("Cylinders", engine_cyls, index=cyl_idx)
        st.session_state['pred_engine_cyl'] = engine_cyl

        # 5. CASCADE: Filter by Cylinders
        if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns:
            spec_df = spec_df[spec_df['Engine_Cylinders'] == engine_cyl]

        disp_opts = sorted(spec_df['Engine_Displacement_L'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Displacement_L' in spec_df.columns else []
        if disp_opts:
            saved_disp = st.session_state['pred_displacement']
            disp_idx = disp_opts.index(saved_disp) if saved_disp in disp_opts else 0
            displacement = st.selectbox("Engine Displacement (L) [0 for EV]", disp_opts, index=disp_idx)
            # 6. CASCADE: Filter by Displacement
            if not spec_df.empty:
                spec_df = spec_df[spec_df['Engine_Displacement_L'] == displacement]
        else:
            displacement = st.number_input("Engine Displacement (L) [0 for EV]", min_value=0.0, max_value=10.0, value=float(st.session_state['pred_displacement']), step=0.1, key='pred_displacement_input')
        st.session_state['pred_displacement'] = displacement

        gears_opts = sorted(spec_df['Gears'].dropna().unique().tolist()) if not spec_df.empty and 'Gears' in spec_df.columns else []
        if gears_opts:
            gears_opts = [int(g) for g in gears_opts]
            saved_gears = st.session_state['pred_gears']
            gears_idx = gears_opts.index(saved_gears) if saved_gears in gears_opts else 0
            gears = st.selectbox("Gears", gears_opts, index=gears_idx)
        else:
            gears = st.slider("Gears", 1, 10, st.session_state['pred_gears'], key='pred_gears_slider')
        st.session_state['pred_gears'] = gears

    st.divider()

    # --- AUCTION TEXT ---
    st.subheader("📝 2. Auction Description")
    st.info("Paste the exact text from the listing. The app will extract features like mods, flaws, and condition indicators.")

    with st.expander("Click to expand and paste auction text blocks", expanded=True):
        text_col1, text_col2 = st.columns(2)
        with text_col1:
            highlights = st.text_area("Highlights", key='pred_highlights')
            equipment = st.text_area("Equipment", key='pred_equipment')
            flaws = st.text_area("Known Flaws", key='pred_flaws')
            modifications = st.text_area("Modifications", key='pred_modifications')
        with text_col2:
            service_history = st.text_area("Recent Service History", key='pred_service_history')
            ownership_history = st.text_area("Ownership History", key='pred_ownership_history')
            included_items = st.text_area("Other Items Included in Sale", key='pred_included_items')
            seller_notes = st.text_area("Seller Notes", key='pred_seller_notes')

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SUBMISSION LOGIC ---
    submitted = st.button("💰 Predict Market Price", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "Make": make.strip(),
            "Model": model.strip(),
            "Year": year,
            "Mileage": mileage,
            "State": state.strip().upper(),
            "Exterior Color": exterior_color,
            "Interior Color": interior_color,
            "Title Status": title_status,
            "Seller Type": seller_type,
            "Drivetrain": drivetrain,
            "Body Style": body_style,
            "Transmission_Type": transmission,
            "Engine_Cylinders": engine_cyl,
            "Gears": float(gears),
            "Engine_Displacement_L": float(displacement),
            "Highlights": highlights,
            "Equipment": equipment,
            "Known Flaws": flaws,
            "Modifications": modifications,
            "Recent Service History": service_history,
            "Ownership History": ownership_history,
            "Other Items Included in Sale": included_items,
            "Seller Notes": seller_notes,
            "trim_slug": trim_slug
        }

        with st.spinner("Analyzing data and communicating with AWS Lambda..."):
            try:
                # --- 1. Get Prediction from Lambda ---
                response = requests.post(API_URL, json=payload)
                price = 0

                if response.status_code == 200:
                    result = response.json()
                    st.session_state['last_prediction'] = result
                    if 'estimated_price' in result:
                        price = result['estimated_price']
                    elif 'body' in result:
                        body_data = json.loads(result['body'])
                        price = body_data.get('estimated_price', 0)
                    else:
                        st.error(f"Unexpected response format: {result}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

                # --- 2. Calculate Historical Average with Fallbacks ---
                historical_avg = None
                historical_count = 0
                match_level = "No historical data found"
                applied_conditions = []
                matching_cars_df = pd.DataFrame()

                if not df_history.empty:
                    target_col = 'Sold_Price'

                    fallback_filters = [
                        {"name": "Make, Model, Year, Trans", "conditions": [('Make', make), ('Model', model), ('Year', year), ('Transmission_Type', transmission)]},
                        {"name": "Make, Model, Year", "conditions": [('Make', make), ('Model', model), ('Year', year)]},
                        {"name": "Make & Model", "conditions": [('Make', make), ('Model', model)]},
                        {"name": "Make Only", "conditions": [('Make', make)]}
                    ]

                    for f in fallback_filters:
                        temp_df = df_history.copy()

                        for col_name, val in f["conditions"]:
                            if col_name in temp_df.columns:
                                temp_df = temp_df[temp_df[col_name] == val]

                        if not temp_df.empty and target_col in temp_df.columns:
                            historical_avg = temp_df[target_col].mean()
                            historical_count = len(temp_df)
                            match_level = f["name"]
                            applied_conditions = f["conditions"]
                            matching_cars_df = temp_df
                            break

                # --- 3. Display Results ---
                if price > 0:
                    st.success("Analysis Complete!")

                    metric_col1, metric_col2 = st.columns([2, 1])

                    with metric_col1:
                        st.markdown(f"""
                        <div style="background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:28px 32px;">
                            <p style="font-size:0.875rem; color:#666; font-weight:500; margin:0 0 8px 0;">Estimated Auction Value (AI)</p>
                            <p style="font-size:3rem; font-weight:700; color:#1a1a1a; margin:0; line-height:1.1;">${price:,.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with metric_col2:
                        avg_val = f"${historical_avg:,.0f}" if historical_avg is not None else "N/A"
                        count_val = f"{historical_count} cars" if historical_count > 0 else "0 cars"
                        help_tip = f"Matched on: {match_level}" if historical_avg is not None else "No matching historical data found."
                        st.markdown(f"""
                        <div style="background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:14px 20px; margin-bottom:10px;" title="{help_tip}">
                            <p style="font-size:0.8rem; color:#666; font-weight:500; margin:0 0 4px 0;">Historical Average</p>
                            <p style="font-size:1.5rem; font-weight:700; color:#1a1a1a; margin:0;">{avg_val}</p>
                        </div>
                        <div style="background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:14px 20px;">
                            <p style="font-size:0.8rem; color:#666; font-weight:500; margin:0 0 4px 0;">Historical Sample Size</p>
                            <p style="font-size:1.5rem; font-weight:700; color:#1a1a1a; margin:0;">{count_val}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if historical_avg is not None:
                        filter_text = " | ".join([f"**{col}**: {val}" for col, val in applied_conditions])
                        st.info(f"**Historical average calculated using ({historical_count} matches):** {filter_text}")

                        st.subheader("Historical Sales Data")

                        display_cols = ['URL', 'Make', 'Model', 'Year', 'Mileage', 'Exterior Color', 'Transmission_Type','Drivetrain', 'Body Style', 'Sold_Price']
                        available_cols = [c for c in display_cols if c in matching_cars_df.columns]

                        if available_cols:
                            st.dataframe(
                                matching_cars_df[available_cols],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "URL": st.column_config.LinkColumn("Listing Link", display_text="View Auction")
                                }
                            )
                        else:
                            st.dataframe(matching_cars_df, use_container_width=True, hide_index=True)

                    else:
                        st.info(f"**Historical average unable to be calculated**")

            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    st.title("📈 Model Insights")

    # --- SHAP GLOBAL FEATURE IMPORTANCE ---
    st.subheader("1. What Drives Price? (Global Feature Importance)")
    st.markdown("Think of this as the model's voting card — the longer the bar, the more that feature consistently swayed price estimates across thousands of auctions. It doesn't tell you which direction, just how much each factor matters overall.")

    if not df_shap.empty:
        fig_shap = alt.Chart(df_shap).mark_bar(color='#C4895A').encode(
            x=alt.X('mean_abs_shap:Q', title='Mean Absolute SHAP Value (log $)'),
            y=alt.Y('feature:N', sort='-x', title=None),
            tooltip=['feature', alt.Tooltip('mean_abs_shap:Q', format='.4f', title='Importance')]
        )
        st.altair_chart(fig_shap, use_container_width=True)
    else:
        st.warning("SHAP importance data not found. Run the SHAP cell in model_insights.ipynb and export shap_importance.csv.")

    st.divider()

    # --- RESIDUAL MATRIX ---
    st.subheader("2. Market Hype vs. Value (Residual Analysis)")
    st.markdown("Each dot is a real auction result. If a dot sits above the diagonal line, that car sold for more than the model expected — usually a bidding war or rare find. Below the line means the buyer likely got a deal.")

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

    # --- PDP PLOTS ---
    st.subheader("3. Partial Dependency Insights (Macro Trends)")
    st.markdown("This shows what happens to the estimated price when you slide just one variable up or down while everything else stays fixed — like a test drive for a single feature to see how the model reacts to it.")

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
