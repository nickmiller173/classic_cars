import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import os

# --- CONFIGURATION ---
API_URL = "https://r0fo8f5io3.execute-api.us-west-2.amazonaws.com/default/CarPriceApp"

st.set_page_config(page_title="carsandbids.com: Classic Car Price Predictor", page_icon="🚗", layout="wide")

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
            default_make_idx = makes.index("Porsche") if "Porsche" in makes else 0
            make = st.selectbox("Make", makes, index=default_make_idx)
            
            models = sorted(df_cars[df_cars['Make'] == make]['Model'].dropna().unique().tolist())
            default_model_idx = models.index("996 911") if "996 911" in models else 0
            model = st.selectbox("Model", models, index=default_model_idx)
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
            default_year_idx = years.index(2002) if 2002 in years else 0
            year = st.selectbox("Year", years, index=default_year_idx)
        else:
            year = st.number_input("Year", min_value=1925, max_value=2025, value=2002)

        # 1. CASCADE: Filter by Year
        if not spec_df.empty and 'Year' in spec_df.columns:
            spec_df = spec_df[spec_df['Year'] == year]

        mileage = st.number_input("Mileage", min_value=0, value=50000, step=500)
        state = st.text_input("State Registered (e.g. AZ, CA)", max_chars=2, value="AZ")

    with col2:
        # Static UI Elements (Not filtered by specs)
        exterior_color = st.selectbox("Exterior Color", ['Black', 'White', 'Gray', 'Silver', 'Red', 'Blue', 'Green', 'Brown', 'Beige', 'Yellow', 'Orange', 'Purple', 'Other'])
        interior_color = st.selectbox("Interior Color", ['Black', 'Beige', 'Gray', 'Brown', 'Red', 'White', 'Blue', 'Other'])
        title_status = st.selectbox("Title Status", ["Clean", "Rebuilt/Salvage", "Mileage Issue", "Buyback", "Alternate Doc", "Other", "Unknown"])
        seller_type = st.selectbox("Seller Type", ["Private Party", "Dealer", "Other"])
        
        # Dynamic Elements
        drivetrains = sorted(spec_df['Drivetrain'].dropna().unique().tolist()) if not spec_df.empty and 'Drivetrain' in spec_df.columns else []
        if not drivetrains: drivetrains = ["Rear-wheel drive", "4WD/AWD", "Front-wheel drive"]
        drivetrain = st.selectbox("Drivetrain", drivetrains)

        # 2. CASCADE: Filter by Drivetrain
        if not spec_df.empty and 'Drivetrain' in spec_df.columns:
            spec_df = spec_df[spec_df['Drivetrain'] == drivetrain]

    with col3:
        body_styles = sorted(spec_df['Body Style'].dropna().unique().tolist()) if not spec_df.empty and 'Body Style' in spec_df.columns else []
        if not body_styles: body_styles = ["Convertible", "Coupe", "Hatchback", "SUV/Crossover", "Sedan", "Truck", "Van/Minivan", "Wagon"]
        body_style = st.selectbox("Body Style", body_styles)
        
        # 3. CASCADE: Filter by Body Style
        if not spec_df.empty and 'Body Style' in spec_df.columns:
            spec_df = spec_df[spec_df['Body Style'] == body_style]
        
        transmissions = sorted(spec_df['Transmission_Type'].dropna().unique().tolist()) if not spec_df.empty and 'Transmission_Type' in spec_df.columns else []
        if not transmissions: transmissions = ["Automatic", "Manual", "Other"]
        transmission = st.selectbox("Transmission", transmissions)

        # 4. CASCADE: Filter by Transmission
        if not spec_df.empty and 'Transmission_Type' in spec_df.columns:
            spec_df = spec_df[spec_df['Transmission_Type'] == transmission]
        
        engine_cyls = sorted(spec_df['Engine_Cylinders'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns else []
        if not engine_cyls: engine_cyls = ["I4", "I6", "V6", "V8", "V10", "V12", "H4", "H6", "Electric", "Rotary", "Other", "Unknown"]
        engine_cyl = st.selectbox("Cylinders", engine_cyls)

        # 5. CASCADE: Filter by Cylinders
        if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns:
            spec_df = spec_df[spec_df['Engine_Cylinders'] == engine_cyl]

        disp_opts = sorted(spec_df['Engine_Displacement_L'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Displacement_L' in spec_df.columns else []
        if disp_opts:
            displacement = st.selectbox("Engine Displacement (L) [0 for EV]", disp_opts)
            # 6. CASCADE: Filter by Displacement
            if not spec_df.empty:
                spec_df = spec_df[spec_df['Engine_Displacement_L'] == displacement]
        else:
            displacement = st.number_input("Engine Displacement (L) [0 for EV]", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

        gears_opts = sorted(spec_df['Gears'].dropna().unique().tolist()) if not spec_df.empty and 'Gears' in spec_df.columns else []
        if gears_opts:
            gears_opts = [int(g) for g in gears_opts]
            gears = st.selectbox("Gears", gears_opts)
        else:
            gears = st.slider("Gears", 1, 10, 6)
    
    st.divider()

    # --- AUCTION TEXT ---
    st.subheader("📝 2. Auction Description")
    st.info("Paste the exact text from the listing. The app will extract features like mods, flaws, and condition indicators.")

    with st.expander("Click to expand and paste auction text blocks", expanded=True):
        text_col1, text_col2 = st.columns(2)
        with text_col1:
            highlights = st.text_area("Highlights")
            equipment = st.text_area("Equipment")
            flaws = st.text_area("Known Flaws")
            modifications = st.text_area("Modifications")
        with text_col2:
            service_history = st.text_area("Recent Service History")
            ownership_history = st.text_area("Ownership History")
            included_items = st.text_area("Other Items Included in Sale")
            seller_notes = st.text_area("Seller Notes")

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
            "Seller Notes": seller_notes
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
                        {"name": "Make, Model, Year, Trans, Color", "conditions": [('Make', make), ('Model', model), ('Year', year), ('Transmission_Type', transmission), ('Exterior Color', exterior_color)]},
                        {"name": "Make, Model, Year, Trans", "conditions": [('Make', make), ('Model', model), ('Year', year), ('Transmission_Type', transmission)]},
                        {"name": "Make, Model, Year", "conditions": [('Make', make), ('Model', model), ('Year', year)]},
                        {"name": "Make & Model", "conditions": [('Make', make), ('Model', model)]},
                        {"name": "Make Only", "conditions": [('Make', make)]}
                    ]
                    
                    for f in fallback_filters:
                        temp_df = df_history.copy()
                        
                        # Apply filters dynamically
                        for col_name, val in f["conditions"]:
                            if col_name in temp_df.columns:
                                temp_df = temp_df[temp_df[col_name] == val]
                        
                        # If we found matches, calculate the average and break the loop
                        if not temp_df.empty and target_col in temp_df.columns:
                            historical_avg = temp_df[target_col].mean()
                            historical_count = len(temp_df)
                            match_level = f["name"]
                            applied_conditions = f["conditions"]
                            matching_cars_df = temp_df 
                            break

                # --- 3. Display Results Side-by-Side ---
                if price > 0:
                    st.success("Analysis Complete!")
                    
                    # Create columns for the metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(label="Estimated Auction Value (AI)", value=f"${price:,.2f}")
                        
                    with metric_col2:
                        if historical_avg is not None:
                            st.metric(
                                label=f"Historical Average", 
                                value=f"${historical_avg:,.2f}", 
                                help=f"Based on historical data matching: {match_level}. NOTE: If the dataset doesn't have a car that matches the exact combination of inputs, the app relies on a cascading list of fallback filters."
                            )
                        else:
                            st.metric(label="Historical Average", value="N/A", help="No matching historical data found.")
                            
                    with metric_col3:
                        if historical_count > 0:
                            st.metric(
                                label="Historical Sample Size", 
                                value=f"{historical_count} cars",
                                help="The number of past sales used to calculate the historical average."
                            )
                        else:
                            st.metric(label="Historical Sample Size", value="0 cars")

                    # --- 4. Print the exact filters used ---
                    if historical_avg is not None:
                        filter_text = " | ".join([f"**{col}**: {val}" for col, val in applied_conditions])
                        st.info(f"**Historical average calculated using ({historical_count} matches):** {filter_text}")
                        
                        # --- 5. Display the Table of Matching Cars ---
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
    st.markdown("The top features influencing auction prices across the entire dataset, ranked by their average absolute SHAP value.")

    if not df_shap.empty:
        fig_shap = alt.Chart(df_shap).mark_bar(color='#00bfa5').encode(
            x=alt.X('mean_abs_shap:Q', title='Mean Absolute SHAP Value (log $)'),
            y=alt.Y('feature:N', sort='-x', title=None),
            tooltip=['feature', alt.Tooltip('mean_abs_shap:Q', format='.4f', title='Importance')]
        )
        st.altair_chart(fig_shap, use_container_width=True)
    else:
        st.warning("SHAP importance data not found. Run the SHAP cell in model_insights.ipynb and export shap_importance.csv.")

    st.divider()

    # --- IDEA 3: RESIDUAL MATRIX ---
    st.subheader("2. Market Hype vs. Value (Residual Analysis)")
    st.markdown("Compare actual auction hammer prices against the model's prediction. Cars far above the line represent 'Bidding Wars' (Hype), while cars below the line represent 'Well Bought' deals (Value).")
    
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

    # --- EXISTING PDP PLOTS ---
    st.subheader("3. Partial Dependency Insights (Macro Trends)")
    st.markdown("Explore how individual vehicle attributes impact the model's estimated price across the entire market, assuming all other variables remain constant.")
    
    if not df_pdp.empty:
        features = sorted(df_pdp['Feature'].unique())
        selected_feature = st.selectbox("Select a Variable to Analyze:", features)
        
        feature_data = df_pdp[df_pdp['Feature'] == selected_feature]
        
        fig_pdp = alt.Chart(feature_data).mark_line(color='#00bfa5', point=True).encode(
            x=alt.X('Feature_Value:Q', title=selected_feature),
            y=alt.Y('Predicted_Price:Q', title='Estimated Price ($)', axis=alt.Axis(format='$,.0f')),
            tooltip=[alt.Tooltip('Feature_Value:Q', title=selected_feature),
                     alt.Tooltip('Predicted_Price:Q', format='$,.0f', title='Est. Price')]
        )
        st.altair_chart(fig_pdp, use_container_width=True)
    else:
        st.warning("PDP data not found. Please run the PDP generation script.")