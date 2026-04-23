import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import os

API_URL = st.secrets["API_URL"]

# page config and custom CSS
st.set_page_config(page_title="Price Predictor", page_icon="🚗", layout="wide")

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

# data loading
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

# session state defaults
# Simple, static inputs use key= so Streamlit manages their state automatically.
# Cascading dropdowns (make → model → year → trim → ...) are managed manually
# via session_state reads/writes instead, because Streamlit throws a ValueError
# if the widget's key holds a value that no longer exists in the new options list
# (e.g., switching make would leave a stale model stored under key=).
_defaults = {
    'pred_make': 'Porsche',
    'pred_model': '996 911',
    'pred_year': 2002,
    'pred_trim': 'unknown',
    'pred_mileage': 50000,
    'pred_state': 'AZ',
    'pred_drivetrain': 'Rear-wheel drive',
    'pred_body_style': 'Coupe',
    'pred_transmission': 'Manual',
    'pred_engine_cyl': '',
    'pred_displacement': 3.0,
    'pred_gears': 6,
    # Static selectboxes
    'pred_exterior_color': 'Beige',
    'pred_interior_color': 'Black',
    'pred_title_status': 'Clean',
    'pred_seller_type': 'Private Party',
    # Text areas — plain session state keys (not widget keys) so Streamlit never clears them
    'pred_highlights': '',
    'pred_equipment': '',
    'pred_flaws': '',
    'pred_modifications': '',
    'pred_service_history': '',
    'pred_ownership_history': '',
    'pred_included_items': '',
    'pred_seller_notes': '',
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# header and intro copy
st.title("🚗 Car Price Predictor")
st.markdown(
    "This tool uses a machine learning model trained on thousands of historical Cars & Bids auction results to estimate "
    "what a vehicle is likely to sell for at auction. I built this tool for two primary use cases:"
)
st.markdown(
    "- **Evaluating an active listing:** If a car is currently up for auction, you can enter its specs and "
    "paste them in the listing text below. The predicted price and historical averages will help you assess whether the "
    "current bid represents fair value, a potential deal, or an overpriced vehicle.\n"
    "- **Exploring a car's market value:** Even without an active listing, you can enter any vehicle's specifications "
    "to get a price estimate and understand how factors like mileage, trim, transmission, and condition affect value."
)
st.divider()

# vehicle specs form
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

# spec_df must be created here, outside of any column block, so that col2 and col3
# can filter it further as the user selects drivetrain, body style, transmission, etc.
# If this were inside the col1 block, it would be out of scope when col2/col3 render.
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

    mileage = st.number_input("Mileage", min_value=0, step=500, key='pred_mileage')
    state = st.text_input("State Registered (e.g. AZ, CA)", max_chars=2, key='pred_state')

with col2:
    ext_colors = ['Black', 'White', 'Gray', 'Silver', 'Red', 'Blue', 'Green', 'Brown', 'Beige', 'Yellow', 'Orange', 'Purple', 'Other']
    saved_ext = st.session_state['pred_exterior_color']
    exterior_color = st.selectbox("Exterior Color", ext_colors, index=ext_colors.index(saved_ext) if saved_ext in ext_colors else 8)
    st.session_state['pred_exterior_color'] = exterior_color

    int_colors = ['Black', 'Beige', 'Gray', 'Brown', 'Red', 'White', 'Blue', 'Other']
    saved_int = st.session_state['pred_interior_color']
    interior_color = st.selectbox("Interior Color", int_colors, index=int_colors.index(saved_int) if saved_int in int_colors else 0)
    st.session_state['pred_interior_color'] = interior_color

    title_opts = ["Clean", "Rebuilt/Salvage", "Mileage Issue", "Buyback", "Alternate Doc", "Other", "Unknown"]
    saved_title = st.session_state['pred_title_status']
    title_status = st.selectbox("Title Status", title_opts, index=title_opts.index(saved_title) if saved_title in title_opts else 0)
    st.session_state['pred_title_status'] = title_status

    seller_opts = ["Private Party", "Dealer", "Other"]
    saved_seller = st.session_state['pred_seller_type']
    seller_type = st.selectbox("Seller Type", seller_opts, index=seller_opts.index(saved_seller) if saved_seller in seller_opts else 0)
    st.session_state['pred_seller_type'] = seller_type

    drivetrains = sorted(spec_df['Drivetrain'].dropna().unique().tolist()) if not spec_df.empty and 'Drivetrain' in spec_df.columns else []
    if not drivetrains: drivetrains = ["Rear-wheel drive", "4WD/AWD", "Front-wheel drive"]
    saved_dt = st.session_state['pred_drivetrain']
    dt_idx = drivetrains.index(saved_dt) if saved_dt in drivetrains else (drivetrains.index('Rear-wheel drive') if 'Rear-wheel drive' in drivetrains else 0)
    drivetrain = st.selectbox("Drivetrain", drivetrains, index=dt_idx)
    st.session_state['pred_drivetrain'] = drivetrain

    if not spec_df.empty and 'Drivetrain' in spec_df.columns:
        spec_df = spec_df[spec_df['Drivetrain'] == drivetrain]

with col3:
    body_styles = sorted(spec_df['Body Style'].dropna().unique().tolist()) if not spec_df.empty and 'Body Style' in spec_df.columns else []
    if not body_styles: body_styles = ["Convertible", "Coupe", "Hatchback", "SUV/Crossover", "Sedan", "Truck", "Van/Minivan", "Wagon"]
    saved_bs = st.session_state['pred_body_style']
    bs_idx = body_styles.index(saved_bs) if saved_bs in body_styles else (body_styles.index('Coupe') if 'Coupe' in body_styles else 0)
    body_style = st.selectbox("Body Style", body_styles, index=bs_idx)
    st.session_state['pred_body_style'] = body_style

    if not spec_df.empty and 'Body Style' in spec_df.columns:
        spec_df = spec_df[spec_df['Body Style'] == body_style]

    transmissions = sorted(spec_df['Transmission_Type'].dropna().unique().tolist()) if not spec_df.empty and 'Transmission_Type' in spec_df.columns else []
    if not transmissions: transmissions = ["Automatic", "Manual", "Other"]
    saved_tx = st.session_state['pred_transmission']
    tx_idx = transmissions.index(saved_tx) if saved_tx in transmissions else (transmissions.index('Manual') if 'Manual' in transmissions else 0)
    transmission = st.selectbox("Transmission", transmissions, index=tx_idx)
    st.session_state['pred_transmission'] = transmission

    if not spec_df.empty and 'Transmission_Type' in spec_df.columns:
        spec_df = spec_df[spec_df['Transmission_Type'] == transmission]

    engine_cyls = sorted(spec_df['Engine_Cylinders'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns else []
    if not engine_cyls: engine_cyls = ["I4", "I6", "V6", "V8", "V10", "V12", "H4", "H6", "Electric", "Rotary", "Other", "Unknown"]
    saved_cyl = st.session_state['pred_engine_cyl']
    cyl_idx = engine_cyls.index(saved_cyl) if saved_cyl in engine_cyls else 0
    engine_cyl = st.selectbox("Cylinders", engine_cyls, index=cyl_idx)
    st.session_state['pred_engine_cyl'] = engine_cyl

    if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns:
        spec_df = spec_df[spec_df['Engine_Cylinders'] == engine_cyl]

    disp_opts = sorted(spec_df['Engine_Displacement_L'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Displacement_L' in spec_df.columns else []
    if disp_opts:
        saved_disp = st.session_state['pred_displacement']
        disp_idx = disp_opts.index(saved_disp) if saved_disp in disp_opts else 0
        displacement = st.selectbox("Engine Displacement (L)", disp_opts, index=disp_idx)
        if not spec_df.empty:
            spec_df = spec_df[spec_df['Engine_Displacement_L'] == displacement]
    else:
        displacement = st.number_input("Engine Displacement (L)", min_value=0.0, max_value=10.0, value=float(st.session_state['pred_displacement']), step=0.1, key='pred_displacement_input')
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

# auction text form
st.subheader("📝 2. Auction Description")
st.markdown(
    "These fields mirror the exact text sections used in every Cars & Bids listing. If you are evaluating an active auction, just "
    "paste the corresponding text directly from the listing page for the most accurate estimate. The machine learning model will parse each "
    "section for condition signals (e.g., modifications, known flaws, service history, and more) that meaningfully influence price. "
    "Typing a few casual words will probably have little effect because the model was trained on full listing text and responds to specific "
    "keywords and phrases (e.g., 'one owner', 'california car', '2 keys', 'emissions'). If you are just exploring a vehicle "
    "generally, these fields can be left blank."
)

with st.expander("Click to expand and paste auction text blocks", expanded=True):
    text_col1, text_col2 = st.columns(2)
    with text_col1:
        highlights = st.text_area("Highlights", value=st.session_state['pred_highlights'])
        st.session_state['pred_highlights'] = highlights
        equipment = st.text_area("Equipment", value=st.session_state['pred_equipment'])
        st.session_state['pred_equipment'] = equipment
        flaws = st.text_area("Known Flaws", value=st.session_state['pred_flaws'])
        st.session_state['pred_flaws'] = flaws
        modifications = st.text_area("Modifications", value=st.session_state['pred_modifications'])
        st.session_state['pred_modifications'] = modifications
    with text_col2:
        service_history = st.text_area("Recent Service History", value=st.session_state['pred_service_history'])
        st.session_state['pred_service_history'] = service_history
        ownership_history = st.text_area("Ownership History", value=st.session_state['pred_ownership_history'])
        st.session_state['pred_ownership_history'] = ownership_history
        included_items = st.text_area("Other Items Included in Sale", value=st.session_state['pred_included_items'])
        st.session_state['pred_included_items'] = included_items
        seller_notes = st.text_area("Seller Notes", value=st.session_state['pred_seller_notes'])
        st.session_state['pred_seller_notes'] = seller_notes

st.markdown("<br>", unsafe_allow_html=True)

# submission and API call
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
            # timeout=90 accommodates Lambda cold starts, which can take 30–60 s when the
            # function hasn't been invoked recently. A shorter timeout would false-fail on warm-up.
            response = requests.post(API_URL, json=payload, timeout=90)
            if response.status_code == 503:
                st.toast("Model is warming up — retrying...", icon="⏳")
                response = requests.post(API_URL, json=payload, timeout=90)

            price = 0

            if response.status_code == 200:
                result = response.json()
                if 'estimated_price' in result:
                    price = result['estimated_price']
                elif 'body' in result:
                    body_data = json.loads(result['body'])
                    price = body_data.get('estimated_price', 0)
                else:
                    st.error(f"Unexpected response format: {result}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")

            # historical comp lookup
            historical_avg = None
            historical_count = 0
            match_level = "No historical data found"
            applied_conditions = []
            matching_cars_df = pd.DataFrame()

            if not df_history.empty:
                target_col = 'Sold_Price'

                # Fallback order goes from most specific to least specific so the user always
                # gets the tightest available comp set — only widening the search when no
                # records match the more precise criteria.
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

            # Storing in session_state means the prediction output survives if the user
            # navigates away and returns, without needing to re-submit the form.
            if price > 0:
                st.session_state['pred_results'] = {
                    'price': price,
                    'historical_avg': historical_avg,
                    'historical_count': historical_count,
                    'match_level': match_level,
                    'applied_conditions': applied_conditions,
                    'matching_cars_df': matching_cars_df,
                }

        except Exception as e:
            st.error(f"Connection failed: {e}")

# results display
if 'pred_results' in st.session_state:
    r = st.session_state['pred_results']
    price = r['price']
    historical_avg = r['historical_avg']
    historical_count = r['historical_count']
    match_level = r['match_level']
    applied_conditions = r['applied_conditions']
    matching_cars_df = r['matching_cars_df']

    st.success("Analysis Complete!")

    avg_val = f"${historical_avg:,.0f}" if historical_avg is not None else "N/A"
    count_val = f"{historical_count} cars" if historical_count > 0 else "0 cars"
    help_tip = f"Matched on: {match_level}" if historical_avg is not None else "No matching historical data found."
    st.markdown(f"""
    <div style="display:flex; gap:16px; align-items:stretch;">
        <div style="flex:2; background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:28px 32px; display:flex; flex-direction:column; justify-content:center;">
            <p style="font-size:0.875rem; color:#666; font-weight:500; margin:0 0 8px 0;">Predicted Price</p>
            <p style="font-size:3rem; font-weight:700; color:#1a1a1a; margin:0; line-height:1.1;">${price:,.0f}</p>
        </div>
        <div style="flex:1; display:flex; flex-direction:column; gap:10px;">
            <div style="flex:1; background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:14px 20px; display:flex; flex-direction:column; justify-content:center;" title="{help_tip}">
                <p style="font-size:0.8rem; color:#666; font-weight:500; margin:0 0 4px 0;">Historical Average</p>
                <p style="font-size:1.5rem; font-weight:700; color:#1a1a1a; margin:0;">{avg_val}</p>
            </div>
            <div style="flex:1; background-color:#EDE8DF; border:1px solid #C4A882; border-radius:10px; padding:14px 20px; display:flex; flex-direction:column; justify-content:center;">
                <p style="font-size:0.8rem; color:#666; font-weight:500; margin:0 0 4px 0;">Historical Sample Size</p>
                <p style="font-size:1.5rem; font-weight:700; color:#1a1a1a; margin:0;">{count_val}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if historical_avg is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        filter_text = " | ".join([f"**{col}**: {val}" for col, val in applied_conditions])
        st.info(f"**Historical average calculated using ({historical_count} matches):** {filter_text}")

        st.subheader("Historical Sales Data")

        display_cols = ['URL', 'Make', 'Model', 'Year', 'Mileage', 'Exterior Color', 'Transmission_Type', 'Drivetrain', 'Body Style', 'Sold_Price']
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
        st.info("**Historical average unable to be calculated**")
