import streamlit as st
import requests
import json
import pandas as pd
import os

# --- CONFIGURATION ---
API_URL = "https://r0fo8f5io3.execute-api.us-west-2.amazonaws.com/default/CarPriceApp"

st.set_page_config(page_title="carsandbids.com: Classic Car Price Predictor", page_icon="🚗", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_car_data():
    # Path if running from inside the 'frontend' folder locally
    file_path = "../data/dropdown_options.csv"
    
    # Fallback path if Streamlit Cloud runs from the root folder
    if not os.path.exists(file_path):
        file_path = "data/dropdown_options.csv"
        
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Could not find the options data at {file_path}. Please check your file paths.")
        return pd.DataFrame() 

df_cars = load_car_data()

st.title("🚗 Classic Car Price Estimator")
st.write("Enter the vehicle specifications and paste the auction text below.")

# --- UI COMPONENTS ---
st.subheader("1. Vehicle Specifications")
col1, col2, col3 = st.columns(3)

with col1:
    if not df_cars.empty and 'Make' in df_cars.columns:
        makes = sorted(df_cars['Make'].dropna().unique().tolist())
        default_make_idx = makes.index("Porsche") if "Porsche" in makes else 0
        make = st.selectbox("Make", makes, index=default_make_idx)
        
        models = sorted(df_cars[df_cars['Make'] == make]['Model'].dropna().unique().tolist())
        model = st.selectbox("Model", models)
    else:
        # Fallbacks if CSV fails to load
        make = st.text_input("Make", value="Porsche") 
        model = st.text_input("Model", value="996 911")

    year = st.number_input("Year", min_value=1900, max_value=2025, value=2015)
    mileage = st.number_input("Mileage", min_value=0, value=50000, step=500)
    state = st.text_input("State Registered (e.g. AZ, CA)", max_chars=2, value="AZ")

# Filter dataframe for dynamic dropdowns
if not df_cars.empty:
    spec_df = df_cars[(df_cars['Make'] == make) & (df_cars['Model'] == model)]
else:
    spec_df = pd.DataFrame()

with col2:
    exterior_color = st.selectbox("Exterior Color", ['Black', 'White', 'Gray', 'Silver', 'Red', 'Blue', 'Green', 'Brown', 'Beige', 'Yellow', 'Orange', 'Purple', 'Other'])
    interior_color = st.selectbox("Interior Color", ['Black', 'Beige', 'Gray', 'Brown', 'Red', 'White', 'Blue', 'Other'])
    title_status = st.selectbox("Title Status", ["Clean", "Rebuilt/Salvage", "Mileage Issue", "Buyback", "Alternate Doc", "Other", "Unknown"])
    seller_type = st.selectbox("Seller Type", ["Private Party", "Dealer", "Other"])
    
    # Dynamic Drivetrain
    drivetrains = sorted(spec_df['Drivetrain'].dropna().unique().tolist()) if not spec_df.empty and 'Drivetrain' in spec_df.columns else []
    if not drivetrains: drivetrains = ["Rear-wheel drive", "4WD/AWD", "Front-wheel drive"]
    drivetrain = st.selectbox("Drivetrain", drivetrains)

with col3:
    # Dynamic Body Style
    body_styles = sorted(spec_df['Body Style'].dropna().unique().tolist()) if not spec_df.empty and 'Body Style' in spec_df.columns else []
    if not body_styles: body_styles = ["Convertible", "Coupe", "Hatchback", "SUV/Crossover", "Sedan", "Truck", "Van/Minivan", "Wagon"]
    body_style = st.selectbox("Body Style", body_styles)
    
    # Dynamic Transmission
    transmissions = sorted(spec_df['Transmission_Type'].dropna().unique().tolist()) if not spec_df.empty and 'Transmission_Type' in spec_df.columns else []
    if not transmissions: transmissions = ["Automatic", "Manual", "Other"]
    transmission = st.selectbox("Transmission", transmissions)
    
    # Dynamic Cylinders
    engine_cyls = sorted(spec_df['Engine_Cylinders'].dropna().unique().tolist()) if not spec_df.empty and 'Engine_Cylinders' in spec_df.columns else []
    if not engine_cyls: engine_cyls = ["I4", "I6", "V6", "V8", "V10", "V12", "H4", "H6", "Electric", "Rotary", "Other", "Unknown"]
    engine_cyl = st.selectbox("Cylinders", engine_cyls)

    gears = st.slider("Gears", 1, 10, 6)
    displacement = st.number_input("Engine Displacement (L) [0 for EV]", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

st.markdown("---")
st.subheader("2. Auction Description (Raw Text)")
st.caption("Paste the exact text from the auction listing. The AI will automatically extract features like mods, flaws, and condition indicators.")

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

# --- SUBMISSION LOGIC ---
submitted = st.button("💰 Predict Market Price", use_container_width=True)

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

    st.info("Analyzing text and generating prediction...")
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if 'estimated_price' in result:
                price = result['estimated_price']
            elif 'body' in result:
                body_data = json.loads(result['body'])
                price = body_data.get('estimated_price', 0)
            else:
                price = 0
                st.error(f"Unexpected response format: {result}")

            if price > 0:
                st.success(f"## Estimated Price: ${price:,.2f}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        st.error(f"Connection failed: {e}")