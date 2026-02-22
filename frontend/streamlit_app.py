import streamlit as st
import requests
import json

# --- CONFIGURATION ---
# PASTE YOUR API GATEWAY URL HERE (from the Lambda console trigger tab)
API_URL = "https://r0fo8f5io3.execute-api.us-west-2.amazonaws.com/default/CarPriceApp"

st.set_page_config(page_title="Classic Car Price Predictor", page_icon="🚗")

# --- HEADER ---
st.title("🚗 Classic Car Price Estimator")
st.write("Enter the details of the car below to get an AI-predicted market price.")

# --- INPUT FORM ---
with st.form("car_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        make = st.selectbox("Make", ["Porsche", "BMW", "Ferrari", "Toyota", "Ford", "Chevrolet", "Other"])
        model = st.text_input("Model (e.g. 911, M3, Supra)")
        mileage = st.number_input("Mileage", min_value=0, value=50000, step=500)
        year = st.number_input("Year", min_value=1950, max_value=2025, value=2015) # Optional if your model uses it
        
        exterior_color = st.selectbox("Exterior Color", 
            ['Black', 'White', 'Gray', 'Silver', 'Red', 'Blue', 'Green', 'Brown', 'Beige', 'Yellow', 'Orange', 'Purple'])
        interior_color = st.selectbox("Interior Color", 
            ['Black', 'Beige', 'Gray', 'Brown', 'Red', 'White', 'Blue'])

    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        gears = st.slider("Gears", 3, 10, 6)
        
        engine_cyl = st.selectbox("Cylinders", ["I4", "I6", "V6", "V8", "V10", "V12", "Flat-6"])
        displacement = st.number_input("Engine Displacement (L)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
        
        drivetrain = st.selectbox("Drivetrain", ["RWD", "AWD", "FWD", "4WD"])
        body_style = st.selectbox("Body Style", ["Coupe", "Convertible", "Sedan", "Hatchback", "Wagon", "SUV"])
        
        seller_type = st.selectbox("Seller Type", ["Private Party", "Dealer"])
        title_status = st.selectbox("Title Status", ["Clean", "Salvage", "Rebuilt"])

    # Submit Button
    submitted = st.form_submit_button("💰 Predict Price")

# --- PREDICTION LOGIC ---
if submitted:
    # 1. Prepare Data Payload (Must match your training columns EXACTLY)
    payload = {
        "Make": make,
        "Model": model,
        "Mileage": str(mileage), # Sending as string to match your cleaning logic
        "Exterior Color": exterior_color,
        "Interior Color": interior_color,
        "Title Status": title_status,
        "Seller Type": seller_type,
        "Drivetrain": drivetrain,
        "Body Style": body_style,
        "Transmission_Type": transmission,
        "Gears": str(gears),
        "Engine_Displacement_L": str(displacement),
        "Engine_Cylinders": engine_cyl
        "Year": year, # Add this!
        "Highlights": "Paste from carsandbids.com",
        "Equipment": "Paste from carsandbids.com",
        "Known Flaws": "Paste from carsandbids.com",
        "Modifications": "Paste from carsandbids.com",
        "Recent Service History": "Paste from carsandbids.com",
        "Ownership History": "Paste from carsandbids.com",
        "Other Items Included in Sale": "Paste from carsandbids.com",
        "Seller Notes": "Paste from carsandbids.com",
    }

    st.info("Sending data to AI model...")
    
    try:
        # 2. Send Request to Lambda
        response = requests.post(API_URL, json=payload)
        
        # 3. Handle Response
        if response.status_code == 200:
            result = response.json()
            # Depending on how your Lambda returns data, it might be nested
            # Check if 'body' is inside the response or if it's direct
            if 'estimated_price' in result:
                price = result['estimated_price']
            elif 'body' in result:
                body_data = json.loads(result['body'])
                price = body_data.get('estimated_price', 0)
            else:
                price = 0
                st.error(f"Unexpected response format: {result}")

            st.success(f"### Estimated Price: ${price:,.2f}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        st.error(f"Connection failed: {e}")