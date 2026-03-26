import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from utils import engineer_sharp_features, extract_trim_slug, extract_performance_trim_flag, assign_trim_tier

# 1. Load Artifacts
with open('model_artifacts_002.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

with open('encoding_artifacts_002.pkl', 'rb') as f:
    encoding_artifacts = pickle.load(f)

model = model_artifacts['model']
train_cols = model_artifacts['training_columns']

label_encoders = encoding_artifacts['label_encoders']
tfidf_vectorizer = encoding_artifacts['tfidf_vectorizer'] # NEW: Load TF-IDF
svd_model = encoding_artifacts['svd_model']               # NEW: Load SVD

def lambda_handler(event, context):
    body = json.loads(event['body'])
    if body.get('warmup'):
        return {"statusCode": 200, "body": json.dumps({"status": "warm"})}
    input_df = pd.DataFrame([body]) 
    
    # 2. Recreate missing calculated features 
    current_year = datetime.now().year
    model_year = float(input_df['Year'].iloc[0])
    
    input_df['car_age'] = max(current_year - model_year, 0)
    input_df['model_year'] = model_year
    input_df['make_model_year'] = str(body.get('Make', '')) + '_' + str(body.get('Model', '')) + '_' + str(int(model_year))
    raw_mileage = float(input_df['Mileage'].iloc[0])
    input_df['mileage_per_year'] = raw_mileage / (input_df['car_age'] + 0.5)
    input_df['Mileage'] = np.log1p(raw_mileage)
    
    input_df['flaw_count'] = input_df['Known Flaws'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() != '' else 0
    )
    
    # --- 3. NLP Text Blob Creation ---
    text_cols = ['Highlights', 'Equipment', 'Modifications', 'Known Flaws', 
                 'Recent Service History', 'Ownership History', 'Seller Notes', 'Other Items Included in Sale']
    
    # Fill missing text fields and lower (prevents inference crash)
    for col in text_cols:
        if col not in input_df.columns:
            input_df[col] = ""
        input_df[col] = input_df[col].astype(str).fillna('').str.lower()
    
    # Create the blob before engineer_sharp_features drops it
    text_blob = input_df[text_cols].apply(lambda x: ' '.join(x), axis=1)

    # 4. Sharp features and Boolean flags from utils.py
    input_df = engineer_sharp_features(input_df)

    # Trim features — convert frontend trim_slug selection to tier for model
    trim_slug = body.get('trim_slug', 'unknown')
    input_df['trim_tier'] = assign_trim_tier(trim_slug)
    input_df['is_performance_trim'] = extract_performance_trim_flag(trim_slug)
    
    # S&P 500 (live fetch with fallback) & Auction Month proxy
    try:
        sp500_hist = yf.Ticker("^GSPC").history(period="1d")
        sp500_close = float(sp500_hist['Close'].iloc[-1]) if not sp500_hist.empty else 5000.0
    except Exception:
        sp500_close = 5000.0
    input_df['SP500_Close'] = sp500_close
    input_df['auction_month'] = datetime.now().month
    input_df['auction_year'] = datetime.now().year

    # --- Interaction Features ---
    mileage_bins = [0, 20000, 50000, 100000, 150000, float('inf')]
    mileage_labels = ['0-20k', '20-50k', '50-100k', '100-150k', '150k+']
    mileage_bucket = pd.cut(pd.Series([raw_mileage]), bins=mileage_bins, labels=mileage_labels).iloc[0]
    input_df['make_model_mileage_bucket'] = (str(body.get('Make', '')) + '_' +
                                              str(body.get('Model', '')) + '_' + str(mileage_bucket))
    input_df['make_model_trim'] = (str(body.get('Make', '')) + '_' + str(body.get('Model', '')) + '_' +
                                   str(int(model_year)) + '_' + str(input_df['trim_tier'].iloc[0]))
    input_df['seller_x_title'] = str(body.get('Seller Type', '')) + '_' + str(body.get('Title Status', ''))
    input_df['car_age_x_mileage'] = float(input_df['car_age'].iloc[0]) * raw_mileage
    input_df['model_year_x_mileage_per_year'] = model_year * float(input_df['mileage_per_year'].iloc[0])
    input_df['sp500_x_auction_year'] = sp500_close * datetime.now().year
    input_df['dry_climate_x_car_age'] = float(input_df['is_dry_climate_car'].iloc[0]) * float(input_df['car_age'].iloc[0])
    input_df['flaw_severity_x_model_year'] = float(input_df['flaw_severity_score'].iloc[0]) * model_year

    # 5. Apply Label Encoders safely to Colors
    for col in ['Exterior Color', 'Interior Color']:
        le = label_encoders.get(col)
        if le and col in input_df.columns:
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            elif 'Other' in le.classes_:
                input_df[col] = le.transform(['Other'])[0]
            else:
                input_df[col] = -1

    # 6. Create One-Hot Encodings
    one_hot_cols = ['Title Status', 'Seller Type', 'Drivetrain', 'Transmission_Type',
                    'Body Style', 'Engine_Cylinders', 'mod_status', 'auction_month', 'trim_tier']
    
    input_df = pd.get_dummies(input_df, columns=[c for c in one_hot_cols if c in input_df.columns])
    
    # --- 7. Apply Text Embeddings ---
    # Transform the text blob into sparse TF-IDF, then reduce to dense 20 SVD components
    tfidf_matrix = tfidf_vectorizer.transform(text_blob)
    text_embeddings = svd_model.transform(tfidf_matrix)

    for i in range(20):
        input_df[f'text_component_{i}'] = text_embeddings[:, i]
        
    # 8. Align to the exact columns the XGBoost Pipeline expects
    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    # 9. Predict & Reverse Log Transform
    prediction_log = model.predict(input_df)[0]
    final_price = float(np.expm1(prediction_log))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "estimated_price": final_price
        })
    }