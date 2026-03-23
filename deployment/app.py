import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from utils import engineer_sharp_features

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
    input_df = pd.DataFrame([body]) 
    
    # 2. Recreate missing calculated features 
    current_year = datetime.now().year
    model_year = float(input_df['Year'].iloc[0])
    
    input_df['car_age'] = max(current_year - model_year, 0)
    input_df['mileage_per_year'] = float(input_df['Mileage'].iloc[0]) / (input_df['car_age'] + 0.5)
    
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
    
    # S&P 500 (live fetch with fallback) & Auction Month proxy
    try:
        sp500_hist = yf.Ticker("^GSPC").history(period="1d")
        sp500_close = float(sp500_hist['Close'].iloc[-1]) if not sp500_hist.empty else 5000.0
    except Exception:
        sp500_close = 5000.0
    input_df['SP500_Close'] = sp500_close
    input_df['auction_month'] = datetime.now().month
    input_df['auction_year'] = datetime.now().year

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
                    'Body Style', 'Engine_Cylinders', 'mod_status', 'auction_month']
    
    input_df = pd.get_dummies(input_df, columns=[c for c in one_hot_cols if c in input_df.columns])
    
    # --- 7. Apply Text Embeddings ---
    # Transform the text blob into sparse TF-IDF, then reduce to dense 10 SVD components
    tfidf_matrix = tfidf_vectorizer.transform(text_blob)
    text_embeddings = svd_model.transform(tfidf_matrix)
    
    for i in range(10):
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