import json
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
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
    
    # S&P 500 Placeholder & Auction Month proxy
    input_df['SP500_Close'] = 5000.0 
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
    
    # Extract the preprocessor and the tree model from the pipeline
    # Note: SHould change 'xgb' to other model name if model family ever changes
    preprocessor = model.named_steps['preprocessor']
    tree_model = model.named_steps['xgb'] 
    
    # Transform the input data into the exact numeric array the tree sees
    X_transformed = preprocessor.transform(input_df)

    # Use XGBoost's native SHAP computation (pred_contribs=True) - avoids the SHAP library entirely.
    # Output shape: (1, n_features + 1) where the last column is the bias term (expected value).
    contribs = tree_model.get_booster().predict(xgb.DMatrix(X_transformed), pred_contribs=True)
    base_value_log = float(contribs[0, -1])
    base_price = float(np.expm1(base_value_log))

    feature_names = [n.split('__')[-1] for n in preprocessor.get_feature_names_out()]
    log_impacts = contribs[0, :-1]
    total_log_impact = np.sum(log_impacts)
    dollar_difference = final_price - base_price
    
    shap_breakdown = {"Base Market Value": base_price}
    
    # Isolate the top 5 most impactful features to keep the frontend chart clean
    impact_magnitudes = np.abs(log_impacts)
    top_5_indices = np.argsort(impact_magnitudes)[-5:]
    
    for idx in top_5_indices:
        feat_name = feature_names[idx]
        log_val = log_impacts[idx]
        
        # Proportional dollar allocation
        if total_log_impact != 0:
            dollar_impact = (log_val / total_log_impact) * dollar_difference
        else:
            dollar_impact = 0
            
        shap_breakdown[feat_name] = float(dollar_impact)
        
    # Group all remaining minor features into an "Other Factors" bucket to ensure the math balances
    other_impact = dollar_difference - sum([v for k, v in shap_breakdown.items() if k != "Base Market Value"])
    shap_breakdown["Other Factors"] = float(other_impact)

    # 7. Return the updated JSON response
    return {
        "statusCode": 200,
        "body": json.dumps({
            "estimated_price": final_price,
            "shap_breakdown": shap_breakdown
        })
    }