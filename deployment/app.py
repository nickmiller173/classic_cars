import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from utils import engineer_sharp_features, extract_trim_slug, extract_performance_trim_flag, assign_trim_tier

# artifact loading
with open('model_artifacts_002.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

with open('encoding_artifacts_002.pkl', 'rb') as f:
    encoding_artifacts = pickle.load(f)

model = model_artifacts['model']
train_cols = model_artifacts['training_columns']

label_encoders = encoding_artifacts['label_encoders']
tfidf_vectorizer = encoding_artifacts['tfidf_vectorizer']
svd_model = encoding_artifacts['svd_model']

def lambda_handler(event, context):
    body = json.loads(event['body'])
    if body.get('warmup'):
        return {"statusCode": 200, "body": json.dumps({"status": "warm"})}
    input_df = pd.DataFrame([body])

    # feature derivation: age and mileage
    current_year = datetime.now().year
    model_year = float(input_df['Year'].iloc[0])
    
    # Clamped to 0 so a future model_year (e.g. a just-released model) never produces a
    # negative age, which would corrupt mileage_per_year and the age-based interactions.
    input_df['car_age'] = max(current_year - model_year, 0)
    input_df['model_year'] = model_year
    input_df['make_model_year'] = str(body.get('Make', '')) + '_' + str(body.get('Model', '')) + '_' + str(int(model_year))
    # raw_mileage is saved before log-transform so downstream features (mileage_per_year,
    # mileage_bucket, car_age_x_mileage) can use the original odometer value.
    raw_mileage = float(input_df['Mileage'].iloc[0])
    # +0.5 prevents division-by-zero for brand-new cars (car_age == 0) and softens the
    # rate estimate for very young cars where one extra year makes a huge difference.
    input_df['mileage_per_year'] = raw_mileage / (input_df['car_age'] + 0.5)
    # Log-transform compresses the heavy right skew of odometer readings so the model
    # treats the difference between 10k and 20k miles similarly to 100k vs 110k.
    input_df['Mileage'] = np.log1p(raw_mileage)
    
    # text processing
    input_df['flaw_count'] = input_df['Known Flaws'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() != '' else 0
    )

    text_cols = ['Highlights', 'Equipment', 'Modifications', 'Known Flaws',
                 'Recent Service History', 'Ownership History', 'Seller Notes', 'Other Items Included in Sale']
    
    # Fill missing text fields and lower (prevents inference crash)
    for col in text_cols:
        if col not in input_df.columns:
            input_df[col] = ""
        input_df[col] = input_df[col].astype(str).fillna('').str.lower()
    
    # engineer_sharp_features internally builds and then drops 'full_text_blob', so we
    # capture our own copy here first for the TF-IDF vectorizer step further below.
    text_blob = input_df[text_cols].apply(lambda x: ' '.join(x), axis=1)

    # sharp feature engineering
    input_df = engineer_sharp_features(input_df)

    # trim features — convert frontend trim_slug selection to tier for model
    trim_slug = body.get('trim_slug', 'unknown')
    input_df['trim_tier'] = assign_trim_tier(trim_slug)
    input_df['is_performance_trim'] = extract_performance_trim_flag(trim_slug)
    
    # market signals: S&P 500 (live fetch with fallback) & auction date
    try:
        sp500_hist = yf.Ticker("^GSPC").history(period="1d")
        sp500_close = float(sp500_hist['Close'].iloc[-1]) if not sp500_hist.empty else 5000.0
    except Exception:
        sp500_close = 5000.0
    input_df['SP500_Close'] = sp500_close
    input_df['auction_month'] = datetime.now().month
    input_df['auction_year'] = datetime.now().year

    # interaction features
    mileage_bins = [0, 20000, 50000, 100000, 150000, float('inf')]
    mileage_labels = ['0-20k', '20-50k', '50-100k', '100-150k', '150k+']
    mileage_bucket = pd.cut(pd.Series([raw_mileage]), bins=mileage_bins, labels=mileage_labels).iloc[0]
    # Groups the same model at different wear levels — lets the model learn that a low-mileage
    # Ferrari is priced differently than a high-mileage one even within the same make/model.
    input_df['make_model_mileage_bucket'] = (str(body.get('Make', '')) + '_' +
                                              str(body.get('Model', '')) + '_' + str(mileage_bucket))
    # Captures trim-level premiums per model year (e.g. a 2002 911 GT3 vs a 2002 911 base).
    input_df['make_model_trim'] = (str(body.get('Make', '')) + '_' + str(body.get('Model', '')) + '_' +
                                   str(int(model_year)) + '_' + str(input_df['trim_tier'].iloc[0]))
    # Captures whether private or dealer sellers discount clean vs problem titles differently.
    input_df['seller_x_title'] = str(body.get('Seller Type', '')) + '_' + str(body.get('Title Status', ''))
    # Overall usage intensity — an old car with very high miles is worse than age or miles alone.
    input_df['car_age_x_mileage'] = float(input_df['car_age'].iloc[0]) * raw_mileage
    # Captures whether a newer model is being driven unusually hard relative to its age.
    input_df['model_year_x_mileage_per_year'] = model_year * float(input_df['mileage_per_year'].iloc[0])
    # Lets the model account for collector-market conditions at auction time.
    input_df['sp500_x_auction_year'] = sp500_close * datetime.now().year
    # Dry-climate cars (less rust) depreciate more slowly; this amplifies that advantage with age.
    input_df['dry_climate_x_car_age'] = float(input_df['is_dry_climate_car'].iloc[0]) * float(input_df['car_age'].iloc[0])
    # Penalizes flaws more heavily on older cars where repairs are harder to source.
    input_df['flaw_severity_x_model_year'] = float(input_df['flaw_severity_score'].iloc[0]) * model_year

    # encoding
    for col in ['Exterior Color', 'Interior Color']:
        le = label_encoders.get(col)
        if le and col in input_df.columns:
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            elif 'Other' in le.classes_:
                # Unseen colors at inference are mapped to 'Other' rather than raising an
                # error — keeps the API robust to novel color strings without retraining.
                input_df[col] = le.transform(['Other'])[0]
            else:
                input_df[col] = -1

    one_hot_cols = ['Title Status', 'Seller Type', 'Drivetrain', 'Transmission_Type',
                    'Body Style', 'Engine_Cylinders', 'mod_status', 'auction_month', 'trim_tier']

    # pd.get_dummies is used instead of a fitted encoder because the exact column names and
    # ordering are enforced by the reindex(train_cols) step below — no fitted state needed.
    input_df = pd.get_dummies(input_df, columns=[c for c in one_hot_cols if c in input_df.columns])
    
    # text embeddings: TF-IDF -> SVD
    tfidf_matrix = tfidf_vectorizer.transform(text_blob)
    text_embeddings = svd_model.transform(tfidf_matrix)

    for i in range(20):
        input_df[f'text_component_{i}'] = text_embeddings[:, i]
        
    # column alignment and prediction
    # Aligns the single-row DataFrame to exactly the columns the model was trained on.
    # fill_value=0 handles one-hot columns absent from this request (e.g. a trim_tier
    # category not present in the payload) without raising a KeyError.
    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    prediction_log = model.predict(input_df)[0]
    # The model was trained on log1p(price), so expm1 is the exact inverse to recover dollars.
    final_price = float(np.expm1(prediction_log))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "estimated_price": final_price
        })
    }