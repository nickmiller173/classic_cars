import json
import pickle
import pandas as pd
import numpy as np
from utils import clean_mileage, get_main_color, extract_engine_info, engineer_sharp_features, engineer_date_features # Import your helpers

# Load artifacts ONCE (global scope) to speed up warm starts
with open('model_artifacts_001.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

with open('encoding_artifacts_001.pkl', 'rb') as f:
    encoding_artifacts = pickle.load(f)

model = model_artifacts['model']
l_encoders = encoding_artifacts['label_encoders']
t_map = encoding_artifacts['target_encoder_map']
train_cols = model_artifacts['training_columns']

def lambda_handler(event, context):
    # 1. Parse Input
    body = json.loads(event['body'])
    input_df = pd.DataFrame([body]) # Convert dict to single-row DataFrame
    input_df = engineer_date_features(input_df, is_inference=True)
    input_df = engineer_sharp_features(input_df)

    # 2. Apply cleaning (Use functions from utils.py)
    input_df['Mileage'] = input_df['Mileage'].apply(clean_mileage)
    # ... apply all other cleaning functions matching preprocessing.ipynb ...

    # 3. Apply Label Encoding
    for col, le in l_encoders.items():
        # Handle unseen labels safely (e.g. map to "Unknown" or mode)
        input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown') 
        # Note: You might need to add 'Unknown' class to your encoders during training
        input_df[col] = le.transform(input_df[col])

    # 4. Apply Target Encoding
    # Map model name to average price. If unknown model, use overall mean.
    global_mean = t_map.mean()
    input_df['Model_Target_Encoded'] = input_df['Model'].map(t_map).fillna(global_mean)
    input_df = input_df.drop(columns=['Model'])

    # 5. Apply One-Hot Encoding & Alignment
    # This generates dummies for this SINGLE row
    input_df = pd.get_dummies(input_df)
    
    # CRITICAL: Reindex aligns this single row to the 50+ columns the model expects
    # It adds missing columns (filling with 0) and drops extra ones
    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    # 6. Predict
    prediction = model.predict(input_df)

    return {
        'statusCode': 200,
        'body': json.dumps({'estimated_price': prediction[0]})
    }