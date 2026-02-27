import pandas as pd
import re
from datetime import datetime
import numpy as np

# --- NEW FEATURE ENGINEERING LOGIC ---

def categorize_mods(text):
    """Categorizes modification severity."""
    if pd.isna(text): return "stock"
    text = str(text).lower()
    if "stock" in text and len(text) < 20: return "stock"
    
    if re.search(r'\b(turbo kit|supercharger|engine swap|ls swap|k20|roll cage|drilled|turbo|suspension)\b', text):
        return "heavy_mod"
    if re.search(r'\b(exhaust|muffler|intake|wheels|coilovers|springs|tint|bluetooth|screen|touchscreen|hitch|emblem)\b', text):
        return "light_mod"
    
    return "unknown_mod"

def calculate_flaw_severity(row):
    """Calculates a numeric penalty score for flaws."""
    # Handle missing columns gracefully
    flaws = str(row.get('Known Flaws', ''))
    notes = str(row.get('Seller Notes', ''))
    text = (flaws + " " + notes).lower()
    
    score = 0
    # Tier 1: Catastrophic
    if re.search(r'\b(salvage|rebuilt|branded|flood|frame damage|rolled)\b', text): score += 50
    if re.search(r'\b(tmu|true mileage unknown|odometer broken)\b', text): score += 40
    if re.search(r'\b(rust holes|rot|perforated|frame rust)\b', text): score += 40
    if re.search(r'\b(knock|smoke|overheat|head gasket|trans slip)\b', text): score += 30

    # Tier 2: Expensive
    if re.search(r'\b(leak|seep|drip|fluid)\b', text): score += 10
    if re.search(r'\b(crack|tear|rip|worn)\b', text): score += 5 
    if re.search(r'\b(dent|ding|scratch|paint chip|peeling)\b', text): score += 5 
    
    return score

def engineer_sharp_features(df):
    """
    Master function to generate all 'sharp' text features.
    Works on both Training Data (Many Rows) and Inference Data (Single Row).
    """
    df = df.copy()
    
    # 1. Ensure text columns exist (prevents crash on inference if JSON is incomplete)
    text_cols = ['Highlights', 'Equipment', 'Modifications', 'Known Flaws', 
                 'Recent Service History', 'Ownership History', 'Seller Notes', 'Other Items Included in Sale']
    
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna('').str.lower()

    # Create a blob for broad searching
    df['full_text_blob'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)

    # 2. Boolean Flags (The Multipliers)
    df['2_keys_ind'] = df['Other Items Included in Sale'].apply(
    lambda x: 1 if re.search(r'\b(2 keys|3 keys)\b', x) else 0
    )

    df['is_dry_climate_car'] = df['full_text_blob'].apply(
        lambda x: 1 if re.search(r'\b(california|arizona|texas|nevada|dry state|no rust)\b', x) else 0
    )

    df['is_project_car'] = df['full_text_blob'].apply(
        lambda x: 1 if re.search(r'\b(project|needs restoration|not running|tow away)\b', x) else 0
    )

    df['has_new_tires'] = df['Recent Service History'].apply(
    lambda x: 1 if re.search(r'\b(new tires|replaced tires|michelin|fresh rubber)\b', x) else 0
    )

    df['has_sport_seats'] = df['full_text_blob'].apply(
        lambda x: 1 if re.search(r'\b(recaro|sport seat|bucket seat|vader|wingback)\b', x) else 0
    )

    df['emissions_ind'] = df['Seller Notes'].apply(
        lambda x: 1 if re.search(r'\b(emissions)\b', x) else 0
    )

    df['loan_ind'] = df['Seller Notes'].apply(
        lambda x: 1 if re.search(r'\b(loan)\b', x) else 0
    )

    df['one_owner_ind'] = df['full_text_blob'].apply(
        lambda x: 1 if re.search(r'\b(one owner)\b', x) else 0
    )

    df['carfax_ind'] = df['Known Flaws'].apply(
        lambda x: 1 if re.search(r'\b(carfax)\b', x) else 0
    )

    # 3. Calculated Scores
    df['flaw_severity_score'] = df.apply(calculate_flaw_severity, axis=1)

    # 4. Recent Maintenance (Capex)
    df['recent_major_service'] = df['Recent Service History'].apply(
        lambda x: 1 if re.search(r'\b(timing belt|clutch|ims|head gasket|water pump|transmission replaced|engine replaced|)\b', x) else 0
    )

    # 5. Categorical extraction
    df['mod_status'] = df['Modifications'].apply(categorize_mods)

    # Drop the temporary blob to keep things clean
    df = df.drop(columns=['full_text_blob'])
    
    return df