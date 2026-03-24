from datetime import datetime
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import sys
import os
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def clean_currency(x):
    """
    Converts '$8,400' string to 8400.0 float.
    """
    if pd.isna(x):
        return None
    if isinstance(x, str):
        # Remove symbols
        x_clean = x.replace('$', '').replace(',', '').strip()
        try:
            return float(x_clean)
        except ValueError:
            # If we can't convert it (e.g. it's text), return None
            return None
    return float(x)

def clean_mileage(x):
    """
    Converts '53,700' string to 53700.0 float.
    """
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x_clean = x.replace(',', '').strip()
        try:
            return float(x_clean)
        except ValueError:
            return None
    return float(x)

def clean_model(x):
    if pd.isna(x):
        return x
    return x.replace('\nSave', '').strip()

def clean_and_group_title(x):
    if pd.isna(x):
        return 'Unknown' 
    
    # Apply original split and strip, then lowercase to handle casing issues
    val = str(x).split('(')[0].strip().lower()
    
    # 1. Clean Titles
    if val in ['clean', 'clen']:
        return 'Clean'
    
    # 2. Rebuilt / Salvage / Restored
    if any(keyword in val for keyword in ['rebuilt', 'salvage', 'reconstructed', 'totaled', 'restored']):
        return 'Rebuilt/Salvage'
        
    # 3. Mileage & Odometer Issues
    if any(keyword in val for keyword in ['mileage', 'odometer', 'mechanical limits']):
        return 'Mileage Issue'
        
    # 4. Lemon Law / Manufacturer Buyback
    if 'buyback' in val:
        return 'Buyback'
        
    # 5. Alternate Documentation (Bill of sale, Registered only, etc.)
    if val in ['bill of sale', 'no title', 'registered']:
        return 'Alternate Doc'
        
    # Catch-all for anything missed
    return 'Other'

def clean_transmission_type(x):
    if pd.isna(x):
        return "Unknown"
        
    # Split on parenthesis and strip trailing spaces
    val = str(x).split('(')[0].strip()
    
    # Strictly enforce valid types to filter out the charity auction paragraphs
    if val in ['Automatic', 'Manual']:
        return val
        
    return 'Other'

def extract_gears(x):
    if pd.isna(x):
        return None
        
    x_str = str(x)
    
    # 1. Improved Regex: 
    # (?i) makes it case-insensitive (catches "speed" and "Speed")
    # [-\s] allows either a hyphen or a space
    match = re.search(r'(?i)(\d+)[-\s]speed', x_str)
    
    if match:
        return float(match.group(1))
        
    # 2. Handle CVTs explicitly (usually coded as 1 gear for ML purposes)
    if 'CVT' in x_str.upper():
        return 1.0
        
    return None

def extract_engine_info(x):
    # 1. Handle entirely missing/blank values
    if pd.isna(x) or str(x).strip() == '':
        return None, "Unknown"  # None for numeric, "Unknown" for categorical
    
    # --- Fix Displacement ---
    disp_l = re.search(r'(\d+\.?\d*)\s*L', x, re.IGNORECASE)
    disp_cc = re.search(r'(\d+)\s*cc', x, re.IGNORECASE)
    disp_ci = re.search(r'(\d+)\s*ci', x, re.IGNORECASE)
    
    # Default to None for missing numeric values
    d_val = None 
    if disp_l:
        d_val = float(disp_l.group(1))
    elif disp_cc:
        d_val = round(float(disp_cc.group(1)) / 1000.0, 1) 
    elif disp_ci:
        d_val = round(float(disp_ci.group(1)) / 61.0237, 1) 
        
    # --- Fix Cylinders ---
    c_val = "Other" 
    
    cyl = re.search(r'([VIW])[- ]?(\d+)', x, re.IGNORECASE)
    flat = re.search(r'Flat[- ]?(\d+)', x, re.IGNORECASE)
    inline = re.search(r'Inline[- ]?(\d+)', x, re.IGNORECASE)
    l_typo = re.search(r'(l)(\d+)', x)
    
    if cyl:
        c_val = cyl.group(1).upper() + cyl.group(2)
    elif inline:
        c_val = 'I' + inline.group(1)
    elif flat:
        c_val = 'H' + flat.group(1) 
    elif l_typo:
        c_val = 'I' + l_typo.group(2)
    elif 'Rotary' in x:
        c_val = 'Rotary'
    elif 'Electric' in x or 'Motor' in x:
        c_val = 'Electric'
        
    return d_val, c_val

def get_main_color(x):
    # 1. Catch missing values and group them into "Other"
    if pd.isna(x):
        return "Other"
        
    # 2. Grab the primary color before slashes or " and "
    x = str(x).split('/')[0].split(' and ')[0].strip()
    x_lower = x.lower()
    
    # 3. Check for specific keywords FIRST to prevent substring collisions 
    special_map = {
        # Edge cases, Collisions & Exterior Bleed-over
        'titanium': 'Gray', 'titan': 'Black', 'mustang': 'Brown', 'tanzanite': 'Blue',
        'stainless': 'Silver', 'mercury': 'Silver', 'magnetic': 'Gray', 'thunder': 'Gray',
        
        # Blacks / Darks
        'ebony': 'Black', 'nero': 'Black', 'carbon': 'Black', 'onyx': 'Black', 
        'jet': 'Black', 'obsidian': 'Black', 'beluga': 'Black', 'panther': 'Black',
        'amido': 'Black', 'midnight': 'Black', 'anthracite': 'Gray', 'zebra': 'Black',
        
        # Grays / Silvers
        'granite': 'Gray', 'charcoal': 'Gray', 'graphite': 'Gray', 'slate': 'Gray', 
        'ash': 'Gray', 'agate': 'Gray', 'stone': 'Gray', 'shale': 'Gray', 
        'platinum': 'Gray', 'pewter': 'Gray', 'palladium': 'Gray', 'meteor': 'Gray',
        'flint': 'Gray', 'ocean': 'Gray',
        
        # Whites / Lights
        'chalk': 'White', 'ivory': 'White', 'pearl': 'White', 'porcelain': 'White', 
        'alabaster': 'White', 'bianco': 'White', 'magnolia': 'White', 'oyster': 'White',
        'ice': 'White', 'ceramic': 'White',
        
        # Beiges / Tans / Browns
        'parchment': 'Beige', 'linen': 'Beige', 'cream': 'Beige', 'ecru': 'Beige', 
        'luxor': 'Beige', 'cashmere': 'Beige', 'savanna': 'Beige', 'almond': 'Beige', 
        'bamboo': 'Beige', 'wheat': 'Beige', 'champagne': 'Beige', 'kalahari': 'Beige', 
        'gobi': 'Beige', 'macchiato': 'Beige', 'taupe': 'Beige', 'sand': 'Beige', 
        'dune': 'Beige', 'saddle': 'Brown', 'oak': 'Brown', 'cocoa': 'Brown', 
        'cognac': 'Brown', 'caramel': 'Brown', 'cuoio': 'Brown', 'cinnamon': 'Brown', 
        'java': 'Brown', 'havanna': 'Brown', 'havana': 'Brown', 'mocha': 'Brown', 
        'espresso': 'Brown', 'nougat': 'Brown', 'chestnut': 'Brown', 'amaro': 'Brown', 
        'sepia': 'Brown', 'truffle': 'Brown', 'walnut': 'Brown', 'tartufo': 'Brown', 
        'terra': 'Brown', 'natural': 'Brown', 'palomino': 'Tan', 'camel': 'Tan', 
        'khaki': 'Tan', 'atacama': 'Tan',
        
        # Reds / Oranges
        'salsa': 'Red', 'coral': 'Red', 'imola': 'Red', 'fox': 'Red', 
        'burgundy': 'Red', 'magma': 'Red', 'carrera': 'Red', 'maroon': 'Red', 
        'chateau': 'Red', 'bordeaux': 'Red', 'fiona': 'Red', 'scarlet': 'Red', 
        'garnet': 'Red', 'crimson': 'Red', 'ruby': 'Red', 'cabernet': 'Red', 
        'rosso': 'Red', 'sakhir': 'Orange', 'kyalami': 'Orange',
        
        # Greens / Blues
        'jade': 'Green', 'cypress': 'Green', 'forest': 'Green', 'nordkap': 'Blue', 
        'nautic': 'Blue', 'yachting': 'Blue', 'estoril': 'Blue', 'marina': 'Blue'
    }
    
    for key, val in special_map.items():
        if key in x_lower:
            return val

    # 4. Check standard baseline colors
    std_colors = [
        'black', 'white', 'gray', 'grey', 'silver', 'red', 'blue', 
        'green', 'brown', 'beige', 'yellow', 'orange', 'gold', 'purple', 'tan'
    ]
    
    for color in std_colors:
        if color in x_lower:
            return 'Gray' if color == 'grey' else color.capitalize()
            
    # 5. Everything else becomes "Other"
    return "Other"

def clean_seller_type(x):
    if pd.isna(x):
        return "Unknown"
        
    val = str(x)
    
    # Consolidate all Dealer types (ignores doc fees, etc.)
    if 'Dealer' in val:
        return 'Dealer'
        
    # Consolidate all Private Party types (ignores liens, temporary tags, \n, etc.)
    elif 'Private Party' in val:
        return 'Private Party'

    else:
        return 'Other'

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
    lambda x: 1 if re.search(r'\b(2 keys|3 keys|both original keys|both keys)\b', x) else 0
    )

    df['owners_manual_ind'] = df['Other Items Included in Sale'].apply(
    lambda x: 1 if re.search(r'\b(manual|owners)\b', x) else 0
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
        lambda x: 1 if re.search(r'\b(timing belt|clutch|ims|head gasket|water pump|transmission replaced|engine replaced)\b', x) else 0
    )

    # 5. Categorical extraction
    df['mod_status'] = df['Modifications'].apply(categorize_mods)
    
    return df

HIGH_VALUE_TRIM_PATTERN = r'\b(gt2|gt3|gt2rs|gt3rs|turbo-s|zr1|z06|hellcat|gt500|gt350|shelby|trd-pro|raptor|competition|black-series|gts-4)\b'

def extract_trim_slug(url, make, model=''):
    """
    Extracts the trim from a Cars & Bids auction URL by stripping year, make, and model tokens.
    E.g. '.../2006-porsche-996-911-gt3' -> 'gt3'
         '.../2003-porsche-996-911-carrera-4s' -> 'carrera-4s'
    Returns 'unknown' when no URL is available (target encoder falls back to mean).
    """
    if pd.isna(url) or not isinstance(url, str) or not url.strip():
        return 'unknown'

    slug = url.rstrip('/').split('/')[-1].lower()
    slug = re.sub(r'^\d{4}-', '', slug)  # strip year prefix

    # Build stop-token set from make + model (handles multi-word like "mercedes-benz", "996 911")
    make_tokens = set(re.sub(r'[^a-z0-9]+', '-', str(make).lower()).strip('-').split('-'))
    model_tokens = set(re.sub(r'[^a-z0-9]+', '-', str(model).lower()).strip('-').split('-'))
    stop_tokens = make_tokens | model_tokens

    # Filter ALL matching tokens (not just leading) so "996" and "911" are removed wherever they appear
    filtered = [t for t in slug.split('-') if t not in stop_tokens]

    result = '-'.join(filtered)
    return result if result else 'base'


def extract_performance_trim_flag(url):
    """Returns 1 if the URL slug contains a known high-value performance trim."""
    if pd.isna(url) or not isinstance(url, str):
        return 0
    slug = url.rstrip('/').split('/')[-1].lower()
    return 1 if re.search(HIGH_VALUE_TRIM_PATTERN, slug) else 0


def clean_date(x):
    """
    Robustly extracts date from strings like 'Feb 18, 2026 1:48 PM MST'.
    Ignores time and timezone.
    """
    if pd.isna(x):
        return None

    # NEW: If it's already a datetime or pandas Timestamp (from our sorting step), just return it
    if isinstance(x, datetime) or type(x).__name__ == 'Timestamp':
        return x

    x = str(x).strip()

    # REGEX STRATEGY: Look for "Mmm DD, YYYY" pattern
    # This matches "Feb 18, 2026" inside the longer string
    match = re.search(r'([A-Za-z]{3}\s+\d{1,2},\s+\d{4})', x)

    if match:
        date_str = match.group(1) # "Feb 18, 2026"
        try:
            return datetime.strptime(date_str, "%b %d, %Y")
        except ValueError:
            return None

    # Fallback for ISO dates (2026-02-18) just in case
    try:
        # Added .split(' ')[0] to safely strip off time if it exists in the string
        return datetime.strptime(x.split(' ')[0], "%Y-%m-%d")
    except ValueError:
        return None

def engineer_date_features(df, is_inference=False):
    """
    Extracts Year, Month, and calculates Car Age.
    NOW INCLUDES: Logic to grab Model Year from the "THIS... is a..." intro.
    """
    df = df.copy()
    
    # 1. Date Parsing (Auction End Date)
    if 'Auction_Date' not in df.columns:
        df['Auction_Date'] = None

    cleaned_dates = df['Auction_Date'].apply(clean_date)
    
    if is_inference:
        cleaned_dates = cleaned_dates.fillna(datetime.now())
    
    df['date_obj'] = pd.to_datetime(cleaned_dates)
    df['auction_year'] = df['date_obj'].dt.year
    df['auction_month'] = df['date_obj'].dt.month
    
    # 2. Extract Model Year (The "THIS... is a 1998" Logic)
    def extract_year_from_text(row):
        text = str(row.get('Highlights', ''))
        
        # Added \s* before the dots to catch spaces, and an? to catch "a" or "an"
        match = re.search(r'THIS\s*[…\.]+\s+is\s+an?\s+(\d{4})', text, re.IGNORECASE)
        
        if match:
            return float(match.group(1))

        return None

    df['model_year'] = df.apply(extract_year_from_text, axis=1)

    # 3. Calculate "Age at Sale"
    # If we still can't find the year, we default to the auction year (age=0) to prevent errors
    df['model_year'] = df['model_year'].fillna(df['auction_year'])
    
    df['car_age'] = df['auction_year'] - df['model_year']
    
    # Sanity Check: If age is negative (e.g. 2024 model sold in 2023), clamp to 0
    df['car_age'] = df['car_age'].apply(lambda x: max(x, 0))
        
    # Cleanup
    df = df.drop(columns=['date_obj', 'model_year']) # We drop model_year because 'car_age' is the better predictor
    
    return df