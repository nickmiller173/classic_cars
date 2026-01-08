import pandas as pd
import re


def clean_currency(x):
    """
    Converts '$8,400' string to 8400.0 float.
    NOW ROBUST: Returns None if it encounters text like 'Porsche'.
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
    NOW ROBUST: Returns None if it encounters text.
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


def clean_title(x):
    if pd.isna(x):
        return x
    return x.split('(')[0].strip()


def clean_transmission_type(x):
    if pd.isna(x):
        return x
    return x.split('(')[0].strip()


def extract_gears(x):
    if pd.isna(x):
        return None
    match = re.search(r'\((\d+)-Speed\)', x)
    if match:
        return int(match.group(1))
    return None


def extract_engine_info(x):
    if pd.isna(x):
        return None, None
    disp = re.search(r'(\d+\.\d+)L', x)
    cyl = re.search(r'([V|I|H|W]\d+)', x)
    d_val = float(disp.group(1)) if disp else None
    c_val = cyl.group(1) if cyl else None
    return d_val, c_val


def get_main_color(x):
    if pd.isna(x):
        return "Unknown"

    x = x.split('/')[0].strip()
    x_lower = x.lower()

    special_map = {
        'bianco': 'White', 'salsa': 'Red', 'granite': 'Gray', 'anthracite': 'Gray',
        'carbon': 'Black', 'jade': 'Green', 'cypress': 'Green', 'ebony': 'Black',
        'linen': 'Beige', 'cream': 'Beige', 'macchiato': 'Brown', 'charcoal': 'Gray',
        'graphite': 'Gray', 'slate': 'Gray', 'chalk': 'White', 'cocoa': 'Brown'
    }

    for key, val in special_map.items():
        if key in x_lower:
            return val

    std_colors = ['Black', 'White', 'Gray', 'Grey', 'Silver', 'Red', 'Blue', 
                  'Green', 'Brown', 'Beige', 'Yellow', 'Orange', 'Gold', 'Purple', 'Tan']

    for color in std_colors:
        if color.lower() in x_lower:
            return 'Gray' if color == 'Grey' else color

    return "Other"