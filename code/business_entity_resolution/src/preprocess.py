import string
import pandas as pd
import numpy as np

# Fast translation table converting all punctuation to spaces
PUNCT_TRANS = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def fast_clean_list(vals):
    """
    Sub-second, memory-efficient string translation.
    Returns python list of clean strings.
    """
    return [str(s).lower().translate(PUNCT_TRANS) if pd.notnull(s) else "" for s in vals]

def preprocess_dataframe(df):
    """
    In-place memory-efficient preprocessing of a DataFrame without df.copy().
    """
    if 'business_name' in df.columns:
        df['norm_name'] = fast_clean_list(df['business_name'].values)
    else:
        df['norm_name'] = ""

    if 'business_address' in df.columns:
        df['norm_address'] = fast_clean_list(df['business_address'].values)
    else:
        df['norm_address'] = ""

    if 'country' in df.columns:
        df['country'] = [str(c).strip().upper() if pd.notnull(c) else "" for c in df['country'].values]

    return df

def extract_blocking_tokens(name_str):
    """
    Extract tokens of length >= 3 for candidate blocking.
    """
    if not name_str or pd.isnull(name_str):
        return []
    tokens = str(name_str).lower().translate(PUNCT_TRANS).split()
    return [t for t in tokens if len(t) >= 3]
