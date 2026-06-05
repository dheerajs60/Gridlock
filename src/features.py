import pandas as pd
import numpy as np
from src.utils import decode_geohash, parse_time_features, impute_missing_values

def build_features(df, train_df=None, is_train=True):
    """
    Builds purely raw environmental and geospatial features for df.
    Eliminates target-leaky lag features.
    """
    # 1. Parse basic time features
    df = parse_time_features(df.copy())
    
    # 2. Decode geohashes
    lats_lons = df['geohash'].apply(decode_geohash)
    df['latitude'] = [x[0] for x in lats_lons]
    df['longitude'] = [x[1] for x in lats_lons]
    
    # 3. Geohash prefixes
    df['geohash_prefix4'] = df['geohash'].str[:4]
    df['geohash_prefix5'] = df['geohash'].str[:5]
    
    # 4. Impute missing values
    if train_df is not None:
        train_parsed = parse_time_features(train_df.copy())
        df = impute_missing_values(df, train_parsed)
    else:
        df = impute_missing_values(df)
        
    # Final NaNs safety check for features only (not target)
    feature_cols = [c for c in df.columns if c != 'demand']
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df
