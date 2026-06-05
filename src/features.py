import pandas as pd
import numpy as np
from src.utils import decode_geohash, parse_time_features, impute_missing_values

def get_day48_grid(train_df):
    """
    Creates a complete interpolated grid of demand on Day 48 for all geohashes.
    """
    day48 = train_df[train_df['day'] == 48].copy()
    day48 = parse_time_features(day48)
    
    geohashes = day48['geohash'].unique()
    times = sorted(day48['time_minutes'].unique())
    
    # Create complete grid
    grid = pd.MultiIndex.from_product([geohashes, times], names=['geohash', 'time_minutes']).to_frame().reset_index(drop=True)
    grid = grid.merge(day48[['geohash', 'time_minutes', 'demand']], on=['geohash', 'time_minutes'], how='left')
    
    # Sort and interpolate
    grid = grid.sort_values(['geohash', 'time_minutes'])
    grid['demand48'] = grid.groupby('geohash')['demand'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both').fillna(0.0)
    )
    
    # Add prefix columns for neighborhood profiling on Day 48
    grid['geohash_prefix5'] = grid['geohash'].str[:5]
    grid['geohash_prefix4'] = grid['geohash'].str[:4]
    
    return grid

def build_features(df, train_df=None):
    """
    Builds spatiotemporal features and maps Day 48 scaled profiles.
    - Decodes geohashes & imputes environmental variables.
    - Computes Day 48 early morning demand profile.
    - Scales Day 48 daytime profiles using Day 49 early morning ratios (with Neighborhood-Scaled Imputation fallback).
    - Excludes targets from early morning demand features to prevent leakage during training.
    """
    df = df.copy()
    
    # 1. Parse basic time features
    df = parse_time_features(df)
    
    # 2. Decode geohashes
    lats_lons = df['geohash'].apply(decode_geohash)
    df['latitude'] = [x[0] for x in lats_lons]
    df['longitude'] = [x[1] for x in lats_lons]
    
    # 3. Geohash prefixes
    df['geohash_prefix4'] = df['geohash'].str[:4]
    df['geohash_prefix5'] = df['geohash'].str[:5]
    
    # 4. Impute missing values
    ref_train = train_df if train_df is not None else df[df['demand'].notna()]
    train_parsed = parse_time_features(ref_train.copy())
    df = impute_missing_values(df, train_parsed)
    
    # 5. Day 48 (previous day) historical grid & profiles
    grid48 = get_day48_grid(ref_train)
    
    # Merge Day 48 demand
    df = df.merge(grid48[['geohash', 'time_minutes', 'demand48']], on=['geohash', 'time_minutes'], how='left')
    df['demand48'] = df['demand48'].fillna(0.0)
    
    # Day 48 early morning average per geohash
    d48_clean = ref_train[ref_train['day'] == 48].copy()
    d48_clean = parse_time_features(d48_clean)
    d48_early = d48_clean[d48_clean['time_minutes'] <= 120].groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_48'})
    df = df.merge(d48_early, on='geohash', how='left')
    df['early_48'] = df['early_48'].fillna(df['early_48'].mean())
    
    # Day 49 observed early morning statistics
    d49_early_rows = ref_train[(ref_train['day'] == 49) & (parse_time_features(ref_train)['time_minutes'] <= 120)].copy()
    early_stats = d49_early_rows.groupby('geohash')['demand'].agg(['sum', 'count']).reset_index()
    df = df.merge(early_stats, on='geohash', how='left')
    df['sum'] = df['sum'].fillna(0.0)
    df['count'] = df['count'].fillna(0)
    
    # Non-leaky observed early average
    # Subtract current row's demand if it is part of Day 49 early morning (0:00 to 2:00) during training
    is_d49_early = (df['day'] == 49) & (df['time_minutes'] <= 120) & (df['demand'].notna())
    df['early_49_obs'] = np.where(
        is_d49_early,
        (df['sum'] - df['demand'].fillna(0.0)) / (df['count'] - 1).clip(lower=1),
        df['sum'] / df['count'].clip(lower=1)
    )
    df['early_49_obs'] = np.where(df['count'] == 0, np.nan, df['early_49_obs'])
    
    # Neighborhood and global ratios for imputation
    early_by_geohash = d49_early_rows.groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_49_g'})
    early_by_geohash = early_by_geohash.merge(d48_early, on='geohash', how='left')
    early_by_geohash['early_48'] = early_by_geohash['early_48'].fillna(early_by_geohash['early_48'].mean())
    early_by_geohash['geohash_prefix5'] = early_by_geohash['geohash'].str[:5]
    early_by_geohash['geohash_prefix4'] = early_by_geohash['geohash'].str[:4]
    
    p5_means = early_by_geohash.groupby('geohash_prefix5')[['early_49_g', 'early_48']].mean().reset_index()
    p5_means.columns = ['geohash_prefix5', 'p5_early_49', 'p5_early_48']
    p5_means['prefix5_ratio'] = (p5_means['p5_early_49'] + 0.01) / (p5_means['p5_early_48'] + 0.01)
    
    p4_means = early_by_geohash.groupby('geohash_prefix4')[['early_49_g', 'early_48']].mean().reset_index()
    p4_means.columns = ['geohash_prefix4', 'p4_early_49', 'p4_early_48']
    p4_means['prefix4_ratio'] = (p4_means['p4_early_49'] + 0.01) / (p4_means['p4_early_48'] + 0.01)
    
    df = df.merge(p5_means[['geohash_prefix5', 'prefix5_ratio', 'p5_early_49']], on='geohash_prefix5', how='left')
    df = df.merge(p4_means[['geohash_prefix4', 'prefix4_ratio', 'p4_early_49']], on='geohash_prefix4', how='left')
    
    global_early_49 = d49_early_rows['demand'].mean()
    global_early_48 = d48_clean[d48_clean['time_minutes'] <= 120]['demand'].mean()
    global_ratio = (global_early_49 + 0.01) / (global_early_48 + 0.01)
    
    df['prefix5_ratio'] = df['prefix5_ratio'].fillna(global_ratio)
    df['prefix4_ratio'] = df['prefix4_ratio'].fillna(global_ratio)
    df['p5_early_49'] = df['p5_early_49'].fillna(global_early_49)
    df['p4_early_49'] = df['p4_early_49'].fillna(global_early_49)
    
    early_49_imputed = []
    for idx, row in df.iterrows():
        if not np.isnan(row['early_49_obs']):
            early_49_imputed.append(row['early_49_obs'])
        elif not np.isnan(row['early_48']):
            early_49_imputed.append(row['early_48'] * row['prefix5_ratio'])
        else:
            early_49_imputed.append(row['p5_early_49'])
    df['early_49'] = early_49_imputed
    
    # Assign scaling early demand based on day
    df['early_demand'] = np.where(df['day'] == 48, df['early_48'], df['early_49'])
    
    # Scaling ratios
    df['geohash_ratio'] = (df['early_demand'] + 0.01) / (df['early_48'] + 0.01)
    df['scaled_demand48'] = df['demand48'] * df['geohash_ratio']
    
    # Neighborhood profiles of demand48
    p5_stats = grid48.groupby(['geohash_prefix5', 'time_minutes'])['demand48'].mean().reset_index().rename(columns={'demand48': 'p5_demand48_t'})
    p4_stats = grid48.groupby(['geohash_prefix4', 'time_minutes'])['demand48'].mean().reset_index().rename(columns={'demand48': 'p4_demand48_t'})
    df = df.merge(p5_stats, on=['geohash_prefix5', 'time_minutes'], how='left')
    df = df.merge(p4_stats, on=['geohash_prefix4', 'time_minutes'], how='left')
    df['p5_demand48_t'] = df['p5_demand48_t'].fillna(0.0)
    df['p4_demand48_t'] = df['p4_demand48_t'].fillna(0.0)
    
    # Clean up temporary columns
    cols_to_drop = [
        'sum', 'count', 'early_49_obs', 'prefix5_ratio', 'prefix4_ratio', 
        'p5_early_49', 'p4_early_49', 'p5_sum', 'p5_count', 'p4_sum', 'p4_count'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    feature_cols = [c for c in df.columns if c != 'demand']
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df
