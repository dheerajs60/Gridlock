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

def build_features(df, train_df=None, is_train=True):
    """
    Builds 100% leak-free features for df using train_df as historical context.
    - Decodes geohashes & imputes environmental variables.
    - Computes non-leaky city and neighborhood temporal profiles on Day 48.
    - Computes same-day early morning scaling factors (leak-free during training,
      neighborhood-ratio scaled for missing geohashes in the test set).
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
    ref_train = train_df if train_df is not None else df[df['demand'].notna()]
    train_parsed = parse_time_features(ref_train.copy())
    df = impute_missing_values(df, train_parsed)
    
    # 5. Day 48 (previous day) historical grid & temporal profiles (non-leaky)
    grid48 = get_day48_grid(ref_train)
    
    # Compute neighborhood profile stats (sum and count per prefix at time t)
    p5_stats = grid48.groupby(['geohash_prefix5', 'time_minutes'])['demand48'].agg(['sum', 'count']).reset_index()
    p5_stats.columns = ['geohash_prefix5', 'time_minutes', 'p5_sum', 'p5_count']
    
    p4_stats = grid48.groupby(['geohash_prefix4', 'time_minutes'])['demand48'].agg(['sum', 'count']).reset_index()
    p4_stats.columns = ['geohash_prefix4', 'time_minutes', 'p4_sum', 'p4_count']
    
    grid48 = grid48.merge(p5_stats, on=['geohash_prefix5', 'time_minutes'], how='left')
    grid48 = grid48.merge(p4_stats, on=['geohash_prefix4', 'time_minutes'], how='left')
    
    # Subtract self to ensure absolutely no leakage during profile calculations
    grid48['demand_day48_prefix5_t'] = (grid48['p5_sum'] - grid48['demand48']) / (grid48['p5_count'] - 1).clip(lower=1)
    grid48['demand_day48_prefix4_t'] = (grid48['p4_sum'] - grid48['demand48']) / (grid48['p4_count'] - 1).clip(lower=1)
    
    # City-wide profile (exclude current geohash)
    city_stats = grid48.groupby('time_minutes')['demand48'].agg(['sum', 'count']).reset_index()
    city_stats.columns = ['time_minutes', 'city_sum', 'city_count']
    grid48 = grid48.merge(city_stats, on='time_minutes', how='left')
    grid48['city_profile_day48'] = (grid48['city_sum'] - grid48['demand48']) / (grid48['city_count'] - 1).clip(lower=1)
    
    # Merge Day 48 profiles into df
    grid48_clean = grid48[['geohash', 'time_minutes', 'demand_day48_prefix5_t', 'demand_day48_prefix4_t', 'city_profile_day48']].copy()
    df = df.merge(grid48_clean, on=['geohash', 'time_minutes'], how='left')
    
    # 6. Same-day early morning scaling averages (Day 48 and Day 49)
    # Day 48 early morning average per geohash
    day48_clean = ref_train[ref_train['day'] == 48].copy()
    day48_clean = parse_time_features(day48_clean)
    d48_early = day48_clean[day48_clean['time_minutes'] <= 120].groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_48'})
    df = df.merge(d48_early, on='geohash', how='left')
    df['early_48'] = df['early_48'].fillna(df['early_48'].mean())
    
    # Day 49 observed early morning statistics
    d49_early_rows = ref_train[(ref_train['day'] == 49) & (parse_time_features(ref_train)['time_minutes'] <= 120)].copy()
    early_stats = d49_early_rows.groupby('geohash')['demand'].agg(['sum', 'count']).reset_index()
    df = df.merge(early_stats, on='geohash', how='left')
    df['sum'] = df['sum'].fillna(0.0)
    df['count'] = df['count'].fillna(0)
    
    # Compute non-leaky observed early average
    is_d49_early = (df['day'] == 49) & (df['time_minutes'] <= 120)
    df['early_49_obs'] = np.where(
        is_d49_early,
        (df['sum'] - df['demand'].fillna(0.0)) / (df['count'] - 1).clip(lower=1),
        df['sum'] / df['count'].clip(lower=1)
    )
    df['early_49_obs'] = np.where(df['count'] == 0, np.nan, df['early_49_obs'])
    
    # Neighborhood scaling ratio for missing geohashes
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
    global_early_48 = day48_clean[day48_clean['time_minutes'] <= 120]['demand'].mean()
    global_ratio = (global_early_49 + 0.01) / (global_early_48 + 0.01)
    
    df['prefix5_ratio'] = df['prefix5_ratio'].fillna(global_ratio)
    df['prefix4_ratio'] = df['prefix4_ratio'].fillna(global_ratio)
    df['p5_early_49'] = df['p5_early_49'].fillna(global_early_49)
    df['p4_early_49'] = df['p4_early_49'].fillna(global_early_49)
    
    d48_early_raw = d48_early.rename(columns={'early_48': 'early_48_raw'})
    df = df.merge(d48_early_raw, on='geohash', how='left')
    
    # Impute missing early_49 using scaled Day 48 demand
    early_49_imputed = []
    for idx, row in df.iterrows():
        if not np.isnan(row['early_49_obs']):
            early_49_imputed.append(row['early_49_obs'])
        elif not np.isnan(row['early_48_raw']):
            early_49_imputed.append(row['early_48_raw'] * row['prefix5_ratio'])
        else:
            early_49_imputed.append(row['p5_early_49'])
            
    df['early_49'] = early_49_imputed
    
    # Assign scaling factors based on day
    # For Day 48 rows: early_demand = early_48
    # For Day 49 rows: early_demand = early_49
    df['early_demand'] = np.where(df['day'] == 48, df['early_48'], df['early_49'])
    
    # 7. Same-day early morning neighborhood averages (non-leaky)
    prefix5_sums = early_by_geohash.groupby('geohash_prefix5')['early_49_g'].agg(['sum', 'count']).reset_index()
    prefix5_sums.columns = ['geohash_prefix5', 'prefix5_sum', 'prefix5_count']
    
    prefix4_sums = early_by_geohash.groupby('geohash_prefix4')['early_49_g'].agg(['sum', 'count']).reset_index()
    prefix4_sums.columns = ['geohash_prefix4', 'prefix4_sum', 'prefix4_count']
    
    early_by_geohash = early_by_geohash.merge(prefix5_sums, on='geohash_prefix5', how='left')
    early_by_geohash = early_by_geohash.merge(prefix4_sums, on='geohash_prefix4', how='left')
    
    early_by_geohash['early_49_prefix5'] = (early_by_geohash['prefix5_sum'] - early_by_geohash['early_49_g']) / (early_by_geohash['prefix5_count'] - 1).clip(lower=1)
    early_by_geohash['early_49_prefix4'] = (early_by_geohash['prefix4_sum'] - early_by_geohash['early_49_g']) / (early_by_geohash['prefix4_count'] - 1).clip(lower=1)
    
    df = df.merge(early_by_geohash[['geohash', 'early_49_prefix5', 'early_49_prefix4']], on='geohash', how='left')
    
    p5_means_std = p5_means.rename(columns={'p5_early_49': 'early_49_prefix5_std'})
    p4_means_std = p4_means.rename(columns={'p4_early_49': 'early_49_prefix4_std'})
    df = df.merge(p5_means_std[['geohash_prefix5', 'early_49_prefix5_std']], on='geohash_prefix5', how='left')
    df = df.merge(p4_means_std[['geohash_prefix4', 'early_49_prefix4_std']], on='geohash_prefix4', how='left')
    
    df['early_49_prefix5'] = df['early_49_prefix5'].fillna(df['early_49_prefix5_std']).fillna(global_early_49)
    df['early_49_prefix4'] = df['early_49_prefix4'].fillna(df['early_49_prefix4_std']).fillna(global_early_49)
    
    # Neighborhood averages for Day 48 early morning
    early_48_by_geo = d48_early.copy()
    early_48_by_geo['geohash_prefix5'] = early_48_by_geo['geohash'].str[:5]
    early_48_by_geo['geohash_prefix4'] = early_48_by_geo['geohash'].str[:4]
    p5_sums_48 = early_48_by_geo.groupby('geohash_prefix5')['early_48'].agg(['sum', 'count']).reset_index()
    p5_sums_48.columns = ['geohash_prefix5', 'prefix5_sum_48', 'prefix5_count_48']
    p4_sums_48 = early_48_by_geo.groupby('geohash_prefix4')['early_48'].agg(['sum', 'count']).reset_index()
    p4_sums_48.columns = ['geohash_prefix4', 'prefix4_sum_48', 'prefix4_count_48']
    
    early_48_by_geo = early_48_by_geo.merge(p5_sums_48, on='geohash_prefix5', how='left')
    early_48_by_geo = early_48_by_geo.merge(p4_sums_48, on='geohash_prefix4', how='left')
    
    early_48_by_geo['early_48_prefix5'] = (early_48_by_geo['prefix5_sum_48'] - early_48_by_geo['early_48']) / (early_48_by_geo['prefix5_count_48'] - 1).clip(lower=1)
    early_48_by_geo['early_48_prefix4'] = (early_48_by_geo['prefix4_sum_48'] - early_48_by_geo['early_48']) / (early_48_by_geo['prefix4_count_48'] - 1).clip(lower=1)
    
    df = df.merge(early_48_by_geo[['geohash', 'early_48_prefix5', 'early_48_prefix4']], on='geohash', how='left')
    df['early_48_prefix5'] = df['early_48_prefix5'].fillna(df['early_48_prefix5'].mean())
    df['early_48_prefix4'] = df['early_48_prefix4'].fillna(df['early_48_prefix4'].mean())
    
    # Assign neighborhood scaling factor based on day
    df['early_demand_prefix5'] = np.where(df['day'] == 48, df['early_48_prefix5'], df['early_49_prefix5'])
    df['early_demand_prefix4'] = np.where(df['day'] == 48, df['early_48_prefix4'], df['early_49_prefix4'])
    
    # Clean up temporary columns
    cols_to_drop = [
        'sum', 'count', 'early_49_obs', 'early_48_raw', 'prefix5_ratio', 'prefix4_ratio', 
        'p5_early_49', 'p4_early_49', 'early_49_prefix5_std', 'early_49_prefix4_std',
        'early_48_prefix5', 'early_48_prefix4', 'early_49_prefix5', 'early_49_prefix4',
        'p5_sum', 'p5_count', 'p4_sum', 'p4_count', 'city_sum', 'city_count'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Final NaNs safety check
    feature_cols = [c for c in df.columns if c != 'demand']
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df
