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
    grid['demand_interpolated'] = grid.groupby('geohash')['demand'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both').fillna(0.0)
    )
    
    # Add prefix columns for neighborhood profiling on Day 48
    grid['geohash_prefix5'] = grid['geohash'].str[:5]
    grid['geohash_prefix4'] = grid['geohash'].str[:4]
    
    return grid

def build_features(df, train_df=None, is_train=True):
    """
    Builds leak-free geospatial, Day 48 temporal profiles, and same-day early morning scaling features.
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
        
    # Context data for Day 48 and Day 49 early morning
    ref_train = train_df if train_df is not None else df[df['demand'].notna()]
    
    # 5. Day 48 (previous day) historical features
    grid48 = get_day48_grid(ref_train)
    grid48_clean = grid48[['geohash', 'time_minutes', 'demand_interpolated']].copy()
    grid48_clean_dict = grid48_clean.set_index(['geohash', 'time_minutes'])['demand_interpolated'].to_dict()
    
    offsets = {
        'demand_day48_t': 0,
        'demand_day48_t_minus_15': -15,
        'demand_day48_t_minus_30': -30,
        'demand_day48_t_minus_60': -60,
        'demand_day48_t_plus_15': 15,
        'demand_day48_t_plus_30': 30,
        'demand_day48_t_plus_60': 60
    }
    
    for col_name, offset in offsets.items():
        df[col_name] = df.apply(
            lambda row: grid48_clean_dict.get((row['geohash'], np.clip(round((row['time_minutes'] + offset)/15)*15, 0, 1425)), 0.0),
            axis=1
        )
        
    # Window rolling statistics on Day 48
    df['demand_day48_rolling_mean_30'] = (df['demand_day48_t_minus_15'] + df['demand_day48_t'] + df['demand_day48_t_plus_15']) / 3.0
    df['demand_day48_rolling_mean_60'] = (df['demand_day48_t_minus_30'] + df['demand_day48_t_minus_15'] + df['demand_day48_t'] + df['demand_day48_t_plus_15'] + df['demand_day48_t_plus_30']) / 5.0
    
    # Day 48 overall geohash statistics
    stats48 = grid48.groupby('geohash')['demand_interpolated'].agg(['mean', 'std', 'max', 'min']).reset_index()
    stats48.columns = ['geohash', 'day48_overall_mean', 'day48_overall_std', 'day48_overall_max', 'day48_overall_min']
    df = df.merge(stats48, on='geohash', how='left')
    
    # Day 48 neighborhood stats
    prefix5_stats = grid48.groupby('geohash_prefix5')['demand_interpolated'].agg(['mean', 'std', 'max', 'min']).reset_index()
    prefix5_stats.columns = ['geohash_prefix5', 'day48_prefix5_mean', 'day48_prefix5_std', 'day48_prefix5_max', 'day48_prefix5_min']
    df = df.merge(prefix5_stats, on='geohash_prefix5', how='left')
    
    prefix4_stats = grid48.groupby('geohash_prefix4')['demand_interpolated'].agg(['mean', 'std', 'max', 'min']).reset_index()
    prefix4_stats.columns = ['geohash_prefix4', 'day48_prefix4_mean', 'day48_prefix4_std', 'day48_prefix4_max', 'day48_prefix4_min']
    df = df.merge(prefix4_stats, on='geohash_prefix4', how='left')
    
    # Day 48 neighborhood profiles at time t
    prefix5_profile = grid48.groupby(['geohash_prefix5', 'time_minutes'])['demand_interpolated'].mean().reset_index()
    prefix5_profile.columns = ['geohash_prefix5', 'time_minutes', 'demand_day48_prefix5_t']
    df = df.merge(prefix5_profile, on=['geohash_prefix5', 'time_minutes'], how='left')
    df['demand_day48_prefix5_t'] = df['demand_day48_prefix5_t'].fillna(0.0)
    
    prefix4_profile = grid48.groupby(['geohash_prefix4', 'time_minutes'])['demand_interpolated'].mean().reset_index()
    prefix4_profile.columns = ['geohash_prefix4', 'time_minutes', 'demand_day48_prefix4_t']
    df = df.merge(prefix4_profile, on=['geohash_prefix4', 'time_minutes'], how='left')
    df['demand_day48_prefix4_t'] = df['demand_day48_prefix4_t'].fillna(0.0)
    
    # Day 48 city-wide profile
    city_profile48 = grid48.groupby('time_minutes')['demand_interpolated'].mean().to_dict()
    df['city_profile_day48'] = df['time_minutes'].map(city_profile48)
    
    # 6. Same-day early morning scaling averages (Day 48 and Day 49)
    # Day 48 early morning average
    day48_clean = ref_train[ref_train['day'] == 48].copy()
    day48_clean = parse_time_features(day48_clean)
    d48_early = day48_clean[day48_clean['time_minutes'] <= 120].groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_48'})
    df = df.merge(d48_early, on='geohash', how='left')
    df['early_48'] = df['early_48'].fillna(df['early_48'].mean())
    
    # Day 49 early morning average (non-leaky)
    d49_early_rows = ref_train[(ref_train['day'] == 49) & (parse_time_features(ref_train)['time_minutes'] <= 120)].copy()
    
    # Group by geohash and sum/count the early morning demand
    early_stats = d49_early_rows.groupby('geohash')['demand'].agg(['sum', 'count']).reset_index()
    df = df.merge(early_stats, on='geohash', how='left')
    df['sum'] = df['sum'].fillna(0.0)
    df['count'] = df['count'].fillna(0)
    
    # Non-leaky early_49: subtract current row's demand if it is part of Day 49 early morning
    is_d49_early = (df['day'] == 49) & (df['time_minutes'] <= 120)
    df['early_49'] = np.where(
        is_d49_early,
        (df['sum'] - df['demand'].fillna(0.0)) / (df['count'] - 1).clip(lower=1),
        df['sum'] / df['count'].clip(lower=1)
    )
    df['early_49'] = df['early_49'].fillna(df['early_49'].mean())
    
    # Drop temporary sum and count columns
    df = df.drop(columns=['sum', 'count'])
    
    # Day 49 early morning neighborhood averages (non-leaky)
    early_by_geohash = d49_early_rows.groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_mean_g'})
    early_by_geohash['geohash_prefix5'] = early_by_geohash['geohash'].str[:5]
    early_by_geohash['geohash_prefix4'] = early_by_geohash['geohash'].str[:4]
    
    prefix5_sums = early_by_geohash.groupby('geohash_prefix5')['early_mean_g'].agg(['sum', 'count']).reset_index()
    prefix5_sums.columns = ['geohash_prefix5', 'prefix5_sum', 'prefix5_count']
    
    prefix4_sums = early_by_geohash.groupby('geohash_prefix4')['early_mean_g'].agg(['sum', 'count']).reset_index()
    prefix4_sums.columns = ['geohash_prefix4', 'prefix4_sum', 'prefix4_count']
    
    early_by_geohash = early_by_geohash.merge(prefix5_sums, on='geohash_prefix5', how='left')
    early_by_geohash = early_by_geohash.merge(prefix4_sums, on='geohash_prefix4', how='left')
    
    early_by_geohash['early_49_prefix5'] = (early_by_geohash['prefix5_sum'] - early_by_geohash['early_mean_g']) / (early_by_geohash['prefix5_count'] - 1).clip(lower=1)
    early_by_geohash['early_49_prefix4'] = (early_by_geohash['prefix4_sum'] - early_by_geohash['early_mean_g']) / (early_by_geohash['prefix4_count'] - 1).clip(lower=1)
    
    df = df.merge(early_by_geohash[['geohash', 'early_49_prefix5', 'early_49_prefix4']], on='geohash', how='left')
    df['early_49_prefix5'] = df['early_49_prefix5'].fillna(df['early_49_prefix5'].mean())
    df['early_49_prefix4'] = df['early_49_prefix4'].fillna(df['early_49_prefix4'].mean())
    
    # Final safety checks for all non-target columns
    feature_cols = [c for c in df.columns if c != 'demand']
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df

