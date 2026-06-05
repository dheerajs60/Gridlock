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

def build_features(df, train_df, is_train=True):
    """
    Builds features for df using train_df as context/history.
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
        
    # 5. Same-day lag features (lags from 0:00 to 2:00)
    # Filter for same-day history (up to 2:00 AM / 120 minutes)
    history_0_2 = df[df['time_minutes'] <= 120].copy()
    
    # Pivot to get geohash x day x time_minutes demand
    pivot_0_2 = history_0_2.pivot_table(
        index=['geohash', 'day'], 
        columns='time_minutes', 
        values='demand'
    ).reset_index()
    
    # Ensure all required timestamps are present in pivot columns
    for t in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
        if t not in pivot_0_2.columns:
            pivot_0_2[t] = np.nan
            
    # Interpolate same-day lags row-wise (for each geohash/day)
    lag_cols = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    pivot_0_2[lag_cols] = pivot_0_2[lag_cols].interpolate(axis=1, limit_direction='both').fillna(0.0)
    
    # Compute mean over the 0:00-2:00 window
    pivot_0_2['mean_demand_0_to_2'] = pivot_0_2[lag_cols].mean(axis=1)
    
    # Rename columns for merging
    pivot_0_2 = pivot_0_2.rename(columns={
        120: 'demand_at_2_00',
        105: 'demand_at_1_45',
        90: 'demand_at_1_30',
        75: 'demand_at_1_15',
        60: 'demand_at_1_00'
    })
    
    # Compute neighborhood level same-day lags
    pivot_0_2['geohash_prefix5'] = pivot_0_2['geohash'].str[:5]
    pivot_0_2['geohash_prefix4'] = pivot_0_2['geohash'].str[:4]
    
    prefix5_early = pivot_0_2.groupby(['geohash_prefix5', 'day'])['mean_demand_0_to_2'].mean().reset_index()
    prefix5_early.columns = ['geohash_prefix5', 'day', 'mean_demand_0_to_2_prefix5']
    
    prefix4_early = pivot_0_2.groupby(['geohash_prefix4', 'day'])['mean_demand_0_to_2'].mean().reset_index()
    prefix4_early.columns = ['geohash_prefix4', 'day', 'mean_demand_0_to_2_prefix4']
    
    # Merge same-day lags into df
    df = df.merge(
        pivot_0_2[['geohash', 'day', 'demand_at_2_00', 'demand_at_1_45', 'demand_at_1_30', 'demand_at_1_15', 'demand_at_1_00', 'mean_demand_0_to_2']], 
        on=['geohash', 'day'], 
        how='left'
    )
    df = df.merge(prefix5_early, on=['geohash_prefix5', 'day'], how='left')
    df = df.merge(prefix4_early, on=['geohash_prefix4', 'day'], how='left')
    
    # Fill missing same-day lags with neighborhood values
    df['mean_demand_0_to_2'] = df['mean_demand_0_to_2'].fillna(df['mean_demand_0_to_2_prefix5']).fillna(df['mean_demand_0_to_2_prefix4']).fillna(0.0)
    for col in ['demand_at_2_00', 'demand_at_1_45', 'demand_at_1_30', 'demand_at_1_15', 'demand_at_1_00']:
        df[col] = df[col].fillna(df['mean_demand_0_to_2'])
        
    # 6. Previous day (Day 48) historical features
    if train_df is not None:
        grid48 = get_day48_grid(train_df)
        grid48_clean = grid48[['geohash', 'time_minutes', 'demand_interpolated']].copy()
        
        # We merge Day 48 demand at offset times
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
            temp_time_col = f'temp_time_{offset}'
            df[temp_time_col] = (df['time_minutes'] + offset).clip(0, 1425)
            df[temp_time_col] = (df[temp_time_col] / 15.0).round().astype(int) * 15
            
            df = df.merge(
                grid48_clean.rename(columns={'time_minutes': temp_time_col, 'demand_interpolated': col_name}),
                on=['geohash', temp_time_col],
                how='left'
            )
            df[col_name] = df[col_name].fillna(0.0)
            df = df.drop(columns=[temp_time_col])
        
        # Window rolling statistics on Day 48
        df['demand_day48_rolling_mean_30'] = (df['demand_day48_t_minus_15'] + df['demand_day48_t'] + df['demand_day48_t_plus_15']) / 3.0
        df['demand_day48_rolling_mean_60'] = (df['demand_day48_t_minus_30'] + df['demand_day48_t_minus_15'] + df['demand_day48_t'] + df['demand_day48_t_plus_15'] + df['demand_day48_t_plus_30']) / 5.0
        
        # Day 48 overall geohash statistics
        stats48 = grid48.groupby('geohash')['demand_interpolated'].agg(['mean', 'std', 'max', 'min']).reset_index()
        stats48.columns = ['geohash', 'day48_overall_mean', 'day48_overall_std', 'day48_overall_max', 'day48_overall_min']
        df = df.merge(stats48, on='geohash', how='left')
        
        # Neighborhood stats on Day 48
        prefix5_stats = grid48.groupby('geohash_prefix5')['demand_interpolated'].agg(['mean', 'std', 'max', 'min']).reset_index()
        prefix5_stats.columns = ['geohash_prefix5', 'day48_prefix5_mean', 'day48_prefix5_std', 'day48_prefix5_max', 'day48_prefix5_min']
        df = df.merge(prefix5_stats, on='geohash_prefix5', how='left')
        
        prefix4_stats = grid48.groupby('geohash_prefix4')['demand_interpolated'].agg(['mean', 'std', 'max', 'min']).reset_index()
        prefix4_stats.columns = ['geohash_prefix4', 'day48_prefix4_mean', 'day48_prefix4_std', 'day48_prefix4_max', 'day48_prefix4_min']
        df = df.merge(prefix4_stats, on='geohash_prefix4', how='left')
        
        # Neighborhood profiles on Day 48 at time t
        prefix5_profile = grid48.groupby(['geohash_prefix5', 'time_minutes'])['demand_interpolated'].mean().reset_index()
        prefix5_profile.columns = ['geohash_prefix5', 'time_minutes', 'demand_day48_prefix5_t']
        
        prefix4_profile = grid48.groupby(['geohash_prefix4', 'time_minutes'])['demand_interpolated'].mean().reset_index()
        prefix4_profile.columns = ['geohash_prefix4', 'time_minutes', 'demand_day48_prefix4_t']
        
        df = df.merge(prefix5_profile, on=['geohash_prefix5', 'time_minutes'], how='left')
        df = df.merge(prefix4_profile, on=['geohash_prefix4', 'time_minutes'], how='left')
        
        df['demand_day48_prefix5_t'] = df['demand_day48_prefix5_t'].fillna(0.0)
        df['demand_day48_prefix4_t'] = df['demand_day48_prefix4_t'].fillna(0.0)
        
        # City-wide profile
        city_profile48 = grid48.groupby('time_minutes')['demand_interpolated'].mean().to_dict()
        df['city_profile_day48'] = df['time_minutes'].map(city_profile48)
    else:
        # Fallback values if no train_df is provided
        cols_to_nan = [
            'demand_day48_t', 'demand_day48_t_minus_15', 'demand_day48_t_minus_30', 'demand_day48_t_minus_60', 
            'demand_day48_t_plus_15', 'demand_day48_t_plus_30', 'demand_day48_t_plus_60', 
            'demand_day48_rolling_mean_30', 'demand_day48_rolling_mean_60', 
            'day48_overall_mean', 'day48_overall_std', 'day48_overall_max', 'day48_overall_min',
            'day48_prefix5_mean', 'day48_prefix5_std', 'day48_prefix5_max', 'day48_prefix5_min',
            'day48_prefix4_mean', 'day48_prefix4_std', 'day48_prefix4_max', 'day48_prefix4_min',
            'demand_day48_prefix5_t', 'demand_day48_prefix4_t', 'city_profile_day48'
        ]
        for col in cols_to_nan:
            df[col] = 0.0
            
    # Final NaNs safety check for features only (not target)
    feature_cols = [c for c in df.columns if c != 'demand']
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df
