"""
Improved Traffic Demand Prediction - v2

Key improvements over the 90.14% submission:
1. Train on Day 48 FULL DAY (0:00-23:45) + Day 49 early morning (0:00-2:00)
   - Fixes the domain shift: trees now see all time_minutes 0-825, not just 0-120
2. Regularized geohash_ratio to prevent extreme scaling (no clipping to 1.0)
3. Richer features: demand48_t lags, rolling averages, neighborhood profiles
4. 3-model ensemble (LightGBM + XGBoost + CatBoost)
5. Proper cross-validation with temporal awareness
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: Load Data
# =====================================================================
print("Step 1: Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# =====================================================================
# STEP 2: Basic Feature Engineering
# =====================================================================
print("Step 2: Feature Engineering...")

def parse_time(df):
    df = df.copy()
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    df['sin_time'] = np.sin(2 * np.pi * df['time_minutes'] / 1440.0)
    df['cos_time'] = np.cos(2 * np.pi * df['time_minutes'] / 1440.0)
    return df

def decode_geohash(geohash):
    if not isinstance(geohash, str) or len(geohash) == 0:
        return np.nan, np.nan
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for char in geohash:
        if char not in base32:
            return np.nan, np.nan
        val = base32.index(char)
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                lon_interval = (mid, lon_interval[1]) if bit == 1 else (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                lat_interval = (mid, lat_interval[1]) if bit == 1 else (lat_interval[0], mid)
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2.0, (lon_interval[0] + lon_interval[1]) / 2.0

def impute_missing(df, ref_df=None):
    df = df.copy()
    ref = ref_df if ref_df is not None else df
    road_mode = ref['RoadType'].mode()[0] if not ref['RoadType'].mode().empty else 'Residential'
    weather_mode = ref['Weather'].mode()[0] if not ref['Weather'].mode().empty else 'Sunny'
    df['RoadType'] = df['RoadType'].fillna(road_mode)
    df['Weather'] = df['Weather'].fillna(weather_mode)
    ref['hour'] = ref['timestamp'].apply(lambda x: int(x.split(':')[0]))
    temp_map = ref.groupby(['hour', 'Weather'])['Temperature'].mean().to_dict()
    overall_temp = ref['Temperature'].mean()
    if np.isnan(overall_temp): overall_temp = 20.0
    def fill_temp(row):
        if not np.isnan(row['Temperature']): return row['Temperature']
        key = (row['hour'], row['Weather'])
        if key in temp_map and not np.isnan(temp_map[key]): return temp_map[key]
        return overall_temp
    df['Temperature'] = df.apply(fill_temp, axis=1)
    return df

# Parse time features
for df in [train, test]:
    parse_time_result = parse_time(df)
    df['hour'] = parse_time_result['hour']
    df['minute'] = parse_time_result['minute']
    df['time_minutes'] = parse_time_result['time_minutes']
    df['sin_time'] = parse_time_result['sin_time']
    df['cos_time'] = parse_time_result['cos_time']

# Impute missing
train = impute_missing(train)
test = impute_missing(test, train)

# Geohash decode
for df in [train, test]:
    lats_lons = df['geohash'].apply(decode_geohash)
    df['latitude'] = [x[0] for x in lats_lons]
    df['longitude'] = [x[1] for x in lats_lons]
    df['geohash_prefix5'] = df['geohash'].str[:5]
    df['geohash_prefix4'] = df['geohash'].str[:4]

# =====================================================================
# STEP 3: Build Day 48 Interpolated Grid
# =====================================================================
print("Step 3: Building Day 48 grid...")
d48 = train[train['day'] == 48].copy()
d49_train = train[train['day'] == 49].copy()

# Full interpolated grid of Day 48 demand for all geohashes at all times
geohashes_48 = d48['geohash'].unique()
times_48 = sorted(d48['time_minutes'].unique())
grid48 = pd.MultiIndex.from_product([geohashes_48, times_48], names=['geohash', 'time_minutes']).to_frame().reset_index(drop=True)
grid48 = grid48.merge(d48[['geohash', 'time_minutes', 'demand']], on=['geohash', 'time_minutes'], how='left')
grid48 = grid48.sort_values(['geohash', 'time_minutes'])
grid48['demand48'] = grid48.groupby('geohash')['demand'].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both').fillna(0.0)
)
grid48['geohash_prefix5'] = grid48['geohash'].str[:5]
grid48['geohash_prefix4'] = grid48['geohash'].str[:4]

# Compute lag features on the grid
grid48 = grid48.sort_values(['geohash', 'time_minutes'])
grid48['demand48_lag15'] = grid48.groupby('geohash')['demand48'].shift(1).fillna(0.0)
grid48['demand48_lag30'] = grid48.groupby('geohash')['demand48'].shift(2).fillna(0.0)
grid48['demand48_lag60'] = grid48.groupby('geohash')['demand48'].shift(4).fillna(0.0)
grid48['demand48_lead15'] = grid48.groupby('geohash')['demand48'].shift(-1).fillna(0.0)
grid48['demand48_lead30'] = grid48.groupby('geohash')['demand48'].shift(-2).fillna(0.0)

# Rolling mean (3-window = 45 min)
grid48['demand48_roll3'] = (grid48['demand48_lag15'] + grid48['demand48'] + grid48['demand48_lead15']) / 3.0
# Rolling mean (5-window = 75 min)
grid48['demand48_roll5'] = (grid48['demand48_lag30'] + grid48['demand48_lag15'] + grid48['demand48'] + grid48['demand48_lead15'] + grid48['demand48_lead30']) / 5.0

# Neighborhood profiles at each time on Day 48
p5_time = grid48.groupby(['geohash_prefix5', 'time_minutes'])['demand48'].mean().reset_index().rename(columns={'demand48': 'p5_demand48_t'})
p4_time = grid48.groupby(['geohash_prefix4', 'time_minutes'])['demand48'].mean().reset_index().rename(columns={'demand48': 'p4_demand48_t'})
city_time = grid48.groupby('time_minutes')['demand48'].mean().to_dict()

# Overall stats per geohash on Day 48
g48_stats = grid48.groupby('geohash')['demand48'].agg(['mean', 'std', 'max', 'min']).reset_index()
g48_stats.columns = ['geohash', 'g48_overall_mean', 'g48_overall_std', 'g48_overall_max', 'g48_overall_min']

# =====================================================================
# STEP 4: Compute Early Morning Averages & Scaling Ratios
# =====================================================================
print("Step 4: Computing scaling ratios...")

# Day 48 early morning average per geohash (0:00-2:00)
d48_early = d48[d48['time_minutes'] <= 120].groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_48'})
d48_early_mean = d48_early['early_48'].mean()

# Day 49 early morning stats per geohash
d49_early = d49_train[d49_train['time_minutes'] <= 120].copy()
early_stats_49 = d49_early.groupby('geohash')['demand'].agg(['sum', 'count']).reset_index()
early_stats_49.columns = ['geohash', 'early49_sum', 'early49_count']

# Neighborhood ratios for imputation
early_by_gh = d49_early.groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_49_g'})
early_by_gh = early_by_gh.merge(d48_early, on='geohash', how='left')
early_by_gh['early_48'] = early_by_gh['early_48'].fillna(d48_early_mean)
early_by_gh['geohash_prefix5'] = early_by_gh['geohash'].str[:5]
early_by_gh['geohash_prefix4'] = early_by_gh['geohash'].str[:4]

p5_ratio = early_by_gh.groupby('geohash_prefix5')[['early_49_g', 'early_48']].mean().reset_index()
p5_ratio['p5_ratio'] = (p5_ratio['early_49_g'] + 0.01) / (p5_ratio['early_48'] + 0.01)
p5_ratio['p5_early_49'] = p5_ratio['early_49_g']
p5_ratio = p5_ratio[['geohash_prefix5', 'p5_ratio', 'p5_early_49']]

p4_ratio = early_by_gh.groupby('geohash_prefix4')[['early_49_g', 'early_48']].mean().reset_index()
p4_ratio['p4_ratio'] = (p4_ratio['early_49_g'] + 0.01) / (p4_ratio['early_48'] + 0.01)
p4_ratio['p4_early_49'] = p4_ratio['early_49_g']
p4_ratio = p4_ratio[['geohash_prefix4', 'p4_ratio', 'p4_early_49']]

global_early_49 = d49_early['demand'].mean()
global_early_48 = d48_early['early_48'].mean()
global_ratio = (global_early_49 + 0.01) / (global_early_48 + 0.01)
print(f"  Global ratio Day49/Day48 early morning: {global_ratio:.4f}")

def compute_features(df, is_train_d49=False):
    """Compute all features for a dataframe. is_train_d49=True triggers LOO for Day 49 early morning."""
    df = df.copy()
    
    # 1. Merge demand48 and lag features
    lag_cols = ['geohash', 'time_minutes', 'demand48', 'demand48_lag15', 'demand48_lag30', 
                'demand48_lag60', 'demand48_lead15', 'demand48_lead30', 'demand48_roll3', 'demand48_roll5']
    df = df.merge(grid48[lag_cols], on=['geohash', 'time_minutes'], how='left')
    df['demand48'] = df['demand48'].fillna(0.0)
    for col in ['demand48_lag15', 'demand48_lag30', 'demand48_lag60', 'demand48_lead15', 'demand48_lead30', 'demand48_roll3', 'demand48_roll5']:
        df[col] = df[col].fillna(0.0)

    # 2. Neighborhood profiles at time t
    df = df.merge(p5_time, on=['geohash_prefix5', 'time_minutes'], how='left')
    df = df.merge(p4_time, on=['geohash_prefix4', 'time_minutes'], how='left')
    df['p5_demand48_t'] = df['p5_demand48_t'].fillna(0.0)
    df['p4_demand48_t'] = df['p4_demand48_t'].fillna(0.0)
    df['city_profile_day48'] = df['time_minutes'].map(city_time).fillna(0.0)
    
    # 3. Overall stats per geohash on Day 48
    df = df.merge(g48_stats, on='geohash', how='left')
    for col in ['g48_overall_mean', 'g48_overall_std', 'g48_overall_max', 'g48_overall_min']:
        df[col] = df[col].fillna(0.0)
    
    # 4. Early 48 average
    df = df.merge(d48_early, on='geohash', how='left')
    df['early_48'] = df['early_48'].fillna(d48_early_mean)
    
    # 5. Early 49 observations (non-leaky)
    df = df.merge(early_stats_49, on='geohash', how='left')
    df['early49_sum'] = df['early49_sum'].fillna(0.0)
    df['early49_count'] = df['early49_count'].fillna(0)
    
    if is_train_d49:
        # Leave-one-out: only apply to Day 49 early morning rows (training)
        is_d49_early = (df['day'] == 49) & (df['time_minutes'] <= 120) & (df['demand'].notna())
        df['early_49_obs'] = np.where(
            is_d49_early,
            (df['early49_sum'] - df['demand'].fillna(0.0)) / (df['early49_count'] - 1).clip(lower=1),
            df['early49_sum'] / df['early49_count'].clip(lower=1)
        )
        df['early_49_obs'] = np.where(df['early49_count'] == 0, np.nan, df['early_49_obs'])
    else:
        # For test: use full average
        df['early_49_obs'] = np.where(
            df['early49_count'] == 0,
            np.nan,
            df['early49_sum'] / df['early49_count'].clip(lower=1)
        )
    
    # 6. Impute early_49 using neighborhood ratios
    df = df.merge(p5_ratio, on='geohash_prefix5', how='left')
    df = df.merge(p4_ratio, on='geohash_prefix4', how='left')
    df['p5_ratio'] = df['p5_ratio'].fillna(global_ratio)
    df['p4_ratio'] = df['p4_ratio'].fillna(global_ratio)
    df['p5_early_49'] = df['p5_early_49'].fillna(global_early_49)
    df['p4_early_49'] = df['p4_early_49'].fillna(global_early_49)
    
    def impute_early49(row):
        if not np.isnan(row['early_49_obs']) and row['early49_count'] > 0:
            return row['early_49_obs']
        if not np.isnan(row['early_48']) and row['early_48'] > 0:
            return row['early_48'] * row['p5_ratio']
        return row['p5_early_49']
    df['early_49'] = df.apply(impute_early49, axis=1)
    
    # 7. Regularized geohash_ratio (C=0.2 based on validation tuning)
    # This prevents extreme ratios from causing clipping
    C = 0.2
    df['geohash_ratio'] = (df['early_49'] + C * global_early_49) / (df['early_48'] + C * global_early_48)
    # Also cap extreme ratios
    df['geohash_ratio'] = df['geohash_ratio'].clip(0.1, 4.0)
    
    # 8. Scaled baseline (capped)
    df['scaled_demand48'] = df['demand48'] * df['geohash_ratio']
    
    # 9. Scaled lag features
    df['scaled_demand48_lag15'] = df['demand48_lag15'] * df['geohash_ratio']
    df['scaled_demand48_lead15'] = df['demand48_lead15'] * df['geohash_ratio']
    df['scaled_p5_demand48_t'] = df['p5_demand48_t'] * df['geohash_ratio']
    df['scaled_p4_demand48_t'] = df['p4_demand48_t'] * df['geohash_ratio']
    df['scaled_city_profile'] = df['city_profile_day48'] * df['geohash_ratio']
    
    # 10. Interaction features
    df['ratio_x_time'] = df['geohash_ratio'] * df['sin_time']
    df['early49_x_time'] = df['early_49'] * df['time_minutes'] / 825.0
    df['demand48_x_ratio'] = df['demand48'] * df['geohash_ratio']  # = scaled_demand48
    
    # Clean up
    cols_to_drop = ['early49_sum', 'early49_count', 'early_49_obs', 'p5_ratio', 'p4_ratio', 'p5_early_49', 'p4_early_49']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.fillna(0.0)
    
    return df

# =====================================================================
# STEP 5: Prepare Training and Test DataFrames
# =====================================================================
print("Step 5: Preparing features...")

# For training, we use Day 48 FULL DAY + Day 49 early morning
# This fixes the domain shift problem!
train_raw = pd.concat([d48, d49_train], ignore_index=True)
all_raw = pd.concat([train_raw, test], ignore_index=True)

# Compute features for training
print("  Computing training features (Day 48 full day + Day 49 early morning)...")
train_feat = compute_features(train_raw, is_train_d49=True)

print("  Computing test features...")
test_feat = compute_features(test, is_train_d49=False)

# Categorical encoding
cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash', 'geohash_prefix4', 'geohash_prefix5']
# Fit on combined data for consistent encoding
all_cats = pd.concat([train_feat[cat_cols], test_feat[cat_cols]], ignore_index=True)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(all_cats.astype(str))
train_feat[cat_cols] = encoder.transform(train_feat[cat_cols].astype(str))
test_feat[cat_cols] = encoder.transform(test_feat[cat_cols].astype(str))

# Feature list
features = [
    # Categorical
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    # Spatial
    'geohash', 'geohash_prefix4', 'geohash_prefix5', 'latitude', 'longitude',
    # Time
    'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time',
    # Day 48 demand and lags
    'demand48', 'demand48_lag15', 'demand48_lag30', 'demand48_lag60', 
    'demand48_lead15', 'demand48_lead30', 'demand48_roll3', 'demand48_roll5',
    # Neighborhood profiles
    'p5_demand48_t', 'p4_demand48_t', 'city_profile_day48',
    # Day 48 overall stats
    'g48_overall_mean', 'g48_overall_std', 'g48_overall_max', 'g48_overall_min',
    # Early morning averages and ratios
    'early_48', 'early_49', 'geohash_ratio',
    # Scaled features
    'scaled_demand48', 'scaled_demand48_lag15', 'scaled_demand48_lead15',
    'scaled_p5_demand48_t', 'scaled_p4_demand48_t', 'scaled_city_profile',
    # Interaction features
    'ratio_x_time', 'early49_x_time',
]

target = 'demand'

# =====================================================================
# STEP 6: Cross Validation Strategy
# =====================================================================
print("Step 6: Setting up cross-validation...")

# Key insight: Day 48 rows have HUGE N (69427), Day 49 has 7872
# We validate on Day 49 early morning only (the proxy for test)
# Training: Day 48 full day + part of Day 49 early morning

train_d48_feat = train_feat[train_feat['day'] == 48].copy()
train_d49_feat = train_feat[train_feat['day'] == 49].copy()

print(f"  Day 48 training rows: {len(train_d48_feat)}")
print(f"  Day 49 early morning rows: {len(train_d49_feat)}")
print(f"  Test (Day 49 daytime) rows: {len(test_feat)}")

# 5-fold CV on Day 49 early morning
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_d49_feat))
oof_xgb = np.zeros(len(train_d49_feat))
oof_cb = np.zeros(len(train_d49_feat))
test_lgb = np.zeros(len(test_feat))
test_xgb = np.zeros(len(test_feat))
test_cb = np.zeros(len(test_feat))

# Careful hyperparameter tuning for better generalization
lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1
}

xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbosity': 0
}

cb_params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 6,
    'subsample': 0.75,
    'colsample_bylevel': 0.75,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'verbose': 0
}

# =====================================================================
# STEP 7: Training
# =====================================================================
print("\nStep 7: Training models...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_d49_feat)):
    print(f"\n--- Fold {fold} ---")
    
    # Training: Day 48 full day + fold portion of Day 49 early morning
    fold_d49_train = train_d49_feat.iloc[train_idx]
    fold_train = pd.concat([train_d48_feat, fold_d49_train], ignore_index=True)
    
    X_train = fold_train[features]
    y_train = fold_train[target]
    
    # Validation: held-out Day 49 early morning
    fold_val = train_d49_feat.iloc[val_idx]
    X_val = fold_val[features]
    y_val = fold_val[target]
    
    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
    )
    oof_lgb[val_idx] = np.clip(model_lgb.predict(X_val), 0.0, 1.0)
    test_lgb += np.clip(model_lgb.predict(test_feat[features]), 0.0, 1.0) / 5.0
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = np.clip(model_xgb.predict(X_val), 0.0, 1.0)
    test_xgb += np.clip(model_xgb.predict(test_feat[features]), 0.0, 1.0) / 5.0
    
    # CatBoost
    model_cb = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=100)
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    oof_cb[val_idx] = np.clip(model_cb.predict(X_val), 0.0, 1.0)
    test_cb += np.clip(model_cb.predict(test_feat[features]), 0.0, 1.0) / 5.0
    
    y_true = fold_val[target].values
    print(f"  LGBM: {r2_score(y_true, oof_lgb[val_idx])*100:.4f}%  "
          f"XGB: {r2_score(y_true, oof_xgb[val_idx])*100:.4f}%  "
          f"CB: {r2_score(y_true, oof_cb[val_idx])*100:.4f}%")

# =====================================================================
# STEP 8: OOF Evaluation
# =====================================================================
print("\n\nStep 8: OOF Metrics...")
y_d49 = train_d49_feat[target].values

score_lgb = r2_score(y_d49, oof_lgb) * 100
score_xgb = r2_score(y_d49, oof_xgb) * 100
score_cb = r2_score(y_d49, oof_cb) * 100
oof_ensemble = (oof_lgb + oof_xgb + oof_cb) / 3.0
score_ensemble = r2_score(y_d49, oof_ensemble) * 100

print("=" * 50)
print(f"OOF R2 LightGBM:  {score_lgb:.4f}%")
print(f"OOF R2 XGBoost:   {score_xgb:.4f}%")
print(f"OOF R2 CatBoost:  {score_cb:.4f}%")
print(f"OOF R2 Ensemble:  {score_ensemble:.4f}%")
print("=" * 50)

# Also evaluate model on Day 48 daytime as sanity check  
# (this should be very high since demand48_t ≈ target)
# Use pure scaling as reference
d49_feat_ref = train_d49_feat.copy()
scaled_oof = np.clip(d49_feat_ref['scaled_demand48'].values, 0.0, 1.0)
score_scaled = r2_score(y_d49, scaled_oof) * 100
print(f"Pure scaled_demand48 baseline: {score_scaled:.4f}%")

# =====================================================================
# STEP 9: Optimal Blending with Scaled Baseline
# =====================================================================
print("\nStep 9: Optimal blending...")
# Find optimal weight for blending GBDT ensemble with pure scaling baseline
best_alpha = 1.0
best_score = score_ensemble

for alpha in np.linspace(0.7, 1.0, 31):
    blend = alpha * oof_ensemble + (1 - alpha) * scaled_oof
    s = r2_score(y_d49, np.clip(blend, 0.0, 1.0)) * 100
    if s > best_score:
        best_score = s
        best_alpha = alpha

print(f"Best alpha (GBDT weight): {best_alpha:.3f}, score: {best_score:.4f}%")

# Apply best blending to test
scaled_test_baseline = np.clip(test_feat['scaled_demand48'].values, 0.0, 1.0)
final_ensemble = np.clip((test_lgb + test_xgb + test_cb) / 3.0, 0.0, 1.0)
final_preds = best_alpha * final_ensemble + (1 - best_alpha) * scaled_test_baseline
final_preds = np.clip(final_preds, 0.0, 1.0)

print(f"\nFinal predictions statistics:")
print(f"  Mean: {final_preds.mean():.4f}")
print(f"  Std: {final_preds.std():.4f}")
print(f"  Min: {final_preds.min():.4f}")
print(f"  Max: {final_preds.max():.4f}")

# =====================================================================
# STEP 10: Generate Submission File
# =====================================================================
print("\nStep 10: Saving submission...")

submission = pd.DataFrame({
    'Index': test_feat['Index'].astype(int),
    'demand': final_preds
})

# Validate
test_orig = pd.read_csv('test.csv')
assert len(submission) == 41778, f"Wrong shape: {len(submission)}"
assert list(submission.columns) == ['Index', 'demand'], f"Wrong columns: {list(submission.columns)}"
assert not submission['demand'].isna().any(), "NaN values in submission!"
assert (submission['demand'] >= 0.0).all() and (submission['demand'] <= 1.0).all(), "Out of bounds predictions!"
assert (submission['Index'].values == test_orig['Index'].values).all(), "Index mismatch!"

submission.to_csv('submission_v2.csv', index=False)
print("Saved to submission_v2.csv")
print(f"Shape: {submission.shape}")
print("All validation checks passed!")
