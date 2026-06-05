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

train = pd.read_csv('train.csv')

def get_minutes(t):
    parts = t.split(':')
    return int(parts[0]) * 60 + int(parts[1])

train['time_minutes'] = train['timestamp'].apply(get_minutes)

# Decode geohashes
from src.utils import decode_geohash, impute_missing_values, parse_time_features
train = parse_time_features(train)
lats_lons = train['geohash'].apply(decode_geohash)
train['latitude'] = [x[0] for x in lats_lons]
train['longitude'] = [x[1] for x in lats_lons]
train['geohash_prefix4'] = train['geohash'].str[:4]
train['geohash_prefix5'] = train['geohash'].str[:5]
train = impute_missing_values(train)

# Split into Day 48 and Day 49
d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()

# Interpolate Day 48 grid
geohashes = d48['geohash'].unique()
times = sorted(d48['time_minutes'].unique())
grid48 = pd.MultiIndex.from_product([geohashes, times], names=['geohash', 'time_minutes']).to_frame().reset_index(drop=True)
grid48 = grid48.merge(d48[['geohash', 'time_minutes', 'demand']], on=['geohash', 'time_minutes'], how='left')
grid48 = grid48.sort_values(['geohash', 'time_minutes'])
grid48['demand48'] = grid48.groupby('geohash')['demand'].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both').fillna(0.0)
)

# Merge Day 48 demand into Day 49
d49 = d49.merge(grid48[['geohash', 'time_minutes', 'demand48']], on=['geohash', 'time_minutes'], how='left')
d49['demand48'] = d49['demand48'].fillna(0.0)

# Compute early morning averages
d48_early = d48[d48['time_minutes'] <= 120].groupby('geohash')['demand'].mean().reset_index().rename(columns={'demand': 'early_48'})
d49 = d49.merge(d48_early, on='geohash', how='left')
d49['early_48'] = d49['early_48'].fillna(d49['early_48'].mean())

# Non-leaky Day 49 early morning average
d49_early_rows = d49[d49['time_minutes'] <= 120].copy()
early_stats = d49_early_rows.groupby('geohash')['demand'].agg(['sum', 'count']).reset_index()
d49 = d49.merge(early_stats, on='geohash', how='left')
d49['sum'] = d49['sum'].fillna(0.0)
d49['count'] = d49['count'].fillna(0)

is_d49_early = (d49['time_minutes'] <= 120)
d49['early_49'] = np.where(
    is_d49_early,
    (d49['sum'] - d49['demand']) / (d49['count'] - 1).clip(lower=1),
    d49['sum'] / d49['count'].clip(lower=1)
)
d49['early_49'] = np.where(d49['count'] == 0, np.nan, d49['early_49'])
d49['early_49'] = d49['early_49'].fillna(d49['early_49'].mean())

# Scaling ratio and scaled baseline feature
d49['geohash_ratio'] = (d49['early_49'] + 0.01) / (d49['early_48'] + 0.01)
d49['scaled_demand48'] = d49['demand48'] * d49['geohash_ratio']

# Let's add neighborhood profiles on Day 48
grid48['geohash_prefix5'] = grid48['geohash'].str[:5]
grid48['geohash_prefix4'] = grid48['geohash'].str[:4]
p5_stats = grid48.groupby(['geohash_prefix5', 'time_minutes'])['demand48'].mean().reset_index().rename(columns={'demand48': 'p5_demand48_t'})
p4_stats = grid48.groupby(['geohash_prefix4', 'time_minutes'])['demand48'].mean().reset_index().rename(columns={'demand48': 'p4_demand48_t'})
d49 = d49.merge(p5_stats, on=['geohash_prefix5', 'time_minutes'], how='left')
d49 = d49.merge(p4_stats, on=['geohash_prefix4', 'time_minutes'], how='left')
d49['p5_demand48_t'] = d49['p5_demand48_t'].fillna(0.0)
d49['p4_demand48_t'] = d49['p4_demand48_t'].fillna(0.0)

# Encode categoricals
cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash', 'geohash_prefix4', 'geohash_prefix5']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
d49[cat_cols] = encoder.fit_transform(d49[cat_cols].astype(str))

features = [
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time', 'latitude', 'longitude',
    'geohash', 'geohash_prefix4', 'geohash_prefix5',
    'early_48', 'early_49', 'geohash_ratio', 'scaled_demand48',
    'p5_demand48_t', 'p4_demand48_t'
]

# Target is: multiplier = (demand + 0.01) / (scaled_demand48 + 0.01)
# Prediction = np.clip(scaled_demand48 * pred_multiplier - 0.01, 0.0, 1.0)
d49['target_mult'] = (d49['demand'] + 0.01) / (d49['scaled_demand48'] + 0.01)

# K-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_mult = np.zeros(len(d49))

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(d49)):
    X_train, y_train_mult = d49.iloc[train_idx][features], d49.iloc[train_idx]['target_mult']
    X_val, y_val_mult = d49.iloc[val_idx][features], d49.iloc[val_idx]['target_mult']
    y_val = d49.iloc[val_idx]['demand']
    
    model = lgb.LGBMRegressor(**lgb_params)
    # Train to predict target multiplier
    model.fit(X_train, y_train_mult, eval_set=[(X_val, y_val_mult)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    pred_mult = model.predict(X_val)
    # Reconstruct prediction
    pred_demand = X_val['scaled_demand48'] * pred_mult
    oof_mult[val_idx] = np.clip(pred_demand, 0.0, 1.0)

print(f"Multiplicative Correction GBDT OOF R2 Score: {r2_score(d49['demand'], oof_mult)*100:.4f}%")
