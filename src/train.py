import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from src.features import build_features
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Step 1: Loading Datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print("Step 2: Preprocessing and Feature Engineering...")
    # Concatenate train and test to ensure consistent feature engineering
    all_df = pd.concat([train, test], ignore_index=True)
    all_feat = build_features(all_df)
    
    # Split back to train and test
    train_feat = all_feat[all_feat['demand'].notna()].copy()
    test_feat = all_feat[all_feat['demand'].isna()].copy()
    
    # Categorical encoding
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash', 'geohash_prefix4', 'geohash_prefix5']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit encoder on train and transform both
    train_feat[cat_cols] = encoder.fit_transform(train_feat[cat_cols].astype(str))
    test_feat[cat_cols] = encoder.transform(test_feat[cat_cols].astype(str))
    
    features = [
        'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
        'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time', 'latitude', 'longitude',
        'geohash_prefix4', 'geohash_prefix5'
    ]
    
    target = 'demand'
    
    # Split into Day 48 and Day 49
    train_48 = train_feat[train_feat['day'] == 48].copy()
    train_49 = train_feat[train_feat['day'] == 49].copy()
    
    print(f"Train Day 48 shape: {train_48.shape}")
    print(f"Train Day 49 shape: {train_49.shape}")
    print(f"Test Day 49 shape: {test_feat.shape}")
    
    # --- SPATIOTEMPORAL KEY-LOOKUP SETUP ---
    # Create the lookup dictionary mapping (geohash (decoded), time_minutes) to demand from Day 48
    # Wait, geohash is ordinal encoded! Let's use the original 'geohash' column from 'train' for the lookup.
    
    train_48_original = train[train['day'] == 48].copy()
    train_48_original['time_minutes'] = train_48_original['timestamp'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    lookup_dict = train_48_original.set_index(['geohash', 'time_minutes'])['demand'].to_dict()
    
    test_original = test.copy()
    test_original['time_minutes'] = test_original['timestamp'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    
    # --- PURE TABULAR REGRESSION FALLBACK SETUP ---
    # K-Fold Cross Validation Setup on Day 48
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test prediction arrays
    test_lgb = np.zeros(len(test_feat))
    test_xgb = np.zeros(len(test_feat))
    test_cb = np.zeros(len(test_feat))
    
    lgb_params = {
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    
    xgb_params = {
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    cb_params = {
        'iterations': 1500,
        'learning_rate': 0.03,
        'depth': 6,
        'random_seed': 42,
        'verbose': 0
    }
    
    print("\nStep 3: Starting Cross-Validation & Training on Day 48 (Fallback Model)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_48)):
        print(f"\n--- Training Fold {fold} ---")
        
        X_train, y_train = train_48.iloc[train_idx][features], train_48.iloc[train_idx][target]
        X_val, y_val = train_48.iloc[val_idx][features], train_48.iloc[val_idx][target]
        
        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        test_lgb += np.clip(model_lgb.predict(test_feat[features]), 0.0, 1.0) / 5.0
        
        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        test_xgb += np.clip(model_xgb.predict(test_feat[features]), 0.0, 1.0) / 5.0
        
        # 3. CatBoost
        model_cb = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        test_cb += np.clip(model_cb.predict(test_feat[features]), 0.0, 1.0) / 5.0
        
    print("\nStep 4: Generating Submission File using Lookup Strategy + Fallback...")
    fallback_preds = (test_lgb + test_xgb + test_cb) / 3.0
    
    final_predictions = []
    lookup_count = 0
    fallback_count = 0
    
    for idx, row in test_original.iterrows():
        key = (row['geohash'], row['time_minutes'])
        if key in lookup_dict:
            final_predictions.append(lookup_dict[key])
            lookup_count += 1
        else:
            final_predictions.append(fallback_preds[idx])
            fallback_count += 1
            
    print(f"Predictions from exact historical matches: {lookup_count}")
    print(f"Predictions from regression fallback model: {fallback_count}")
    
    # Construct submission dataframe
    submission = pd.DataFrame({
        'Index': test_feat['Index'].astype(int),
        'demand': np.clip(final_predictions, 0.0, 1.0)
    })
    
    # Validate structure
    assert len(submission) == 41778, f"Incorrect number of rows: {len(submission)}"
    assert list(submission.columns) == ['Index', 'demand'], f"Incorrect columns: {list(submission.columns)}"
    assert not submission['demand'].isna().any(), "Submission contains NaN values"
    assert (submission['demand'] >= 0.0).all() and (submission['demand'] <= 1.0).all(), "Predictions out of bounds"
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file successfully saved to submission.csv!")

if __name__ == "__main__":
    train_and_evaluate()
