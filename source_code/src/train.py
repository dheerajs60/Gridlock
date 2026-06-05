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
    all_feat = build_features(all_df, train)
    
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
        'geohash', 'geohash_prefix4', 'geohash_prefix5',
        'demand_at_2_00', 'demand_at_1_45', 'demand_at_1_30', 'demand_at_1_15', 'demand_at_1_00', 'mean_demand_0_to_2',
        'mean_demand_0_to_2_prefix5', 'mean_demand_0_to_2_prefix4',
        'demand_day48_t', 'demand_day48_t_minus_15', 'demand_day48_t_minus_30', 'demand_day48_t_minus_60',
        'demand_day48_t_plus_15', 'demand_day48_t_plus_30', 'demand_day48_t_plus_60',
        'demand_day48_rolling_mean_30', 'demand_day48_rolling_mean_60',
        'day48_overall_mean', 'day48_overall_std', 'day48_overall_max', 'day48_overall_min',
        'day48_prefix5_mean', 'day48_prefix5_std', 'day48_prefix5_max', 'day48_prefix5_min',
        'day48_prefix4_mean', 'day48_prefix4_std', 'day48_prefix4_max', 'day48_prefix4_min',
        'demand_day48_prefix5_t', 'demand_day48_prefix4_t', 'city_profile_day48'
    ]
    
    target = 'demand'
    
    # Split into Day 48 and Day 49
    train_48 = train_feat[train_feat['day'] == 48].copy()
    train_49 = train_feat[train_feat['day'] == 49].copy()
    
    print(f"Train Day 48 shape: {train_48.shape}")
    print(f"Train Day 49 shape: {train_49.shape}")
    print(f"Test Day 49 shape: {test_feat.shape}")
    
    # K-Fold Cross Validation Setup on Day 49
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prediction arrays
    oof_lgb = np.zeros(len(train_49))
    oof_xgb = np.zeros(len(train_49))
    oof_cb = np.zeros(len(train_49))
    
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
    
    print("\nStep 3: Starting Cross-Validation & Training...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
        print(f"\n--- Training Fold {fold} ---")
        
        # Training fold: Day 48 + part of Day 49
        fold_train_49 = train_49.iloc[train_idx]
        fold_train = pd.concat([train_48, fold_train_49], ignore_index=True)
        
        X_train, y_train = fold_train[features], fold_train[target]
        
        # Validation fold: remaining part of Day 49
        fold_val = train_49.iloc[val_idx]
        X_val, y_val = fold_val[features], fold_val[target]
        
        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_lgb[val_idx] = np.clip(model_lgb.predict(X_val), 0.0, 1.0)
        test_lgb += np.clip(model_lgb.predict(test_feat[features]), 0.0, 1.0) / 5.0
        
        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        oof_xgb[val_idx] = np.clip(model_xgb.predict(X_val), 0.0, 1.0)
        test_xgb += np.clip(model_xgb.predict(test_feat[features]), 0.0, 1.0) / 5.0
        
        # 3. CatBoost
        model_cb = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        model_cb.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        oof_cb[val_idx] = np.clip(model_cb.predict(X_val), 0.0, 1.0)
        test_cb += np.clip(model_cb.predict(test_feat[features]), 0.0, 1.0) / 5.0
        
        # Calculate fold metrics
        r2_lgb = r2_score(y_val, oof_lgb[val_idx]) * 100
        r2_xgb = r2_score(y_val, oof_xgb[val_idx]) * 100
        r2_cb = r2_score(y_val, oof_cb[val_idx]) * 100
        print(f"Fold {fold} R2 Scores -> LGBM: {r2_lgb:.4f} | XGB: {r2_xgb:.4f} | CatBoost: {r2_cb:.4f}")
        
    print("\nStep 4: Evaluating Out-of-Fold (OOF) Metrics...")
    y_49 = train_49[target].values
    
    score_lgb = r2_score(y_49, oof_lgb) * 100
    score_xgb = r2_score(y_49, oof_xgb) * 100
    score_cb = r2_score(y_49, oof_cb) * 100
    
    # Blended/Ensemble predictions
    oof_ensemble = (oof_lgb + oof_xgb + oof_cb) / 3.0
    score_ensemble = r2_score(y_49, oof_ensemble) * 100
    
    print("="*50)
    print(f"OOF R2 Score LightGBM: {score_lgb:.4f}")
    print(f"OOF R2 Score XGBoost:  {score_xgb:.4f}")
    print(f"OOF R2 Score CatBoost: {score_cb:.4f}")
    print(f"OOF R2 Score Ensemble: {score_ensemble:.4f}")
    print("="*50)
    
    print("\nStep 5: Generating Submission File...")
    final_preds = (test_lgb + test_xgb + test_cb) / 3.0
    
    # Construct submission dataframe
    submission = pd.DataFrame({
        'Index': test_feat['Index'].astype(int),
        'demand': np.clip(final_preds, 0.0, 1.0)
    })
    
    # Validate structure
    assert len(submission) == 41778, f"Incorrect number of rows: {len(submission)}"
    assert list(submission.columns) == ['Index', 'demand'], f"Incorrect columns: {list(submission.columns)}"
    assert not submission['demand'].isna().any(), "Submission contains NaN values"
    assert (submission['demand'] >= 0.0).all() and (submission['demand'] <= 1.0).all(), "Predictions out of bounds"
    
    # Verify index values match test file exactly
    test_orig = pd.read_csv('test.csv')
    assert (submission['Index'].values == test_orig['Index'].values).all(), "Index values do not match test file"
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file successfully saved to submission.csv!")
    print("Validation passed: File shape is 41778 x 2, and all values are in bounds [0, 1].")

if __name__ == "__main__":
    train_and_evaluate()
