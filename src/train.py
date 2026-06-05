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
    
    # Fit encoder on all train and transform both
    train_feat[cat_cols] = encoder.fit_transform(train_feat[cat_cols].astype(str))
    test_feat[cat_cols] = encoder.transform(test_feat[cat_cols].astype(str))
    
    features = [
        'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
        'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time', 'latitude', 'longitude',
        'geohash', 'geohash_prefix4', 'geohash_prefix5',
        'early_48', 'early_49', 'geohash_ratio', 'scaled_demand48',
        'p5_demand48_t', 'p4_demand48_t'
    ]
    
    target = 'demand'
    
    # We train GBDTs on Day 49 early morning (where residuals vary realistically)
    train_49 = train_feat[train_feat['day'] == 49].copy()
    
    print(f"Train Day 49 early morning shape: {train_49.shape}")
    print(f"Test Day 49 shape: {test_feat.shape}")
    
    # K-Fold Cross Validation Setup on Day 49
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # OOF prediction arrays
    oof_lgb = np.zeros(len(train_49))
    oof_xgb = np.zeros(len(train_49))
    oof_cb = np.zeros(len(train_49))
    
    # Test prediction arrays
    test_res_lgb = np.zeros(len(test_feat))
    test_res_xgb = np.zeros(len(test_feat))
    test_res_cb = np.zeros(len(test_feat))
    
    lgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 5,
        'verbosity': 0,
        'random_state': 42
    }
    
    cb_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 5,
        'verbose': 0,
        'random_seed': 42
    }
    
    print("\nStep 3: Starting Cross-Validation & Training on Residual GBDT Ensemble...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
        print(f"\n--- Training Fold {fold} ---")
        
        # Split features and target
        X_train, y_train = train_49.iloc[train_idx][features], train_49.iloc[train_idx][target]
        X_val, y_val = train_49.iloc[val_idx][features], train_49.iloc[val_idx][target]
        
        # Calculate residuals for training and validation
        y_train_res = y_train - X_train['scaled_demand48']
        y_val_res = y_val - X_val['scaled_demand48']
        
        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_train, y_train_res, eval_set=[(X_val, y_val_res)], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred_res_val = model_lgb.predict(X_val)
        oof_lgb[val_idx] = np.clip(X_val['scaled_demand48'] + pred_res_val, 0.0, 1.0)
        test_res_lgb += model_lgb.predict(test_feat[features]) / 5.0
        
        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_train, y_train_res, eval_set=[(X_val, y_val_res)], verbose=False)
        pred_res_val = model_xgb.predict(X_val)
        oof_xgb[val_idx] = np.clip(X_val['scaled_demand48'] + pred_res_val, 0.0, 1.0)
        test_res_xgb += model_xgb.predict(test_feat[features]) / 5.0
        
        # 3. CatBoost
        model_cb = cb.CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        model_cb.fit(X_train, y_train_res, eval_set=(X_val, y_val_res), verbose=False)
        pred_res_val = model_cb.predict(X_val)
        oof_cb[val_idx] = np.clip(X_val['scaled_demand48'] + pred_res_val, 0.0, 1.0)
        test_res_cb += model_cb.predict(test_feat[features]) / 5.0
        
        # Calculate fold metrics
        r2_lgb = r2_score(y_val, oof_lgb[val_idx]) * 100
        r2_xgb = r2_score(y_val, oof_xgb[val_idx]) * 100
        r2_cb = r2_score(y_val, oof_cb[val_idx]) * 100
        print(f"Fold {fold} R2 Scores -> LGBM: {r2_lgb:.4f}% | XGB: {r2_xgb:.4f}% | CatBoost: {r2_cb:.4f}%")
        
    print("\nStep 4: Evaluating Out-of-Fold (OOF) Metrics...")
    y_49 = train_49[target].values
    
    score_lgb = r2_score(y_49, oof_lgb) * 100
    score_xgb = r2_score(y_49, oof_xgb) * 100
    score_cb = r2_score(y_49, oof_cb) * 100
    
    # Blended/Ensemble predictions
    oof_ensemble = (oof_lgb + oof_xgb + oof_cb) / 3.0
    score_ensemble = r2_score(y_49, oof_ensemble) * 100
    
    print("="*50)
    print(f"OOF R2 Score LightGBM: {score_lgb:.4f}%")
    print(f"OOF R2 Score XGBoost:  {score_xgb:.4f}%")
    print(f"OOF R2 Score CatBoost: {score_cb:.4f}%")
    print(f"OOF R2 Score Ensemble: {score_ensemble:.4f}%")
    print("="*50)
    
    print("\nStep 5: Generating Submission File...")
    blended_test_res = (test_res_lgb + test_res_xgb + test_res_cb) / 3.0
    final_preds = test_feat['scaled_demand48'] + blended_test_res
    final_preds = np.clip(final_preds, 0.0, 1.0)
    
    # Construct submission dataframe
    submission = pd.DataFrame({
        'Index': test_feat['Index'].astype(int),
        'demand': final_preds
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
