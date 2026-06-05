import pandas as pd
import numpy as np

sub = pd.read_csv('submission.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

print("Submission file info:")
print(sub.info())
print("\nFirst 5 rows:")
print(sub.head())
print("\nSummary statistics:")
print(sub.describe())

# Check shape
print(f"\nShape matches test: {len(sub) == len(test)} ({len(sub)} vs {len(test)})")
# Check columns
print(f"Columns match: {list(sub.columns) == ['Index', 'demand']}")
# Check index mapping
print(f"Index order matches test: {(sub['Index'].values == test['Index'].values).all()}")
# Check for NaNs
print(f"Any NaN values: {sub.isna().any().any()}")
# Check bounds
print(f"All predictions in [0, 1]: {(sub['demand'] >= 0.0).all() and (sub['demand'] <= 1.0).all()}")

# Distribution comparison
def get_minutes(t):
    parts = t.split(':')
    return int(parts[0]) * 60 + int(parts[1])

train['time_minutes'] = train['timestamp'].apply(get_minutes)
test['time_minutes'] = test['timestamp'].apply(get_minutes)
test['pred'] = sub['demand']

d48_daytime = train[(train['day'] == 48) & (train['time_minutes'] >= 135)]
test_daytime = test[test['time_minutes'] >= 135]

print("\nDay 48 daytime demand mean:", d48_daytime['demand'].mean())
print("Day 49 test daytime predicted mean:", test_daytime['pred'].mean())
print("Day 48 daytime demand max:", d48_daytime['demand'].max())
print("Day 49 test daytime predicted max:", test_daytime['pred'].max())

# Checking if there are any all-zero or constant predictions
print("\nNumber of unique predictions:", sub['demand'].nunique())
