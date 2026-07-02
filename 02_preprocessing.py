import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

months = ['202502', '202503', '202504', '202505', '202506', '202507',
'202508', '202509', '202510', '202511', '202512', '202601', '202602',
'202603', '202604', '202605']

# -------------------------------------------------------------------
# STEP 1: Define target + features
# -------------------------------------------------------------------
target = 'ClosePrice'

# NOTE: confirm these column names match your actual CSV headers
# (run: pandas.read_csv('CRMLSSold202511.csv').columns.tolist())
numeric_features = [
    'BedroomsTotal', 'BathroomsTotalInteger', 'LivingArea', 'LotSizeSquareFeet',
    'YearBuilt', 'GarageSpaces', 'ViewYN', 'WaterfrontYN', 'BasementYN',
    'PoolPrivateYN', 'AttachedGarageYN', 'FireplaceYN', 'NewConstructionYN',
]

# Location field, kept as a raw string/code through preprocessing and
# one-hot encoded later (in Step 3) since it's the strongest missing
# price signal -- two identical homes can differ 2-3x in price purely
# based on neighborhood.
zip_col = 'PostalCode'

# Columns required to be non-missing (used for dropna). zip_col included
# here so rows with no zip code are dropped rather than silently kept.
features = numeric_features + [zip_col]

# -------------------------------------------------------------------
# STEP 2: Clean + preprocess each month's raw file
# -------------------------------------------------------------------
categorical_fields = ['ViewYN', 'WaterfrontYN', 'BasementYN',
                       'PoolPrivateYN', 'AttachedGarageYN', 'FireplaceYN',
                       'NewConstructionYN']

# Columns to check for outliers using the IQR method. ClosePrice is the
# most important one (kills the RMSE/MAPE blowup from luxury estates,
# land transfers, etc.) but LivingArea and LotSizeSquareFeet are also
# common sources of data-entry errors (e.g. sqft off by 10x).
outlier_check_cols = ['ClosePrice', 'LivingArea', 'LotSizeSquareFeet']


def remove_iqr_outliers(df, cols, k=1.5):
    """Drop rows outside [Q1 - k*IQR, Q3 + k*IQR] for each column in cols."""
    mask = pandas.Series(True, index=df.index)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]


for month in months:
    file_path = f'CRMLSSold{month}.csv'
    df = pandas.read_csv(file_path, low_memory=False)
    df = df[(df['PropertyType'] == 'Residential') &
            (df['PropertySubType'] == 'SingleFamilyResidence')]

    # These fields are booleans, but MLS often leaves them blank instead of
    # writing False for e.g. "not waterfront". Fill missing as False.
    for field in categorical_fields:
        df[field] = df[field].fillna(False).astype(int)

    # Only require non-missing values in the columns we'll actually use.
    # Most MLS columns (school district names, tax year, builder, etc.)
    # are mostly empty and irrelevant here -- dropping on the WHOLE
    # dataframe wipes out every row. Restrict dropna to features + target.
    df = df.dropna(subset=features + [target])

    # Normalize zip code to a clean string (avoids 92109 vs 92109.0 vs
    # '92109' being treated as different categories later)
    df[zip_col] = df[zip_col].astype(str).str.split('.').str[0].str.strip()

    # Remove IQR outliers (luxury estates, land transfers, sqft typos, etc.)
    before = len(df)
    df = remove_iqr_outliers(df, outlier_check_cols)
    print(f'  {month}: removed {before - len(df)} outlier rows ({before} -> {len(df)})')

    df.to_csv(f'Processed_CRMLSSold{month}.csv', index=False)
    print(f'Processed {month}: {len(df)} rows')

# -------------------------------------------------------------------
# STEP 3: Rolling window train/test comparison
# -------------------------------------------------------------------
results = []

for X in [1, 2, 3, 6, 9, 12]:
    test_month = months[-1]
    train_months = months[-(X + 1):-1]  # exclude the test month

    if len(train_months) < X:
        print(f'Skipping X={X}: not enough historical months available')
        continue

    print(f'\nTraining on months: {train_months}, Testing on month: {test_month}')

    train_dfs = [pandas.read_csv(f'Processed_CRMLSSold{m}.csv') for m in train_months]
    train_df = pandas.concat(train_dfs, ignore_index=True)
    test_df = pandas.read_csv(f'Processed_CRMLSSold{test_month}.csv')

    # One-hot encode zip code. Train and test must be encoded TOGETHER,
    # otherwise a zip code that appears in test but not train (or vice
    # versa) creates mismatched columns between X_train and X_test.
    train_df[zip_col] = train_df[zip_col].astype(str)
    test_df[zip_col] = test_df[zip_col].astype(str)
    combined = pandas.concat([train_df, test_df], keys=['train', 'test'])
    combined = pandas.get_dummies(combined, columns=[zip_col], prefix='zip')
    train_df = combined.xs('train')
    test_df = combined.xs('test')

    zip_dummy_cols = [c for c in combined.columns if c.startswith('zip_')]
    model_features = numeric_features + zip_dummy_cols

    X_train, y_train = train_df[model_features], train_df[target]
    X_test, y_test = test_df[model_features], test_df[target]

    model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    results.append({
        'window_months': X,
        'train_size': len(train_df),
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
    })

    print(f'  train_size={len(train_df)}  RMSE=${rmse:,.0f}  MAE=${mae:,.0f}  MAPE={mape:.2f}%')

results_df = pandas.DataFrame(results)
print('\n=== Summary across window lengths ===')
print(results_df)

# -------------------------------------------------------------------
# STEP 4: Plot RMSE and MAE vs. training window length
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(results_df['window_months'], results_df['rmse'], marker='o')
axes[0].set_title('RMSE vs Training Window Length')
axes[0].set_xlabel('Months of training data')
axes[0].set_ylabel('RMSE ($)')

axes[1].plot(results_df['window_months'], results_df['mae'], marker='o', color='orange')
axes[1].set_title('MAE vs Training Window Length')
axes[1].set_xlabel('Months of training data')
axes[1].set_ylabel('MAE ($)')

plt.tight_layout()
plt.savefig('window_comparison.png', dpi=150)
#plt.show()

# -------------------------------------------------------------------
# STEP 5 (optional but useful): Feature importance from the best model
# -------------------------------------------------------------------
best_X = results_df.loc[results_df['rmse'].idxmin(), 'window_months']
print(f'\nBest window length by RMSE: X={best_X} months')

importances = pandas.Series(model.feature_importances_, index=model_features).sort_values(ascending=False)
print('\nTop 20 feature importances (from last model trained):')
print(importances.head(20))