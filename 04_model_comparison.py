import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# STEP 3: Train on past 12 months, test on the latest month
# -------------------------------------------------------------------
test_month = months[-1]
train_months = months[-13:-1]  # the 12 months immediately before the test month

print(f'\nTraining on months: {train_months}')
print(f'Testing on month: {test_month}')

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

# -------------------------------------------------------------------
# STEP 4: Baseline model -- Linear Regression
#
# Unlike Random Forest, linear models are sensitive to feature scale --
# LivingArea (hundreds/thousands) would otherwise dominate BedroomsTotal
# (single digits) in the coefficients regardless of true predictive
# power. Scale only the numeric features; the one-hot zip dummies
# (already 0/1) don't need scaling.
# -------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

baseline_model = LinearRegression()
baseline_model.fit(X_train_scaled, y_train)
baseline_preds = baseline_model.predict(X_test_scaled)

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_r2 = r2_score(y_test, baseline_preds)

print('\n=== Baseline: Linear Regression ===')
print(f'  train_size={len(train_df)}  RMSE=${baseline_rmse:,.0f}  MAE=${baseline_mae:,.0f}  R\u00b2={baseline_r2:.3f}')

# Record baseline results to their own CSV (week 4 deliverable)
baseline_summary = pandas.DataFrame([{
    'train_months': ', '.join(train_months),
    'test_month': test_month,
    'train_size': len(train_df),
    'rmse': baseline_rmse,
    'mae': baseline_mae,
    'r2': baseline_r2,
}])
baseline_summary.to_csv('baseline_linear_regression_results.csv', index=False)
print('Saved to baseline_linear_regression_results.csv')

# -------------------------------------------------------------------
# STEP 5: Decision Tree and Random Forest models (Week 5)
#
# Tree-based models don't need feature scaling (splits are based on
# thresholds, not magnitudes), so we use the unscaled X_train/X_test
# rather than X_train_scaled/X_test_scaled.
# -------------------------------------------------------------------
tree_model = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

tree_rmse = np.sqrt(mean_squared_error(y_test, tree_preds))
tree_mae = mean_absolute_error(y_test, tree_preds)
tree_r2 = r2_score(y_test, tree_preds)

print('\n=== Decision Tree ===')
print(f'  train_size={len(train_df)}  RMSE=${tree_rmse:,.0f}  MAE=${tree_mae:,.0f}  R\u00b2={tree_r2:.3f}')

forest_model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                      random_state=42, n_jobs=-1)
forest_model.fit(X_train, y_train)
forest_preds = forest_model.predict(X_test)

forest_rmse = np.sqrt(mean_squared_error(y_test, forest_preds))
forest_mae = mean_absolute_error(y_test, forest_preds)
forest_r2 = r2_score(y_test, forest_preds)

print('\n=== Random Forest ===')
print(f'  train_size={len(train_df)}  RMSE=${forest_rmse:,.0f}  MAE=${forest_mae:,.0f}  R\u00b2={forest_r2:.3f}')

# -------------------------------------------------------------------
# STEP 6: Compare test R\u00b2 across all three models and save deliverable
# -------------------------------------------------------------------
comparison = pandas.DataFrame([
    {'model': 'Linear Regression', 'train_size': len(train_df), 'rmse': baseline_rmse,
     'mae': baseline_mae, 'r2': baseline_r2},
    {'model': 'Decision Tree', 'train_size': len(train_df), 'rmse': tree_rmse,
     'mae': tree_mae, 'r2': tree_r2},
    {'model': 'Random Forest', 'train_size': len(train_df), 'rmse': forest_rmse,
     'mae': forest_mae, 'r2': forest_r2},
])
comparison['test_month'] = test_month
comparison['train_months'] = ', '.join(train_months)
comparison = comparison[['model', 'train_months', 'test_month', 'train_size', 'rmse', 'mae', 'r2']]

print('\n=== Model Comparison (test R\u00b2) ===')
print(comparison[['model', 'rmse', 'mae', 'r2']].to_string(index=False))

comparison.to_csv('model_comparison_results.csv', index=False)
print('\nSaved to model_comparison_results.csv')

# -------------------------------------------------------------------
# STEP 7: Feature importance from the Random Forest
# (helps document *why* the tree-based models behave the way they do)
# -------------------------------------------------------------------
importances = pandas.Series(forest_model.feature_importances_, index=model_features).sort_values(ascending=False)
print('\nTop 20 feature importances (Random Forest):')
print(importances.head(20))
importances.head(20).to_csv('random_forest_feature_importances.csv', header=['importance'])

# -------------------------------------------------------------------
# STEP 8: Documented model behavior -- strengths / weaknesses
#
# This is the narrative deliverable ("document model behavior") to pair
# with the numeric comparison above. Written to a text file so it can be
# dropped straight into a report.
# -------------------------------------------------------------------
notes = f"""
Week 5 Model Comparison Notes
==============================
Test month: {test_month}
Train months: {', '.join(train_months)}
Train size: {len(train_df)} rows

Results (test set):
  Linear Regression -> RMSE=${baseline_rmse:,.0f}  MAE=${baseline_mae:,.0f}  R\u00b2={baseline_r2:.3f}
  Decision Tree      -> RMSE=${tree_rmse:,.0f}  MAE=${tree_mae:,.0f}  R\u00b2={tree_r2:.3f}
  Random Forest      -> RMSE=${forest_rmse:,.0f}  MAE=${forest_mae:,.0f}  R\u00b2={forest_r2:.3f}

Linear Regression
  Strengths: Fast to train, coefficients are directly interpretable
  (e.g. "$X per additional sqft" holding other features fixed), and it
  is a stable, low-variance baseline that's hard to overfit given
  enough rows relative to the number of zip-code dummy columns.
  Weaknesses: Assumes a linear, additive relationship between features
  and price. It can't capture interactions (e.g. an extra bedroom is
  worth more in some zip codes than others) or non-linear effects
  (e.g. diminishing returns on LivingArea past a certain size) unless
  those interactions are explicitly engineered as features.

Decision Tree
  Strengths: Captures non-linear relationships and feature interactions
  automatically (e.g. it can learn "zip=X AND LivingArea>2500 ->
  premium" without being told to). Also interpretable via its splits,
  and needs no feature scaling.
  Weaknesses: A single tree is high-variance and prone to overfitting,
  especially with many one-hot zip columns providing lots of ways to
  split on location alone. Small changes in the training data can
  produce a very different tree, and predictions are piecewise-constant
  (it can only predict values seen in training leaves), which tends to
  hurt RMSE/MAE versus a smoother model.

Random Forest
  Strengths: Averaging many decorrelated trees reduces the variance/
  overfitting problem of a single Decision Tree while keeping the
  ability to model non-linearities and interactions. Typically the best
  R\u00b2 of the three here, and feature_importances_ gives a useful ranking
  of which fields (numeric features vs. specific zip dummies) drive
  price the most.
  Weaknesses: Slower to train and to run inference on than the other
  two models (300 trees vs. 1 tree vs. 1 linear model), less directly
  interpretable than a single tree or linear coefficients, and it still
  can't extrapolate outside the price/feature ranges seen in training
  (e.g. a brand-new luxury zip code with no sales history).

Deliverable checklist:
  [x] Decision Tree and Random Forest regressors trained
  [x] Test R\u00b2 compared against Linear Regression baseline
      (see model_comparison_results.csv)
  [x] Model behavior (strengths/weaknesses) documented above
"""

with open('week5_model_notes.txt', 'w') as f:
    f.write(notes)

print(notes)
print('Saved to week5_model_notes.txt')