import pandas
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
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
    'BedBathRatio', 'PropertyAge',
]

# Location field, kept as a raw string/code through preprocessing and
# one-hot encoded later (in Step 3) since it's the strongest missing
# price signal -- two identical homes can differ 2-3x in price purely
# based on neighborhood.
zip_col = 'PostalCode'

# Columns required to be non-missing (used for dropna). zip_col included
# here so rows with no zip code are dropped rather than silently kept.
# NOTE: BedBathRatio / PropertyAge are derived AFTER dropna below (from
# BedroomsTotal / BathroomsTotalInteger / YearBuilt, which already are
# required), so they're intentionally left out of this list.
features = [f for f in numeric_features if f not in ('BedBathRatio', 'PropertyAge')] + [zip_col]

# Latitude/Longitude column names -- update these if your CRMLS export
# uses different field names.
lat_col = 'Latitude'
lon_col = 'Longitude'

# -------------------------------------------------------------------
# STEP 1.5: Load + filter school district boundaries (once, outside loop)
# -------------------------------------------------------------------
districts_gdf = gpd.read_file('school_districts.geojson')  # update path as needed

# Only keep Unified school districts
districts_gdf = districts_gdf[districts_gdf['DistrictType'] == 'Unified']

# Spatial joins need matching CRS. GeoJSON is typically WGS84 (EPSG:4326),
# which is also what Latitude/Longitude columns from MLS data assume.
if districts_gdf.crs is None:
    districts_gdf = districts_gdf.set_crs('EPSG:4326')
elif districts_gdf.crs.to_epsg() != 4326:
    districts_gdf = districts_gdf.to_crs('EPSG:4326')


def add_district_name(df, lat_col=lat_col, lon_col=lon_col):
    """Spatial-join each row's lat/lon point against districts_gdf and
    attach the containing Unified district's name as `DistrictName`.
    Rows that don't fall inside any Unified district get NaN."""
    df = df.copy()
    has_coords = df[lat_col].notna() & df[lon_col].notna()

    geometry = [Point(xy) for xy in zip(df.loc[has_coords, lon_col],
                                          df.loc[has_coords, lat_col])]
    points_gdf = gpd.GeoDataFrame(
        df.loc[has_coords], geometry=geometry, crs='EPSG:4326'
    )

    # sjoin keeps points_gdf's index, so we can assign back cleanly.
    joined = gpd.sjoin(points_gdf, districts_gdf[['DistrictName', 'geometry']],
                        how='left', predicate='within')

    # Remove duplicates from points falling directly on boundary lines
    joined = joined[~joined.index.duplicated(keep='first')]

    # FIX: Initialize as object type (or directly map/assign) so pandas 
    # doesn't infer float64 from np.nan before seeing string data.
    df['DistrictName'] = None  # None creates an object-dtype column
    df.loc[joined.index, 'DistrictName'] = joined['DistrictName']
    
    return df


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

    # NEW: attach Unified school district name via spatial join
    df = add_district_name(df)

    # NEW: Bed/Bath ratio. BedroomsTotal is required non-missing (see
    # `features` above) but can legitimately be 0 for studio-style
    # single-family listings, so guard against divide-by-zero rather
    # than letting it produce inf.
    df['BedBathRatio'] = np.where(
        df['BedroomsTotal'] > 0,
        df['BathroomsTotalInteger'] / df['BedroomsTotal'],
        np.nan
    )

    # NEW: Property age at time of sale, in years. Using the sale
    # month's year (rather than today's date) keeps this fixed and
    # historically accurate -- a 2025 sale of a 2000-built home is a
    # 25-year-old property regardless of when this script is *run*.
    sale_year = int(month[:4])
    df['PropertyAge'] = sale_year - df['YearBuilt']
    # A handful of rows can end up negative (new construction sold
    # same year it was built recorded oddly, or data entry errors on
    # YearBuilt) -- clip at 0 rather than dropping, since new
    # construction is a real, valid case.
    df['PropertyAge'] = df['PropertyAge'].clip(lower=0)

    # Rows where BedBathRatio ended up NaN (BedroomsTotal == 0) would
    # otherwise silently drop out of the model later when we select
    # numeric_features -- drop them explicitly here so it's visible.
    before_ratio_drop = len(df)
    df = df.dropna(subset=['BedBathRatio'])
    if before_ratio_drop - len(df) > 0:
        print(f'  {month}: dropped {before_ratio_drop - len(df)} rows with BedroomsTotal=0 (BedBathRatio undefined)')

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

# Fill missing districts (properties outside all Unified district
# polygons) with an explicit category rather than leaving NaN, so
# get_dummies doesn't just silently drop those rows' district signal.
train_df['DistrictName'] = train_df['DistrictName'].fillna('Unknown')
test_df['DistrictName'] = test_df['DistrictName'].fillna('Unknown')

# One-hot encode zip code AND district. Train and test must be encoded
# TOGETHER, otherwise a category that appears in test but not train (or
# vice versa) creates mismatched columns between X_train and X_test.
train_df[zip_col] = train_df[zip_col].astype(str)
test_df[zip_col] = test_df[zip_col].astype(str)
combined = pandas.concat([train_df, test_df], keys=['train', 'test'])
combined = pandas.get_dummies(combined, columns=[zip_col, 'DistrictName'],
                               prefix=['zip', 'district'])
train_df = combined.xs('train')
test_df = combined.xs('test')

zip_dummy_cols = [c for c in combined.columns if c.startswith('zip_')]
district_dummy_cols = [c for c in combined.columns if c.startswith('district_')]
model_features = numeric_features + zip_dummy_cols + district_dummy_cols

X_train, y_train = train_df[model_features], train_df[target]
X_test, y_test = test_df[model_features], test_df[target]

# -------------------------------------------------------------------
# STEP 4: Baseline model -- Linear Regression
#
# Unlike Random Forest, linear models are sensitive to feature scale --
# LivingArea (hundreds/thousands) would otherwise dominate BedroomsTotal
# (single digits) in the coefficients regardless of true predictive
# power. Scale only the numeric features; the one-hot zip/district
# dummies (already 0/1) don't need scaling.
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
  enough rows relative to the number of zip-code/district dummy columns.
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
  especially with many one-hot zip/district columns providing lots of
  ways to split on location alone. Small changes in the training data
  can produce a very different tree, and predictions are
  piecewise-constant (it can only predict values seen in training
  leaves), which tends to hurt RMSE/MAE versus a smoother model.

Random Forest
  Strengths: Averaging many decorrelated trees reduces the variance/
  overfitting problem of a single Decision Tree while keeping the
  ability to model non-linearities and interactions. Typically the best
  R\u00b2 of the three here, and feature_importances_ gives a useful ranking
  of which fields (numeric features vs. specific zip/district dummies)
  drive price the most.
  Weaknesses: Slower to train and to run inference on than the other
  two models (300 trees vs. 1 tree vs. 1 linear model), less directly
  interpretable than a single tree or linear coefficients, and it still
  can't extrapolate outside the price/feature ranges seen in training
  (e.g. a brand-new luxury zip code with no sales history).

Feature engineering notes:
  - BedBathRatio (BathroomsTotalInteger / BedroomsTotal): rows with
    BedroomsTotal == 0 are dropped rather than assigned an infinite or
    arbitrary ratio.
  - PropertyAge (sale year - YearBuilt): computed using each month's
    sale year rather than today's date, so the feature reflects the
    property's age at time of transaction and stays consistent no
    matter when this script is re-run. Negative ages (data entry
    quirks) are clipped to 0.
  - DistrictName (via spatial join against Unified school district
    polygons): one-hot encoded alongside zip code. These two are
    collinear by construction (school districts and zip codes both
    encode location), so watch Linear Regression coefficient stability
    and compare feature_importances_ for zip_* vs district_* columns
    to see which one the tree models actually lean on.

Deliverable checklist:
  [x] Decision Tree and Random Forest regressors trained
  [x] Test R\u00b2 compared against Linear Regression baseline
      (see model_comparison_results.csv)
  [x] Model behavior (strengths/weaknesses) documented above
  [x] School district spatial join (Unified districts only)
  [x] BedBathRatio and PropertyAge feature engineering
"""

with open('week5_model_notes.txt', 'w') as f:
    f.write(notes)

print(notes)
print('Saved to week5_model_notes.txt')

# -------------------------------------------------------------------
# STEP 9: Evaluate trained models on a new, out-of-sample month
# (202606 -- one month past the original train/test window)
# -------------------------------------------------------------------
eval_month = '202606'


def preprocess_month(month):
    """Same cleaning steps as the Step 2 loop above, factored out so it
    can be reused here for a month that isn't part of `months`."""
    file_path = f'CRMLSSold{month}.csv'
    df = pandas.read_csv(file_path, low_memory=False)
    df = df[(df['PropertyType'] == 'Residential') &
            (df['PropertySubType'] == 'SingleFamilyResidence')]

    for field in categorical_fields:
        df[field] = df[field].fillna(False).astype(int)

    df = df.dropna(subset=features + [target])
    df = add_district_name(df)

    df['BedBathRatio'] = np.where(
        df['BedroomsTotal'] > 0,
        df['BathroomsTotalInteger'] / df['BedroomsTotal'],
        np.nan
    )
    sale_year = int(month[:4])
    df['PropertyAge'] = (sale_year - df['YearBuilt']).clip(lower=0)
    df = df.dropna(subset=['BedBathRatio'])

    df[zip_col] = df[zip_col].astype(str).str.split('.').str[0].str.strip()

    # NOTE: outlier removal here uses 202606's own IQR bounds, not the
    # training set's bounds. That's consistent with how every other
    # month was processed in Step 2, but it does mean a handful of
    # legitimately weird 202606 sales get filtered before scoring --
    # i.e. this evaluates the models on a "typical" slice of 202606,
    # not literally every closed sale that month.
    before = len(df)
    df = remove_iqr_outliers(df, outlier_check_cols)
    print(f'  {month}: removed {before - len(df)} outlier rows ({before} -> {len(df)})')

    return df


eval_df = preprocess_month(eval_month)
eval_df.to_csv(f'Processed_CRMLSSold{eval_month}.csv', index=False)
print(f'Processed {eval_month}: {len(eval_df)} rows')

# Fill missing district same as train/test above
eval_df['DistrictName'] = eval_df['DistrictName'].fillna('Unknown')
eval_df[zip_col] = eval_df[zip_col].astype(str)

# Save the target column first before reindexing the feature matrix
y_eval = eval_df[target]

# One-hot encode the evaluation set
eval_encoded = pandas.get_dummies(eval_df, columns=[zip_col, 'DistrictName'],
                                   prefix=['zip', 'district'])

# Reindex ONLY the predictor matrix so it matches model_features perfectly
X_eval = eval_encoded.reindex(columns=model_features, fill_value=0)

# Report any zips/districts in 202606 that weren't seen in training
new_zip_cols = [c for c in eval_df[zip_col].astype(str).apply(lambda z: f'zip_{z}').unique()
                if c not in zip_dummy_cols]
new_district_cols = [c for c in eval_df['DistrictName'].apply(lambda d: f'district_{d}').unique()
                      if c not in district_dummy_cols]
if new_zip_cols:
    print(f'  Warning: {len(new_zip_cols)} zip code(s) in {eval_month} unseen during training '
          f'(treated as no zip match): {new_zip_cols[:10]}{"..." if len(new_zip_cols) > 10 else ""}')
if new_district_cols:
    print(f'  Warning: {len(new_district_cols)} district(s) in {eval_month} unseen during training '
          f'(treated as no district match): {new_district_cols[:10]}{"..." if len(new_district_cols) > 10 else ""}')

# Linear Regression needs the same scaler fit on training data
X_eval_scaled = X_eval.copy()
X_eval_scaled[numeric_features] = scaler.transform(X_eval[numeric_features])
eval_linear_preds = baseline_model.predict(X_eval_scaled)

eval_tree_preds = tree_model.predict(X_eval)
eval_forest_preds = forest_model.predict(X_eval)

eval_results = []
for name, preds in [('Linear Regression', eval_linear_preds),
                     ('Decision Tree', eval_tree_preds),
                     ('Random Forest', eval_forest_preds)]:
    rmse = np.sqrt(mean_squared_error(y_eval, preds))
    mae = mean_absolute_error(y_eval, preds)
    r2 = r2_score(y_eval, preds)
    eval_results.append({'model': name, 'eval_month': eval_month, 'eval_size': len(eval_df),
                          'rmse': rmse, 'mae': mae, 'r2': r2})
    print(f'  {name:20s} RMSE=${rmse:,.0f}  MAE=${mae:,.0f}  R\u00b2={r2:.3f}')

print(f'\n=== Evaluation on out-of-sample month: {eval_month} ===')
eval_comparison = pandas.DataFrame(eval_results)
print(eval_comparison[['model', 'rmse', 'mae', 'r2']].to_string(index=False))

eval_comparison.to_csv(f'eval_results_{eval_month}.csv', index=False)
print(f'\nSaved to eval_results_{eval_month}.csv')