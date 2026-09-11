import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CRMLS Home Price Predictor", page_icon="🏠")

# ---- Load model + metadata (all produced by the Step 1 cell in the notebook) ----
model = joblib.load("model.pkl")
model_features = joblib.load("model_features.pkl")
zip_options = joblib.load("zip_options.pkl")
district_options = joblib.load("district_options.pkl")

st.title("🏠 CRMLS Home Price Predictor")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms Total", min_value=0, value=3, step=1)
    bathrooms = st.number_input("Bathrooms Total (Integer)", min_value=1, value=2, step=1)
    living_area = st.number_input("Living Area (sqft)", min_value=0, value=1500, step=10)
    lot_size = st.number_input("Lot Size (sqft)", min_value=0, value=6000, step=10)
    year_built = st.number_input(
        "Year Built", min_value=1800, max_value=datetime.date.today().year, value=1990
    )
    garage_spaces = st.number_input("Garage Spaces", min_value=0, value=2, step=1)
    zip_code = st.selectbox("ZIP / Postal Code", zip_options)
    district = st.selectbox("District", district_options)

with col2:
    view = st.selectbox("View", ["No", "Yes"])
    waterfront = st.selectbox("Waterfront", ["No", "Yes"])
    basement = st.selectbox("Basement", ["No", "Yes"])
    pool = st.selectbox("Private Pool", ["No", "Yes"])
    attached_garage = st.selectbox("Attached Garage", ["No", "Yes"])
    fireplace = st.selectbox("Fireplace", ["No", "Yes"])
    new_construction = st.selectbox("New Construction", ["No", "Yes"])

if st.button("Predict Price"):
    # --- Engineered features ---------------------------------------------
    # ASSUMPTION — verify against your data-prep step for
    # Processed_CRMLSSold*.csv before trusting predictions.
    bed_bath_ratio = bedrooms / bathrooms if bathrooms else 0
    property_age = datetime.date.today().year - year_built

    # --- Build a single-row feature vector matching model_features exactly ---
    row = pd.Series(0, index=model_features, dtype=float)

    numeric_values = {
        "BedroomsTotal": bedrooms,
        "BathroomsTotalInteger": bathrooms,
        "LivingArea": living_area,
        "LotSizeSquareFeet": lot_size,
        "YearBuilt": year_built,
        "GarageSpaces": garage_spaces,
        "ViewYN": 1 if view == "Yes" else 0,
        "WaterfrontYN": 1 if waterfront == "Yes" else 0,
        "BasementYN": 1 if basement == "Yes" else 0,
        "PoolPrivateYN": 1 if pool == "Yes" else 0,
        "AttachedGarageYN": 1 if attached_garage == "Yes" else 0,
        "FireplaceYN": 1 if fireplace == "Yes" else 0,
        "NewConstructionYN": 1 if new_construction == "Yes" else 0,
        "BedBathRatio": bed_bath_ratio,
        "PropertyAge": property_age,
    }
    for feat, val in numeric_values.items():
        if feat in row.index:
            row[feat] = val

    zip_col = f"zip_{zip_code}"
    district_col = f"district_{district}"
    if zip_col in row.index:
        row[zip_col] = 1
    else:
        st.warning(f"'{zip_col}' wasn't in the training data — prediction may be less reliable.")
    if district_col in row.index:
        row[district_col] = 1
    else:
        st.warning(f"'{district_col}' wasn't in the training data — prediction may be less reliable.")

    X = pd.DataFrame([row])[model_features]  # enforce exact training column order
    predicted_price = model.predict(X)[0]

    st.success(f"💰 Estimated Home Price: ${predicted_price:,.0f}")
