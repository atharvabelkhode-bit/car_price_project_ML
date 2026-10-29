"""
CAR.py — Single-file Streamlit Car Price Predictor Dashboard

Keep this file in the same folder as cars_cleaned.csv.
If car_price_model.pkl and model_columns.pkl exist, the predictor uses them.
The app calculates a three-algorithm comparison itself, so model_metrics.csv is not needed.

Run:
    python -m streamlit run CAR.py
"""
import os
import warnings
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cars_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

st.set_page_config(page_title="AutoValue | Car Price Predictor", page_icon="🚘", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px;
background: linear-gradient(115deg,#101827 0%,#173b5e 60%,#087e8b 100%);
color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .45rem 0 0 0; color: #e5edf7;}
div[data-testid="stMetric"] {background: #111827; border: 1px solid #273449;
padding: 12px 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🚘 AutoValue — Car Price Intelligence</h1>
<p>📊 Dataset insights &nbsp; · &nbsp; 🧠 Machine learning comparison &nbsp; · &nbsp; 💰 Price estimation</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset(path, modified):
    return pd.read_csv(path)

@st.cache_resource
def compare_algorithms(csv_path, modified):
    data = pd.read_csv(csv_path)
    if "Price" not in data.columns:
        raise ValueError("The dataset needs a target column named 'Price'.")
    data = data.dropna(subset=["Price"]).copy()
    y = pd.to_numeric(data["Price"], errors="coerce")
    X = data.drop(columns=["Price"])
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    X = pd.get_dummies(X, dummy_na=False)
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)
    if len(X) < 10:
        raise ValueError("There are too few valid rows to compare models.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=18),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=20
        ),
    }
    rows = []
    fitted = {}
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)
        # Regression "accuracy" is an estimate based on MAPE, not classification accuracy.
        nonzero = y_test.abs() > 1e-9
        if nonzero.any():
            mape = ((y_test[nonzero] - predictions[nonzero]).abs() /
                    y_test[nonzero].abs()).mean() * 100
            accuracy = max(0.0, 100.0 - float(mape))
        else:
            accuracy = float("nan")
        rows.append({"Algorithm": name, "MAE": mae, "RMSE": rmse,
                     "R²": r2, "Accuracy estimate (%)": accuracy})
        fitted[name] = estimator
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, fitted, list(X.columns)

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Put cars_cleaned.csv in the same folder as CAR.py. Run 01_data_cleaning.ipynb first.")
    st.stop()

try:
    df = load_dataset(DATA_PATH, os.path.getmtime(DATA_PATH))
except Exception as e:
    st.error(f"Could not read cars_cleaned.csv: {e}")
    st.stop()

if "Price" not in df.columns:
    st.error("The dataset must contain a column named 'Price'. Check the column name in cars_cleaned.csv.")
    st.stop()

try:
    model_results, trained_models, training_columns = compare_algorithms(
        DATA_PATH, os.path.getmtime(DATA_PATH)
    )
except Exception as e:
    st.error(f"Could not compare algorithms: {e}")
    st.stop()

best_row = model_results.iloc[0]
best_algorithm = str(best_row["Algorithm"])

tab_overview, tab_predict, tab_compare, tab_data = st.tabs([
    "🏠 Overview", "💰 Price Predictor", "🏆 Algorithm Lab", "🗂️ Dataset Explorer"
])

with tab_overview:
    st.subheader("📌 Project Overview")
    prices = pd.to_numeric(df["Price"], errors="coerce").dropna()
    a, b, c = st.columns(3)
    a.metric("🚗 Cars in Dataset", f"{len(df):,}")
    b.metric("💵 Average Listed Price", f"{prices.mean():,.0f}" if len(prices) else "N/A")
    c.metric("📍 Median Listed Price", f"{prices.median():,.0f}" if len(prices) else "N/A")
    d, e, f = st.columns(3)
    d.metric("🏆 Best Algorithm", best_algorithm)
    e.metric("📉 Lowest RMSE", f'{best_row["RMSE"]:,.2f}')
    e2 = best_row["R²"]
    f.metric("📈 Best R² Score", f"{e2:.3f}" if pd.notna(e2) else "N/A")
    st.caption("The best algorithm is selected using the lowest RMSE on the test split. The accuracy estimate is 100 − MAPE, not classification accuracy.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 📊 Car Price Distribution")
        chart_prices = pd.DataFrame({"Price": prices})
        if not chart_prices.empty:
            st.bar_chart(chart_prices["Price"].value_counts(bins=20, sort=False))
            st.caption("Number of listings within each price range.")
    with right:
        st.markdown("#### 🏭 Average Price by Manufacturer")
        if "Manufacturer" in df.columns:
            brand = df.assign(Price=pd.to_numeric(df["Price"], errors="coerce"))
            brand = brand.groupby("Manufacturer")["Price"].mean().dropna().sort_values(ascending=False).head(12)
            if not brand.empty:
                st.bar_chart(brand)
        else:
            st.info("Manufacturer column is not present in this dataset.")

with tab_predict:
    st.subheader("💰 Estimate a Car's Price")
    st.write("Choose vehicle specifications and click **Predict Price**.")
    manufacturers = ["Audi", "BMW", "Chevrolet", "Ford", "Honda", "Hyundai", "Kia", "Mercedes-Benz", "Nissan", "Toyota"]
    categories = ["Coupe", "Hatchback", "Minivan", "SUV", "Sedan"]
    fuel_types = ["CNG", "Diesel", "Electric", "Hybrid", "Petrol"]
    gear_boxes = ["Automatic", "CVT", "Manual", "Tiptronic"]
    drive_wheels = ["4x4", "Front-wheel", "Rear-wheel"]
    colors = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            manufacturer = st.selectbox("🏭 Manufacturer", manufacturers)
            category = st.selectbox("🚘 Category", categories)
            fuel_type = st.selectbox("⛽ Fuel Type", fuel_types)
            gearbox = st.selectbox("⚙️ Gearbox Type", gear_boxes)
            drive = st.selectbox("🛞 Drive Wheels", drive_wheels)
        with col2:
            year = st.slider("📅 Production Year", 2000, 2025, 2018)
            engine = st.slider("🔧 Engine Volume (L)", 1.0, 6.0, 2.5, step=0.1)
            mileage = st.number_input("🛣️ Mileage (km)", min_value=0, max_value=300000, value=80000, step=1000)
            cylinders = st.selectbox("🔩 Cylinders", [3, 4, 6, 8, 12], index=1)
        with col3:
            doors = st.selectbox("🚪 Doors", [2, 4, 5], index=1)
            color = st.selectbox("🎨 Color", colors)
            leather = st.selectbox("🪑 Leather Interior", ["Yes", "No"])
            airbags = st.slider("🛡️ Airbags", 0, 9, 4)
            turbo = st.checkbox("⚡ Turbo engine")
        submitted = st.form_submit_button("💰 Predict Price", type="primary", use_container_width=True)

    if submitted:
        car = {
            "Manufacturer": manufacturer, "Prod_Year": year, "Category": category,
            "Leather_Interior": leather, "Fuel_Type": fuel_type, "Engine_Volume": engine,
            "Mileage": mileage, "Cylinders": cylinders, "Gear_box_type": gearbox,
            "Drive_Wheels": drive, "Doors": doors, "Color": color, "Airbags": airbags,
            "Turbo": int(turbo),
        }
        try:
            # Prefer the previously trained model if its files exist; otherwise use the best
            # algorithm trained in this app, with columns aligned to the training data.
            if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
                saved_model = joblib.load(MODEL_PATH)
                saved_columns = joblib.load(COLUMNS_PATH)
                row = pd.get_dummies(pd.DataFrame([car]))
                row = row.reindex(columns=saved_columns, fill_value=0).astype(float)
                predicted = float(saved_model.predict(row)[0])
                model_label = "Saved model from your notebook"
            else:
                row = pd.get_dummies(pd.DataFrame([car]))
                row = row.reindex(columns=training_columns, fill_value=0).astype(float)
                predicted = float(trained_models[best_algorithm].predict(row)[0])
                model_label = best_algorithm
            predicted = max(predicted, 0)
            st.success(f"### 💵 Estimated Price: {predicted:,.0f}")
            st.caption(f"Prediction generated using: {model_label}. This is an estimate; real prices vary with condition, location, and market.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check that the input field names and categories match those used in your training notebook.")

with tab_compare:
    st.subheader("🏆 Compare the Three Algorithms")
    st.write("Lower MAE/RMSE is better; an R² score closer to 1 is generally better.")
    st.success(f"🏆 Best algorithm by test RMSE: **{best_algorithm}**")
    display_results = model_results.copy()
    st.dataframe(
        display_results.style.format({
            "MAE": "{:,.2f}", "RMSE": "{:,.2f}",
            "R²": "{:.4f}", "Accuracy estimate (%)": "{:.2f}%"
        }),
        use_container_width=True, hide_index=True
    )
    st.markdown("#### 📉 RMSE Comparison")
    st.bar_chart(model_results.set_index("Algorithm")[["RMSE"]])
    st.markdown("#### 🎯 MAE Comparison")
    st.bar_chart(model_results.set_index("Algorithm")[["MAE"]])
    st.markdown("#### 📈 R² Score Comparison")
    st.bar_chart(model_results.set_index("Algorithm")[["R²"]])
    st.markdown("#### 🎯 Accuracy Estimate Comparison")
    st.bar_chart(model_results.set_index("Algorithm")[["Accuracy estimate (%)"]])

with tab_data:
    st.subheader("🗂️ Dataset Explorer")
    x, y, z = st.columns(3)
    x.metric("Rows", f"{df.shape[0]:,}")
    y.metric("Columns", f"{df.shape[1]:,}")
    z.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    st.markdown("#### 🔎 Preview cleaned data")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    if "Manufacturer" in df.columns:
        choices = sorted(df["Manufacturer"].dropna().astype(str).unique())
        selected = st.multiselect("Filter by manufacturer", choices)
        if selected:
            st.dataframe(df[df["Manufacturer"].astype(str).isin(selected)].head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download cleaned dataset", data=df.to_csv(index=False).encode("utf-8"),
        file_name="cars_cleaned.csv", mime="text/csv"
    )

st.divider()
st.caption("AutoValue | Car Price Intelligence • Python · Streamlit · pandas · scikit-learn")
