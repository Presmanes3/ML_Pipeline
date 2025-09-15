import os, sys
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("🏠 House Price Predictor - MLflow Comparison")

# -----------------------------
# Sidebar: select models
# -----------------------------
st.sidebar.header("Model Comparison")

available_models = {
    "Random Forest (sklearn)": {
        "name": "house-prices-predictor",
        "loader": mlflow.sklearn.load_model,
    },
    "Neural Network (PyTorch + Preproc)": {
        "name": "house-prices-nn",
        "loader": mlflow.pyfunc.load_model,  # Using pyfunc loader
    },
}

client = MlflowClient()

selected_families = st.sidebar.multiselect(
    "Choose models to compare",
    list(available_models.keys()),
    default=["Random Forest (sklearn)"],
)

loaded_models = {}
for family in selected_families:
    model_info = available_models[family]
    versions = client.search_model_versions(f"name='{model_info['name']}'")
    version_numbers = [int(v.version) for v in versions]
    latest_version = max(version_numbers)
    model_uri = f"models:/{model_info['name']}/{latest_version}"

    loaded_models[f"{family} v{latest_version}"] = model_info["loader"](model_uri)

st.write("✅ Models loaded:", list(loaded_models.keys()))

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("./data/processed/NY-House-Dataset-Cleaned.csv")

localities = df["LOCALITY_GROUPED"].unique().tolist()
types = df["TYPE"].unique().tolist()

beds_min, beds_max = int(df["BEDS"].min()), int(df["BEDS"].max())
bath_min, bath_max = int(df["BATH"].min()), int(df["BATH"].max())
sqft_min, sqft_max = int(df["PROPERTYSQFT"].min()), int(df["PROPERTYSQFT"].max())
dist_min, dist_max = float(df["DIST_TO_MANHATTAN"].min()), float(df["DIST_TO_MANHATTAN"].max())

# -----------------------------
# User input
# -----------------------------
st.header("🔮 Predict House Price")

locality = st.selectbox("Locality", localities)
house_type = st.selectbox("Type", types)
beds = st.slider("Number of Beds", beds_min, beds_max, 2)
bath = st.slider("Number of Baths", bath_min, bath_max, 1)
sqft = st.slider("Square Footage", sqft_min, sqft_max, 800)
dist = st.slider("Distance to Manhattan (miles)", dist_min, dist_max, step=0.01)

input_data = pd.DataFrame([{
    "LOCALITY_GROUPED": locality,
    "TYPE": house_type,
    "BEDS": beds,
    "BATH": bath,
    "SQFT_LOG": np.log(sqft),
    "DIST_TO_MANHATTAN": dist,
}])

# -----------------------------
# Predictions
# -----------------------------
results = {}
with st.spinner("Calculating predictions..."):
    for model_label, model in loaded_models.items():
        prediction_log = model.predict(input_data)[0]

        # Ensure all values are native floats
        prediction_log = float(prediction_log)
        mean_price = float(np.exp(prediction_log))
        price_per_sqft = float(mean_price / sqft)

        results[model_label] = {
            "log_price": prediction_log,
            "price": mean_price,
            "price_per_sqft": price_per_sqft,
        }

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction Comparison",
    "Price Distribution",
    "Distance vs Price",
    "Locality Comparison"
])

palette = sns.color_palette("tab10", n_colors=len(available_models))
model_colors = {name: palette[i] for i, name in enumerate(available_models.keys())}

# ---- Tab 1: Prediction ----
with tab1:
    st.subheader("Prediction Results (Comparison)")
    st.table(pd.DataFrame(results).T)

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, r in results.items():
        ax.bar(label, r["price"], color=model_colors[label.split(" v")[0]])
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Model Predictions Comparison")
    plt.xticks(rotation=30)
    st.pyplot(fig)

# ---- Tab 2: Price Distribution ----
with tab2:
    st.subheader("Price Distribution in Market")
    prices = np.exp(df["PRICE_LOG"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prices, bins=50, alpha=0.7, label="Training Data ($)", color="gray")

    for label, r in results.items():
        ax.axvline(
            r["price"],
            linestyle="--",
            linewidth=2,
            label=label,
            color=model_colors[label.split(" v")[0]],
        )

    ax.set_xlabel("House Price ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# ---- Tab 3: Distance vs Price ----
with tab3:
    st.subheader("Price vs Distance to Manhattan")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df,
        x="DIST_TO_MANHATTAN",
        y=np.exp(df["PRICE_LOG"]),
        alpha=0.4,
        ax=ax,
        color="gray",
    )

    for label, r in results.items():
        ax.scatter(
            dist,
            r["price"],
            marker="x",
            s=100,
            label=label,
            color=model_colors[label.split(" v")[0]],
        )

    ax.set_xlabel("Distance to Manhattan (miles)")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

# ---- Tab 4: Locality Comparison ----
with tab4:
    st.subheader("Price Distribution by Locality")
    df["PRICE_DOLLARS"] = np.exp(df["PRICE_LOG"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="LOCALITY_GROUPED",
        y="PRICE_DOLLARS",
        ax=ax,
    )

    for label, r in results.items():
        ax.scatter(
            localities.index(locality),
            r["price"],
            marker="x",
            s=100,
            label=label,
            color=model_colors[label.split(" v")[0]],
        )

    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Locality")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Price per sqft
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["PRICE_PER_SQFT"], bins=50, kde=True, ax=ax)

    for label, r in results.items():
        ax.axvline(
            r["price_per_sqft"],
            linestyle="--",
            label=f"{label} per sqft",
            color=model_colors[label.split(" v")[0]],
        )

    ax.set_xlabel("Price per sqft ($)")
    ax.legend()
    st.pyplot(fig)
