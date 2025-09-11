import os
import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from sklearn.inspection import permutation_importance

# Load env vars
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("🏠 House Price Predictor - MLflow Integration")

# Sidebar: select model version
st.sidebar.header("Model Selection")
model_name = "house-prices-predictor"

# Get available versions
client = MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
version_numbers = [v.version for v in versions]
selected_version = st.sidebar.selectbox("Choose model version", version_numbers)

# Load model from registry
model_uri = f"models:/{model_name}/{selected_version}"
model = mlflow.sklearn.load_model(model_uri)

st.write(f"Loaded **{model_name} v{selected_version}**")

# Load df
df = pd.read_csv("./data/processed/NY-House-Dataset-Cleaned.csv")

localities = df["LOCALITY_GROUPED"].unique().tolist()
types = df["TYPE"].unique().tolist()

beds_min, beds_max = int(df["BEDS"].min()), int(df["BEDS"].max())
bath_min, bath_max = int(df["BATH"].min()), int(df["BATH"].max())
sqft_min, sqft_max = int(df["PROPERTYSQFT"].min()), int(df["PROPERTYSQFT"].max())
dist_min, dist_max = float(df["DIST_TO_MANHATTAN"].min()), float(df["DIST_TO_MANHATTAN"].max())

# User input for prediction
st.header("🔮 Predict House Price")
locality = st.selectbox("Locality", localities)
house_type = st.selectbox("Type", types)
beds = st.slider("Number of Beds", beds_min, beds_max, 2)
bath = st.slider("Number of Baths", bath_min, bath_max, 1)
sqft = st.slider("Square Footage", sqft_min, sqft_max, 800)
dist = st.slider("Distance to Manhattan (miles)", dist_min, dist_max, step=0.01)

# Build input dataframe
input_data = pd.DataFrame([{
    "LOCALITY_GROUPED": locality,
    "TYPE": house_type,
    "BEDS": beds,
    "BATH": bath,
    "SQFT_LOG": np.log(sqft),
    "DIST_TO_MANHATTAN": dist,
}])

# Tabs for results
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prediction",
    "Price Distribution",
    "Distance vs Price",
    "Locality Comparison",
    "Feature Importance"
])

with st.spinner("Recalculating prediction..."):
    prediction_log = model.predict(input_data)[0]

with tab1:
    st.subheader("Prediction Result")

    # Get log-price prediction
    prediction_log = model.predict(input_data)[0]
    mean_price = np.exp(prediction_log)
    price_per_sqft = mean_price / sqft

    st.success(f"💰 Predicted price: ${mean_price:,.2f}")
    st.info(f"📏 Price per sqft: ${price_per_sqft:,.2f}")

    # Save in session_state so other tabs can use it
    st.session_state["prediction"] = mean_price
    st.session_state["price_per_sqft"] = price_per_sqft

with tab2:
    st.subheader("Price Distribution in Market")
    if "prediction" in st.session_state:
        mean_price = st.session_state["prediction"]
        prices = np.exp(df["PRICE_LOG"])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(prices, bins=50, alpha=0.7, label="Training Data ($)")
        ax.axvline(mean_price, color="red", linestyle="--", linewidth=2, label="Prediction")
        ax.set_xlabel("House Price ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.subheader("Price vs Distance to Manhattan")
    if "prediction" in st.session_state:
        mean_price = st.session_state["prediction"]

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=df,
            x="DIST_TO_MANHATTAN",
            y=np.exp(df["PRICE_LOG"]),
            alpha=0.4,
            ax=ax
        )
        ax.scatter(dist, mean_price, color="red", marker="x", s=100, label="Prediction")
        ax.set_xlabel("Distance to Manhattan (miles)")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

with tab4:
    st.subheader("Price Distribution by Locality")
    if "prediction" in st.session_state:
        mean_price = st.session_state["prediction"]

        fig, ax = plt.subplots(figsize=(10, 5))

        df["PRICE_DOLLARS"] = np.exp(df["PRICE_LOG"])

        sns.boxplot(
            data=df,
            x="LOCALITY_GROUPED",
            y="PRICE_DOLLARS",
            ax=ax
        )

        ax.scatter(
            localities.index(locality),
            st.session_state["prediction"],
            color="red",
            marker="x",
            s=100,
            label="Prediction"
        )
        ax.set_ylabel("Price ($)")
        ax.set_xlabel("Locality")
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

        # Price per sqft comparison
        price_per_sqft = st.session_state["price_per_sqft"]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["PRICE_PER_SQFT"], bins=50, kde=True, ax=ax)
        ax.axvline(price_per_sqft, color="red", linestyle="--", label="Prediction per sqft")
        ax.set_xlabel("Price per sqft ($)")
        ax.legend()
        st.pyplot(fig)

# with tab5:
#     st.subheader("Feature Importance (Permutation Importance)")
    
#     X = df[["LOCALITY_GROUPED", "TYPE", "BEDS", "BATH", "SQFT_LOG", "DIST_TO_MANHATTAN"]]
#     y = df["PRICE_LOG"]

#     X_transformed = model.named_steps["preprocessor"].transform(X).toarray()
#     feature_names = model.named_steps["preprocessor"].get_feature_names_out()

#     result = permutation_importance(
#         model.named_steps["model"], 
#         X_transformed, 
#         y, 
#         n_repeats=10, 
#         random_state=42,
#         n_jobs=-1
#     )

#     perm_importances = pd.DataFrame({
#         "feature": feature_names,
#         "importance_mean": result.importances_mean,
#         "importance_std": result.importances_std
#     }).sort_values("importance_mean", ascending=True)

#     # Plot
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.barh(perm_importances["feature"], perm_importances["importance_mean"],
#             xerr=perm_importances["importance_std"], alpha=0.7)
#     ax.set_title("Permutation Importance (with std dev)")
#     st.pyplot(fig)
