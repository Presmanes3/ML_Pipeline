import streamlit as st
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import yaml
import folium
from streamlit_folium import st_folium
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

# ---------------------
# Config
# ---------------------
st.set_page_config(page_title="House Price Playground", layout="wide")


mlflow.set_tracking_uri("http://localhost:5000")
    
client = MlflowClient()

# ---------------------
# Cached loaders
# ---------------------
@st.cache_resource
def load_model():
    local_model_dir = "./models/production_model"
    os.makedirs(local_model_dir, exist_ok=True)

    try:
        # --- Intentar sincronización con MLflow ---
        model_version = client.get_model_version_by_alias(
            name="RandomForestRegressor", alias="production"
        )
        model_uri = f"models:/RandomForestRegressor/{model_version.version}"

        # Descargar y sobrescribir SIEMPRE en ./models/production_model
        download_artifacts(model_uri, dst_path=local_model_dir)
        st.success("✅ Modelo sincronizado desde MLflow.")

    except Exception as e:
        st.warning(f"⚠️ No se pudo sincronizar con MLflow ({e}). Usando modelo local...")

    # --- Siempre usar el modelo local ---
    try:
        model = mlflow.sklearn.load_model(local_model_dir)
        return model, "local_model"
    except Exception as e2:
        st.error(f"❌ No se pudo cargar el modelo local: {e2}")
        raise e2


@st.cache_data
def load_metadata():
    local_model_dir = "./models/production_model"
    mlmodel_path = os.path.join(local_model_dir, "MLmodel")
    if os.path.exists(mlmodel_path):
        with open(mlmodel_path) as f:
            mlmodel_yaml = yaml.safe_load(f)
        return mlmodel_yaml.get("metadata", {})
    else:
        st.warning("⚠️ No se encontró el archivo MLmodel en el modelo local.")
        return {}

# ---------------------
# Load model + metadata
# ---------------------
model, model_uri = load_model()
metadata = load_metadata()

# ---------------------
# Session state init
# ---------------------
if "location" not in st.session_state:
    st.session_state["location"] = {"lat": 40.7128, "lon": -74.0060, "clicked": False}

if "history" not in st.session_state:
    st.session_state["history"] = []  # here we store each prediction

# ---------------------
# UI
# ---------------------
st.title("🏠 House Price Prediction Playground")

# ---- Vertical sliders ----
cleaner_params = metadata.get("cleaner_params", {})
min_beds = int(cleaner_params.get("min_beds", 1))
max_beds = int(cleaner_params.get("max_beds", 7))
min_bath = int(cleaner_params.get("min_bath", 1))
max_bath = int(cleaner_params.get("max_bath", 6))
min_sqft = int(cleaner_params.get("min_sqft", 100))
max_sqft = int(cleaner_params.get("max_sqft", 4000))

st.subheader("Enter the house features:")
beds = st.slider("Bedrooms", min_beds, max_beds, 2)
baths = st.slider("Bathrooms", min_bath, max_bath, 1)
sqft = st.slider("Area (sqft)", min_sqft, max_sqft, 800)

# ---- Map ----
st.subheader("📍 Choose location on the map")

m = folium.Map(
    location=[st.session_state["location"]["lat"], st.session_state["location"]["lon"]],
    zoom_start=11,
    tiles="CartoDB positron",
)

if st.session_state["location"]["clicked"]:
    folium.Marker(
        [st.session_state["location"]["lat"], st.session_state["location"]["lon"]],
        popup="Selected Location",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

map_data = st_folium(m, width=700, height=500, key="map")

if map_data and map_data.get("last_clicked"):
    st.session_state["location"]["lat"] = map_data["last_clicked"]["lat"]
    st.session_state["location"]["lon"] = map_data["last_clicked"]["lng"]
    st.session_state["location"]["clicked"] = True

# Show selected lat/lon
if st.session_state["location"]["clicked"]:
    st.info(
        f"**Selected Location**\n\n"
        f"Latitude: `{st.session_state['location']['lat']:.4f}`\n"
        f"Longitude: `{st.session_state['location']['lon']:.4f}`"
    )
else:
    st.warning("⚠️ Please click on the map to select a location.")

# ---- Prediction ----
if st.button("🚀 Run"):
    if not st.session_state["location"]["clicked"]:
        st.error("⚠️ Please select a location on the map before running.")
    else:
        input_df = pd.DataFrame([{
            "BEDS": beds,
            "BATH": baths,
            "PROPERTYSQFT": sqft,
            "LOCALITY": "",
            "SUBLOCALITY": "",
            "LATITUDE": st.session_state["location"]["lat"],
            "LONGITUDE": st.session_state["location"]["lon"],
        }])

        log_pred = model.predict(input_df)[0]
        price_pred = np.expm1(log_pred)

        st.success(f"💰 Estimated price: **${price_pred:,.0f}**")

        # ---- Save to history ----
        st.session_state["history"].append({
            "Beds": beds,
            "Baths": baths,
            "Sqft": sqft,
            "Latitude": round(st.session_state["location"]["lat"], 4),
            "Longitude": round(st.session_state["location"]["lon"], 4),
            "Predicted Price": f"${price_pred:,.0f}"
        })

# ---- Show history ----
if st.session_state["history"]:
    st.subheader("📊 Prediction History")
    history_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(history_df, use_container_width=True)

# ---- Model Performance Section ----
def fmt_currency(value):
    """Format as currency or return N/A"""
    if value is None:
        return "N/A"
    return f"${value:.2f}"

def fmt_percent(value, digits=2):
    """Format as percent or return N/A"""
    if value is None:
        return "N/A"
    return f"{value:.{digits}%}"

# ---- Model Performance Section ----
with st.expander("ℹ️ How to interpret model performance metrics", expanded=False):
    st.subheader("📈 Model Performance (Production)")

    if metadata:
        rmse = metadata.get("rmse_median")
        rmse_pct_median = metadata.get("rmse_median_pct_of_median_price")
        mean_price = metadata.get("mean_price")
        median_price = metadata.get("median_price")
        r2 = metadata.get("r2_median")
        mae = metadata.get("mae_median")
        mae_pct_median = metadata.get("mae_median_pct_of_median_price")

        st.markdown(f"""
        **Model Performance (Key Insights):**

        - ✅ Average prediction error (MAE): **{fmt_currency(mae)}** (~{fmt_percent(mae_pct_median)} of a typical house).  
        - ⚠️ Occasional larger errors (RMSE): **{fmt_currency(rmse)}** (~{fmt_percent(rmse_pct_median)} of the median price).  
        - 📊 The model explains about **{r2:.2f} of price variance (R²)** → solid but not perfect accuracy.  
        """)
    else:
        st.info("No performance metadata available for this model.")
