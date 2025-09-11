import os
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def load_data(data_path: str):
    """Carga dataset procesado"""
    return pd.read_csv(data_path)


def build_pipeline(categorical_features, numeric_features, model_type="linear"):
    """Construye un pipeline con preprocesamiento + modelo"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def eval_metrics(y_true, y_pred):
    """Calcula métricas de regresión"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    # --- Configuración ---
    print("Iniciando pipeline de entrenamiento...")
    data_path = "./data/processed/NY-House-Dataset-Cleaned.csv"
    target = "PRICE_LOG"
    categorical_features = ["LOCALITY_GROUPED", "TYPE"]
    numeric_features = ["BEDS", "BATH", "SQFT_LOG", "DIST_TO_MANHATTAN"]

    model_type = "random_forest"

    # --- Cargar datos ---
    df = load_data(data_path)
    X = df[categorical_features + numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- MLflow Tracking ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")

    with mlflow.start_run():
        pipeline = build_pipeline(categorical_features, numeric_features, model_type)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        rmse, mae, r2 = eval_metrics(y_test, preds)

        # Log params
        mlflow.log_param("model_type", model_type)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
            # registered_model_name="house-prices-predictor"
        )

        print(f"Modelo: {model_type}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
