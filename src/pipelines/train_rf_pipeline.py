# train.py
import os
import argparse
import pandas as pd
import mlflow

import numpy as np
import joblib

from mlflow import sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

# Add os.sys.path to system path
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessor  # <- tu clase con engineered features

# Load environment variables from .env
load_dotenv()

# Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")


def load_data(data_path: str):
    """Load dataset"""
    return pd.read_csv(data_path)


def eval_metrics(y_true, y_pred):
    """Regression metrics en escala original (dólares)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main(args):
    print("🚀 Starting training pipeline...")

    # --- Config ---
    data_path = args.data

    # --- Load data ---
    df = load_data(data_path)

    # Target transformado
    y = np.log1p(df["PRICE"])   # label en log
    X = df.drop(columns=["PRICE"])  # dejamos que el Preprocessor seleccione columnas base

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- MLflow Tracking ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")

    param_grid = [
        {"n_estimators": 200, "max_depth": 10, "random_state": 42},
        # {"n_estimators": 300, "max_depth": 15, "random_state": 42},
    ]

    for i, params in enumerate(param_grid, start=1):
        with mlflow.start_run(run_name="model_training") as run:
            print(f"\nTraining model {i} with params: {params}")

            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"],
            )

            # --- Build pipeline: Preprocessor + Model ---
            preprocessor = Preprocessor(kmeans_model_path="./models/kmeans_model.joblib")
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            # --- Fit ---
            pipeline.fit(X_train, y_train)

            # --- Evaluate ---
            preds_log = pipeline.predict(X_test)
            y_pred = np.expm1(preds_log)   # volver a escala dólares
            y_true = np.expm1(y_test)

            rmse, mae, r2 = eval_metrics(y_true, y_pred)

            # --- Log params & metrics ---
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_param("preprocessor", "custom Preprocessor with engineered features")

            # --- Signature & Example ---
            example = X_test.iloc[:1]
            # Predicción en log porque el pipeline devuelve log
            signature = infer_signature(X_train, pipeline.predict(X_train))

            # --- Register model ---
            model_info = sklearn.log_model(
                pipeline,
                artifact_path="random_forest",
                registered_model_name="house-prices-predictor",
                input_example=example,
                signature=signature,
                metadata={
                    "dataset": os.path.basename(data_path),
                    "target": "PRICE_LOG (log1p transform of PRICE)",
                    "preprocessor": "custom Preprocessor with engineered features",
                },
            )

            model_uri_runs = f"{model_info.model_uri}"

            print(f"✅ Model {i} | RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, R2: {r2:.4f}")
            print(f"📌 Model URI (runs):   {model_uri_runs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price model")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the dataset CSV file (e.g. ./data/processed/train.csv)"
    )
    args = parser.parse_args()
    main(args)
