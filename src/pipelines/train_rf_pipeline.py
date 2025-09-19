# train.py
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

# Load environment variables from .env
load_dotenv()


def load_data(data_path: str):
    """Load dataset"""
    return pd.read_csv(data_path)


def build_pipeline(categorical_features, numeric_features, model):
    """Build pipeline with preprocessing + model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def eval_metrics(y_true, y_pred):
    """Regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main(args):
    print("🚀 Starting training pipeline...")

    # --- Config ---
    data_path = args.data
    target = "PRICE_LOG"
    categorical_features = ["TYPE", "LOCALITY"]
    numeric_features = ["BEDS_PER_SQFT", "BATH_PER_SQFT", "SQFT_LOG", "DIST_TO_MANHATTAN"]

    # --- Load data ---
    df = load_data(data_path)
    X = df[categorical_features + numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- MLflow Tracking ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")

    param_grid = [
        {"n_estimators": 200, "max_depth": 10, "random_state": 42},
        {"n_estimators": 300, "max_depth": 15, "random_state": 42},
        {"n_estimators": 400, "max_depth": 20, "random_state": 42},
        {"n_estimators": 500, "max_depth": 25, "random_state": 42},
    ]

    for i, params in enumerate(param_grid, start=1):
        with mlflow.start_run(run_name="model_training") as run:
            print(f"\nTraining model {i} with params: {params}")

            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"],
            )

            pipeline = build_pipeline(categorical_features, numeric_features, model)
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            rmse, mae, r2 = eval_metrics(y_test, preds)

            # Log params & metrics
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_param("features", f"""{categorical_features + numeric_features}""")

            # Add input example + signature
            example = X_test.iloc[:1]
            signature = infer_signature(X_train, pipeline.predict(X_train))

            # Register model
            model_info = mlflow.sklearn.log_model(
                pipeline,
                artifact_path="random_forest",
                registered_model_name="house-prices-predictor",
                input_example=example,
                signature=signature,
                metadata={
                    "dataset": os.path.basename(data_path),
                    "features": f"""{categorical_features + numeric_features}""",
                },
            )

            model_uri_runs = f"{model_info.model_uri}"

            print(f"✅ Model {i} | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"📌 Model URI (runs):   {model_uri_runs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price model")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the dataset CSV file (e.g. ./data/processed/train.csv)"
    )
    args = parser.parse_args()
    main(args)
