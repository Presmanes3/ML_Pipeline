import os
import argparse
import pandas as pd
import mlflow

import numpy as np
import matplotlib.pyplot as plt

from mlflow import sklearn
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

from src.preprocessing import Preprocessor  # custom feature engineering class
from src.data_cleaner import DataCleaner  # custom data cleaner class


# ------------------------
# Environment setup
# ------------------------
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"



# ------------------------
# Utilities
# ------------------------
def load_data(data_path: str):
    """Load dataset from CSV"""
    return pd.read_csv(data_path)


def eval_metrics(y_true, y_pred):
    """Compute regression metrics on original dollar scale"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def run_cross_validation(pipeline, X, y, n_folds):
    """Perform stratified cross-validation and log metrics to MLflow"""
    # Bin continuous target into categories for stratification
    kbins = KBinsDiscretizer(n_bins=n_folds, encode="ordinal", strategy="quantile")
    y_bins = kbins.fit_transform(y.values.reshape(-1, 1)).ravel().astype(int)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    rmse_scores, mae_scores, r2_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_bins), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)

        preds_val_log = pipeline.predict(X_val)
        y_val_true = np.expm1(y_val)
        y_val_pred = np.expm1(preds_val_log)

        rmse, mae, r2 = eval_metrics(y_val_true, y_val_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(f"Fold {fold} -> RMSE={rmse:,.2f}, MAE={mae:,.2f}, R2={r2:.4f}")

    # Aggregate metrics
    metrics = {
        "rmse": rmse_scores,
        "mae": mae_scores,
        "r2": r2_scores
    }

    for name, scores in metrics.items():
        mlflow.log_metric(f"cv_{name}_mean", float(np.mean(scores)))
        mlflow.log_metric(f"cv_{name}_std", float(np.std(scores)))
        mlflow.log_metric(f"cv_{name}_median", float(np.median(scores)))

    # Scatter plots
    for metric_name, scores in metrics.items():
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        median_val = np.median(scores)

        plt.figure(figsize=(6, 4))
        plt.scatter(range(1, len(scores) + 1), scores, color="steelblue", label="Scores")
        plt.axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
        plt.axhline(median_val, color="green", linestyle="-.", label=f"Median = {median_val:.2f}")
        plt.axhline(mean_val + std_val, color="orange", linestyle=":", label=f"Mean+STD = {mean_val+std_val:.2f}")
        plt.axhline(mean_val - std_val, color="orange", linestyle=":", label=f"Mean-STD = {mean_val-std_val:.2f}")

        plt.title(f"{metric_name.upper()} across folds ({n_folds}-fold CV)")
        plt.xlabel("Fold")
        plt.ylabel(metric_name.upper())
        plt.legend()
        plt.tight_layout()

        plot_path = f"cv_{metric_name.lower()}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # remove local plot file
        os.remove(plot_path)

    return metrics


# ------------------------
# Main training pipeline
# ------------------------
def main(args):
    print("🚀 Starting cross-validation pipeline...")

    # Load data
    df = load_data(args.data)
    
    # Clean data with business rules
    cleaner = DataCleaner(
        min_price   = 100_000, 
        max_price   = 3_000_000, 
        max_beds    = 7,
        min_beds    = 1,
        max_bath    = 6,
        min_bath    = 1,
        min_sqft    = 100,
        max_sqft    = 4_000,
    )
    df = cleaner.fit_transform(df)
    
    median_price = df["PRICE"].median()
    mean_price = df["PRICE"].mean()

    # Target transformation (log1p for stability)
    y = np.log1p(df["PRICE"])
    X = df.drop(columns=["PRICE"])

    # Define folds
    n_folds = 10

    # MLflow Tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")

    param_grid = [
        {"n_estimators": 100, "max_depth": 30, "random_state": 42},
        # {"n_estimators": 200, "max_depth": 10, "random_state": 42},
        # {"n_estimators": 300, "max_depth": 15, "random_state": 42},
        # {"n_estimators": 400, "max_depth": 20, "random_state": 42},
    ]

    for i, params in enumerate(param_grid, start=1):
        with mlflow.start_run(run_name=f"cv_model_training_{i}") as run:
            print(f"\nTraining model {i} with params: {params}")

            model = RandomForestRegressor(**params)

            # Build pipeline (Preprocessor + Model)
            preprocessor = Preprocessor(kmeans_model_path="./models/kmeans_model.joblib")
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            # Run CV
            metrics = run_cross_validation(pipeline, X, y, n_folds=n_folds)

            # Log params
            mlflow.log_params(params)
            mlflow.log_param("cv_type", f"stratified with {n_folds} bins & {n_folds} folds")

            median_rmse = np.median(metrics["rmse"])

            print(f"""
            ✅ Model {i} 
                CV RMSE (median): {median_rmse} ({median_rmse/median_price:.2%} of median price)
                CV RMSE (mean): {median_rmse} ({median_rmse/mean_price:.2%} of mean price)
                Median Price: {median_price}
                Mean Price: {mean_price}
                """)
            
            example = X.iloc[0:1]
            

            # Infer signature from example input and model output
            example_output = pipeline.predict(example)
            signature = infer_signature(example, example_output)
            
            model_info = mlflow.sklearn.log_model(
                sk_model                = pipeline,
                artifact_path           = "random_forest_regressor",
                registered_model_name   = "RandomForestRegressor",
                signature               = signature,
                input_example           = example,
                metadata={
                    "median_price": float(median_price),
                    "mean_price": float(mean_price),
                    "n_folds": int(n_folds),
                    "cv_type": "stratified",
                    "rmse_median": float(median_rmse),
                    "rmse_median_pct_of_median_price": float(median_rmse / median_price),
                    "rmse_median_pct_of_mean_price": float(median_rmse / mean_price),
                    "cleaner_params": cleaner.__dict__,
                    "localities" : X["LOCALITY"].unique(),
                    "sublocalities" : X["SUBLOCALITY"].unique(),
                }
            )
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price model with cross-validation")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the dataset CSV file (e.g. ./data/processed/train.csv)"
    )
    args = parser.parse_args()
    main(args)
