import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def plot_predictions(y_true, y_pred, output_dir):
    """Guardar gráficas de performance del modelo"""
    os.makedirs(output_dir, exist_ok=True)

    # Pred vs True
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True Values (log)")
    plt.ylabel("Predictions (log)")
    plt.title("True vs Predicted (log scale)")
    path1 = os.path.join(output_dir, "true_vs_pred.png")
    plt.savefig(path1)
    plt.close()

    # Residuales
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, edgecolor="black")
    plt.xlabel("Residual (log)")
    plt.ylabel("Frequency")
    plt.title("Residuals Distribution (log scale)")
    path2 = os.path.join(output_dir, "residuals_hist.png")
    plt.savefig(path2)
    plt.close()

    return [path1, path2]


def main(args):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")
    
    print("🔍 Loading model from MLflow...")
    model = mlflow.sklearn.load_model(args.model_uri)

    print("📂 Loading test dataset...")
    df = pd.read_csv(args.data)
    target = "PRICE_LOG"
    y_test = df[target]
    X_test = df.drop(columns=[target])

    print("⚡ Running predictions...")
    preds = model.predict(X_test)

    # Métricas en escala log
    rmse_log, mae_log, r2 = eval_metrics(y_test, preds)

    # Métricas en escala original
    y_test_orig = np.expm1(y_test)   # inversa de log1p
    preds_orig = np.expm1(preds)
    rmse_orig, mae_orig, _ = eval_metrics(y_test_orig, preds_orig)

    with mlflow.start_run(run_name="model_evaluation"):
        # Log metrics en log scale
        mlflow.log_metric("rmse", rmse_log)
        mlflow.log_metric("mae", mae_log)
        mlflow.log_metric("r2", r2)

        # Log metrics en escala original
        mlflow.log_metric("rmse_original", rmse_orig)
        mlflow.log_metric("mae_original", mae_orig)

        # Save and log plots
        artifact_dir = "./artifacts"
        plots = plot_predictions(y_test, preds, artifact_dir)
        for p in plots:
            mlflow.log_artifact(p, artifact_path="plots")

        # Log raw predictions (con y_true e y_pred en ambas escalas)
        results = pd.DataFrame({
            "y_true_log": y_test,
            "y_pred_log": preds,
            "y_true_original": y_test_orig,
            "y_pred_original": preds_orig
        })
        csv_path = os.path.join(artifact_dir, "predictions.csv")
        results.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="predictions")

        print(f"✅ Evaluation complete")
        print(f"   RMSE (log): {rmse_log:.4f}, MAE (log): {mae_log:.4f}, R2: {r2:.4f}")
        print(f"   RMSE (orig): {rmse_orig:.2f}, MAE (orig): {mae_orig:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MLflow model")
    parser.add_argument(
        "--model-uri", type=str, required=True,
        help="Model URI in MLflow (e.g. runs:/<run_id>/random_forest or models:/house-prices-predictor/Production)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the test dataset CSV (e.g. ./data/processed/test.csv)"
    )
    args = parser.parse_args()
    main(args)
