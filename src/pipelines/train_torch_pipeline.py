import os
import pandas as pd
import mlflow
import mlflow.pytorch
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.models.model import SimpleModel  # tu red definida como nn.Module

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------
# Helpers
# --------------------
def load_data(data_path: str):
    return pd.read_csv(data_path)

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# --------------------
# Entrenamiento
# --------------------
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

def predict_model(model, X_tensor):
    model.eval()
    with torch.no_grad():
        return model(X_tensor).numpy()

# --------------------
# Main
# --------------------
def main():
    load_dotenv()
    print("Starting training pipeline...")

    data_path = "./data/processed/NY-House-Dataset-Cleaned.csv"
    target = "PRICE_LOG"
    categorical_features = ["LOCALITY_GROUPED", "TYPE"]
    numeric_features = ["BEDS", "BATH", "SQFT_LOG", "DIST_TO_MANHATTAN"]

    # --- Load data ---
    df = load_data(data_path)
    X = df[categorical_features + numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Convert to tensor
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # --- MLflow ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")

    with mlflow.start_run(run_name="simple_nn"):
        input_dim = X_train_tensor.shape[1]
        lr = 0.001
        epochs = 50

        # Log params
        mlflow.log_param("model_type", "SimpleModel")
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)

        model = SimpleModel(input_dim=input_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Entrenamiento
        train_model(model, train_loader, criterion, optimizer, epochs)

        # Predicciones
        preds = predict_model(model, X_test_tensor)
        rmse, mae, r2 = eval_metrics(y_test_tensor.numpy(), preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Registrar modelo
        mlflow.pytorch.log_model(
            model,
            name="pytorch-model",
            registered_model_name="house-prices-nn"
        )

        print(f"Final metrics | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
