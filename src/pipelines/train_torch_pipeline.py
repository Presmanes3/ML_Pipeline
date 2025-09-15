import os
import pandas as pd
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import numpy as np

import torch
import torch.nn as nn
import joblib

from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.model import SimpleModel  # your model defined as nn.Module


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
# Custom PyFunc Wrapper
# --------------------
class HousePriceNNWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, context, model_input: pd.DataFrame):
        # Apply the preprocessor
        X_processed = self.preprocessor.transform(model_input).toarray()
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)

        # Make predictions with the model
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()

        return preds


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

    # Load data
    df = load_data(data_path)
    X = df[categorical_features + numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
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

    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ML-Pipeline")

    with mlflow.start_run(run_name="simple_nn") as run:
        input_dim = X_train_tensor.shape[1]
        lr = 0.001
        epochs = 400

        # Log parameters
        mlflow.log_param("model_type", "SimpleModel")
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)

        model = SimpleModel(input_dim=input_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training
        train_model(model, train_loader, criterion, optimizer, epochs)

        # Predictions
        preds = predict_model(model, X_test_tensor)
        rmse, mae, r2 = eval_metrics(y_test_tensor.numpy(), preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log preprocessor
        joblib.dump(preprocessor, "preprocessor.pkl")
        mlflow.log_artifact("preprocessor.pkl", artifact_path="preprocessor")

        # Log as pyfunc (encapsulates NN + preprocessor)
        wrapped_model = HousePriceNNWrapper(model, preprocessor)
        mlflow.pyfunc.log_model(
            artifact_path="house-prices-nn",
            python_model=wrapped_model,
            registered_model_name="house-prices-nn",
        )

        print(f"Final metrics | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
