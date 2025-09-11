# Reproducible ML Pipeline with MLflow and Streamlit

🎯 **Objective:** Structure a reproducible machine learning pipeline, integrate experiment tracking with MLflow, and build an interactive Streamlit app to visualize results.

---

## 📂 Dataset
- Tabular dataset: **House Prices** (regression).  

---

## 🔗 Project Pipeline
1. **Data Ingestion and Cleaning**  
    - Data processing with `pandas`for scalability.  
    - Handling missing values and outliers.  

2. **Feature Engineering**  
    - Use of `scikit-learn Pipelines` for reproducibility.  
    - Encoding categorical variables, scaling, and feature selection.  

3. **Model**  
    - Feedforward Neural Network implemented in **PyTorch**.  
    - Flexible architecture to allow tuning.  

4. **Experiment Tracking**  
    - Integration with **MLflow**:  
      - Logging hyperparameters.  
      - Logging metrics (loss, accuracy, RMSE as applicable).  
      - Storing artifacts (models, plots).  

5. **Model Versioning**  
    - Saving the best model and versioning with MLflow Model Registry.  

6. **Interactive Visualization (Streamlit)**  
    - Dashboard to:  
      - Load the trained model and make predictions on new data.  
      - Visualize MLflow runs and compare experiments.  
      - Display feature importance and evaluation metrics.  

---

## 🧪 Extras
- Use of **PyTorch Lightning** to simplify the training loop.  
- Comparison of runs in MLflow UI for hyperparameter optimization.  
- Streamlit integration with **MLflow Tracking Server** to dynamically fetch results.  

---

## 🚀 Clear Goals
- [X] Build a reproducible pipeline for tabular data.  
- [ ] Implement a feedforward model in PyTorch.  
- [X] Integrate MLflow for experiment tracking.  
- [X] Save and version the best trained model.  
- [X] Create a **Streamlit dashboard** for predictions and experiment visualization.  
- [ ] (Optional) Rewrite training using PyTorch Lightning.  

---

## 📌 Expected Results
- Reproducible and modular pipeline for tabular data.  
- Experiment tracking in MLflow with metrics and artifacts.  
- Versioned model ready for deployment.  
- Interactive Streamlit app to:  
  - Input new samples for prediction.  
  - Visualize training metrics, experiment comparisons, and feature importance.  

---

# Installation

Create a .env file and set the variables based on the [.env.example](./.env.example) file.

Then, run docker compose as:
```bash
docker compose --env-file <path_to_your_env_file> up -d
```

Replace `<path_to_your_env_file>` with the path to your desired `.env` file.

# Streamlit App

Execute the following command to run the Streamlit application:

```bash
streamlit run ./src/app.py
```

# MLflow Tracking UI

Access the MLflow Tracking UI at: [http://localhost:5000](http://localhost:5000)

# MinIO UI

Access the MinIO UI at: [http://localhost:9001](http://localhost:9001)






