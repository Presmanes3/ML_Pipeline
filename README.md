# Reproducible ML Pipeline with MLflow and Streamlit

🎯 **Objective:** Structure a reproducible machine learning pipeline, integrate experiment tracking with MLflow, and build an interactive Streamlit app to visualize results.

---

## 📂 Dataset
- Tabular dataset: **Titanic** (survival) or **House Prices** (regression).  

---

## 🔗 Project Pipeline
1. **Data Ingestion and Cleaning**  
    - Data processing with `pandas` or `Dask` for scalability.  
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
- [ ] Build a reproducible pipeline for tabular data.  
- [ ] Implement a feedforward model in PyTorch.  
- [ ] Integrate MLflow for experiment tracking.  
- [ ] Save and version the best trained model.  
- [ ] Create a **Streamlit dashboard** for predictions and experiment visualization.  
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
