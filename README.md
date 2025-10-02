# 🏠 ML Pipeline for House Price Prediction (MLflow + Streamlit)

A **reproducible end-to-end ML pipeline** for house price prediction in New-York, integrating:
- **Experiment tracking** with MLflow,
- **Model versioning** with the MLflow Registry,
- **Interactive dashboards** with Streamlit,
- **Artifact storage** with MinIO.

⚡ Designed to showcase best practices in **MLOps & Data Science**.

---

## 📂 Dataset
- Tabular dataset: [**New York House Prices**](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market) (regression).  

---

## 🔗 Project Pipeline
1. **EDA**  
    - Data ingestion, processing and cleaning with `pandas` and `scikit-learn`.  
    - Handling missing values and outliers.

2. **Feature Engineering**  
    - Use of `scikit-learn Pipelines` for reproducibility.  
    - Encoding categorical variables, scaling, and feature selection based on data correlation with house pricing.  

3. **Experiment Tracking**  
    - Integration with **MLflow**:  
      - Logging hyperparameters.  
      - Logging metrics (loss, accuracy, RMSE as applicable).  
      - Storing artifacts (models, plots).  

4. **Model Versioning**  
    - Saving the best model and versioning with MLflow Model Registry.  

5. **Interactive Visualization (Streamlit)**  
    - Dashboard to:  
      - Make predictions on new data (Playground).  
      - Project overall explanation and business logic.  

---

## 🚀 Clear Goals
- [X] Perform EDA and feature engineering on the dataset.
- [X] Build a reproducible pipeline for tabular data using scikit-learn. 
- [X] Integrate MLflow for experiment tracking.  
- [X] Save and version the best trained model.  
- [ ] Create a **Streamlit dashboard** for playground and project description.  

---

## 📌 Expected Results  
- Support decision-making with explainability visualizations (feature importance, locality distributions, etc).  

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
streamlit run ./src/Home.py
```

# MLflow Tracking UI

Access the MLflow Tracking UI at: [http://localhost:5000](http://localhost:5000)

# MinIO UI

Access the MinIO UI at: [http://localhost:9001](http://localhost:9001)