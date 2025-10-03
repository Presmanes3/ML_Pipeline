import streamlit as st

st.set_page_config(page_title="Business Recap", layout="wide")

st.title("Business Recap")

st.markdown("""
This section provides a clear overview of the **business assumptions, conditions, and limitations** of the house price prediction model.  
It is meant to guide business users and stakeholders on how the model should be interpreted and where it is most useful.
""")

# ------------------- Objective -------------------
with st.expander("🎯 Business Objective"):
    st.markdown("""
    The model is designed to **estimate the market price of a residential property** 
    using structural and geographic features (e.g., number of bedrooms, bathrooms, square footage, latitude/longitude, and neighborhood zones).
    
    The purpose is to support **real estate valuation, benchmarking, and decision-making** — not to replace official appraisals.
    """)

st.markdown("The following expanders explain the **assumptions, data cleaning rules, feature engineering, and model validation process** in more detail.")

# ------------------- Assumptions -------------------
with st.expander("🧩 Business & Data Assumptions"):
    st.markdown("""
    1. Historical property prices are **representative of the current housing market**.  
    2. Input data has **sufficient quality** (e.g., no severe typos, unrealistic values).  
    3. Features used reflect **structural and geographic information only** — no qualitative factors such as interior design, views, or noise are included.  
    4. Relationships between features and price are assumed to be **stable over time**, with no major economic shocks considered.  
    """)

st.markdown("To ensure realistic predictions, a **data cleaning step** is applied before training and inference.")

# ------------------- Data Cleaning -------------------
with st.expander("🧹 Data Cleaning Rules (`DataCleaner`)"):
    st.markdown("""
    The dataset is filtered based on business constraints:
    
    - **Price range**: 100,000 – 3,000,000 USD  
    - **Bedrooms (BEDS)**: 1 – 7  
    - **Bathrooms (BATH)**: 1 – 6  
    - **Square footage (SQFT)**: 100 – 4,000  
    
    Any record outside of these thresholds is removed as an outlier.  
    This ensures the model does not learn from unrealistic or irrelevant properties.
    """)

st.markdown("Once cleaned, the dataset is transformed with **feature engineering** to capture better relationships between variables.")

# ------------------- Feature Engineering -------------------
with st.expander("⚙️ Feature Engineering (`Preprocessor`)"):
    st.markdown("""
    The following derived features are computed:
    
    - `log(PROPERTYSQFT)` → stabilizes variance of square footage.  
    - `bath_per_sqft = BATH / SQFT`  
    - `beds_per_sqft = BEDS / SQFT`  
    - `beds_per_bath = BEDS / BATH`  
    - `baths_per_bed = BATH / BEDS`  
    - **Cluster-based zone encoding** (`KMeansClusterEncoder`) using latitude/longitude.  
    - **Median imputation** for missing values in numeric features.  
    
    These transformations improve model robustness and capture hidden patterns (e.g., density of usage per room, neighborhood clustering).
    """)

st.markdown("Next, the model’s performance is evaluated under a robust validation strategy.")

# ------------------- Validation -------------------
with st.expander("📈 Validation & Metrics"):
    st.markdown("""
    - **Cross-validation**: 10-fold stratified CV (target binned into quantiles).  
    - **Metrics reported**:  
        - RMSE (Root Mean Squared Error) in USD scale  
        - MAE (Mean Absolute Error) in USD scale  
        - R² (explained variance)  
    
    Additional reporting:  
    - Mean, median, and standard deviation across folds  
    - Errors expressed as **% of median and mean house price**  
    
    This provides both an **absolute error range** and a **relative interpretation** of performance.
    """)

st.markdown("While the model is accurate for many cases, there are important **limitations** users should keep in mind.")

# ------------------- Limitations -------------------
with st.expander("🚫 Known Limitations"):
    st.markdown("""
    - **Qualitative factors not included**: renovations, interior design, views, noise, etc.  
    - Lower accuracy in **areas with few historical data points** (e.g., rural locations).  
    - Does not consider **macroeconomic trends** (mortgage rates, inflation, regulatory changes).  
    - Predictions are constrained by the cleaning thresholds (DataCleaner).  
    """)

st.markdown("Finally, here are the **conditions of use** and the roadmap for improvements.")

# ------------------- Conditions -------------------
with st.expander("✅ Conditions of Use"):
    st.markdown("""
    - Use as a **decision-support tool**, not as an official appraisal.  
    - Suitable for **exploratory analysis, comparisons, and initial valuations**.  
    - Should not be the **sole source of truth** for investment or pricing decisions.  
    """)

# ------------------- Next Steps -------------------
with st.expander("🚀 Next Steps & Future Improvements"):
    st.markdown("""
    - Integrate **external variables** (mortgage rates, socioeconomic indicators).  
    - Refine **geographic clusters** using richer market data (average neighborhood prices).  
    - Experiment with **alternative algorithms** (Gradient Boosting, XGBoost, Neural Networks).  
    - Implement **automated monitoring and retraining pipelines** to ensure long-term stability.  
    """)
