# 📊 Business Recap: Assumptions, Rules & Limitations

This section provides an overview of the **business context, assumptions, and constraints** behind the house price prediction model.  
It helps stakeholders understand **how to interpret the model**, where it is reliable, and what limitations to consider.  

---

???+ info "🎯 Business Objective"
    The model estimates the **market price of residential properties** using structural and geographic features  
    (e.g., bedrooms, bathrooms, square footage, latitude/longitude, and neighborhood clusters).

    ✅ Purpose: support **real estate valuation, benchmarking, and decision-making**.  
    ❌ Not intended to replace **official appraisals**.

---

???+ info "🧩 Business & Data Assumptions"
    1. Historical property prices are assumed to be **representative of the current market**.  
    2. Input data must have **reasonable quality** (no severe typos or unrealistic values).  
    3. Only **structural and geographic features** are included — no qualitative aspects like design, views, or noise.  
    4. Relationships between features and price are assumed to be **stable over time**, excluding major economic shocks.  

---

???+ info "🧹 Data Cleaning Rules"
    Business constraints are applied to filter unrealistic records:  

    - **Price range**: 100,000 – 3,000,000 USD  
    - **Bedrooms**: 1 – 7  
    - **Bathrooms**: 1 – 6  
    - **Square footage**: 100 – 4,000  

    Properties outside these ranges are removed as outliers.  
    This prevents the model from learning from unrealistic cases.

---

???+ info "⚙️ Feature Engineering"
    Derived features improve robustness and capture hidden patterns:

    - `log(PROPERTYSQFT)` → stabilizes variance  
    - `BATHS_PER_SQFT = BATH / SQFT`  
    - `BEDS_PER_SQFT = BEDS / SQFT`  
    - `BEDS_PER_BATH = BEDS / BATH`  
    - `BATHS_PER_BED = BATH / BEDS`  
    - **Cluster-based zone encoding** (KMeans on lat/long)  
    - **Median imputation** for missing numeric values  

    These enhance interpretability and reduce noise in model inputs.

---

???+ info "📈 Validation & Metrics"
    Validation strategy ensures robust performance estimates:  

    - **10-fold stratified CV** (target binned into quantiles)  
    - Reported metrics:  
        - RMSE (USD)  
        - MAE (USD)  
        - R² (explained variance)  

    Additional reporting:  
    - Mean, median, and standard deviation across folds  
    - Errors as **% of median and mean house prices**  

    This allows both **absolute error evaluation** and **relative error interpretation**.

---

???+ info "🚫 Known Limitations"
    - Missing qualitative aspects: renovations, design, views, noise, etc.  
    - Lower accuracy in areas with **few historical data points** (e.g., rural zones).  
    - No consideration of **macroeconomic trends** (mortgage rates, inflation).  
    - Predictions constrained by cleaning thresholds (DataCleaner).  

---

???+ info "✅ Conditions of Use"
    - Use as **decision-support**, not official appraisal.  
    - Suitable for **exploratory analysis and benchmarking**.  
    - Should not be the **sole basis for investment decisions**.  

---

???+ info "🚀 Next Steps & Future Improvements"
    - Add external variables: mortgage rates, socioeconomic indicators.  
    - Refine geographic clusters with **market-level price data**.  
    - Test alternative algorithms (Gradient Boosting, XGBoost, Neural Nets).  
    - Implement **automated monitoring & retraining** pipelines.  

---

# ✅ Final Takeaways
- Business assumptions clearly defined.  
- Data cleaned with strict rules to remove outliers.  
- Feature engineering enhances structural and geographic signals.  
- Validation ensures robust, interpretable metrics.  
- Known limitations and future roadmap are acknowledged.  
