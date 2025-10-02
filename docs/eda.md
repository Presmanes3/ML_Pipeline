# 🏙️ Problem Analysis: Market Insights

This project uses a dataset of properties for sale in New York, which can be found on [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market).

---

<details>
<summary>📍 Price over Location</summary>

We first mapped property prices across New York City to visualize their geographical distribution.

![Property prices in New York](/figures/price_map.png)

Some areas appear denser than others due to the use of a low alpha for overlapping points.  

To better capture the distribution, we plotted prices in **logarithmic scale**:

![Property prices in New York (log scale)](figures/price_log_map.png)

</details>

---

<details>
<summary>🧹 Data Cleaning</summary>

The next step was to clean the data by removing **outliers** and erroneous values.

### 🔸 Correlation Matrix Before Cleaning
![Correlation Matrix](figures/correlation_matrix_before_cleaning.png)

The correlation between `PRICE` and numeric variables such as `BEDS`, `BATH`, and `PROPERTY SQFT` is very weak.

We also visualized this with a pairplot:

![Pairplot Before Cleaning](figures/pairplot_before_cleaning.png)

Data appears very clustered in some regions and very sparse in others.

---

We then applied outlier detection using **IQR** and removed meaningless records (e.g., properties with 0 bedrooms or bathrooms).  

Since the price distribution is heavily skewed, we applied a **logarithmic transformation** to `PRICE` (and later to `PROPERTY SQFT`) to normalize the distribution.

![Clean Price Distribution Linear vs Log](figures/price_distribution_comparison.png)

---

### 🔸 Correlation Matrix After Cleaning
![Correlation Matrix After Cleaning](figures/correlation_matrix_after_cleaning.png)

Now, the correlation between `PRICE_LOG` and other variables has increased, especially with `BATH`.  

A new pairplot confirms that data is better grouped and relationships are clearer:  

![Pairplot After Cleaning](figures/pairplot_after_cleaning.png)

**Summary:**  
- Outliers and invalid records removed.  
- Log transformation applied to target and skewed numeric features.  
- Data quality significantly improved for downstream analysis.  

</details>

---

<details>
<summary>🛠️ Feature Engineering</summary>

We engineered new features from existing ones to improve model performance.

### 🔸 Structural Features
- **PRICE_PER_SQFT_LOG** → Price per square foot in log scale.  
- **BEDS_PER_SQFT** → Bedrooms per square foot.  
- **BATH_PER_SQFT** → Bathrooms per square foot.  

### 🔸 Geographical Features
- **DISTANCE_TO_MANHATTAN** → Distance to Manhattan in miles.  
- **LOCALITY** → City locality (zone-based grouping).  
- **SUBLOCALITY** → Sub-locality (finer-grained zones).  

---

<details>
<summary>📌 Locality Clustering</summary>

We mapped New York localities:

![Localities in New York](figures/localities_map.png)

Some locality labels (e.g., "New York") cover the entire city, likely due to poor labeling.  

Price distribution by locality shows heavy imbalance:  

![Locality PRICE Log Distribution](figures/price_log_by_locality.png)

</details>

<details>
<summary>📌 SubLocality Clustering</summary>

We then mapped sub-localities:

![Sublocalities in New York](figures/sublocalities_map.png)

This gives more meaningful partitions, though some sublocalities are still inconsistent (e.g., "New York County" spans too broadly).  

Price distribution by sublocality:  

![Locality Price LOG Distribution](figures/price_log_by_sublocality.png)

Balance is improved compared to locality clustering, but still uneven.  

</details>

<details>
<summary>📌 KMeans Clustering</summary>

We applied **KMeans clustering** using standardized latitude/longitude coordinates.  

![KMeans Clustering](figures/kmeans_clusters.png)  
![Price by KMeans Clusters](figures/kmeans_clusters_polygons.png)  

The clusters are much better defined and balanced.  

Price distribution by KMeans cluster:  
![KMeans Price LOG Distribution](figures/price_log_by_cluster.png)

</details>

---

**Summary:**  
- Engineered both structural and geographical features.  
- Locality/SubLocality were not reliable due to poor labeling.  
- **KMeans clustering** provided more meaningful and balanced segmentation.  

</details>

---

# ✅ Final Takeaways

- Data cleaned (outliers removed, log transformations applied).  
- Engineered structural and geographical features.  
- Evaluated clustering approaches → KMeans proved most useful.  
- Result: higher data quality, clearer feature relationships, and better modeling potential.  
