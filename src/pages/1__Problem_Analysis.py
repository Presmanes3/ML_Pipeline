import streamlit as st

st.set_page_config(page_title="Problem Analysis", layout="wide")

st.title("Problem Analysis: Market Insights")

st.markdown(
    "This project uses a dataset of properties for sale in New York, which can be found on "
    "[Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)."
)

# --------------------------
# Price over Location
# --------------------------
with st.expander("Price over Location", expanded=False):
    st.markdown(
        """
        We first mapped property prices across New York City to visualize their geographical distribution.
        """
    )

    st.image("./docs/figures/price_map.png", caption="Property prices in New York")

    st.markdown(
        """
        Some areas appear denser than others due to the use of a low alpha for overlapping points.

        To better capture the distribution, we plotted prices in **logarithmic scale**.
        """
    )

    st.image(
        "./docs/figures/price_log_map.png",
        caption="Property prices in New York (log scale)"
    )

# --------------------------
# Data Cleaning
# --------------------------
with st.expander("Data Cleaning", expanded=False):
    st.markdown(
        """
        The next step was to clean the data by removing **outliers** and erroneous values.

        We started by computing correlations among numeric variables and visualizing them with a heatmap.
        """
    )

    with st.expander("Correlation Matrix Before Cleaning", expanded=False):
        st.image(
            "./docs/figures/correlation_matrix_before_cleaning.png",
            caption="Correlation Matrix (Before Cleaning)"
        )

        st.markdown(
            """
            The correlation between `PRICE` and numeric variables such as `BEDS`, `BATH`, and `PROPERTY SQFT` is very weak.

            We also inspected a pairplot to see relationships among numeric variables.
            """
        )

        st.image(
            "./docs/figures/pairplot_before_cleaning.png",
            caption="Pairplot Before Cleaning"
        )

        st.markdown(
            """
            We can observe that data is highly clustered in some regions and very sparse in others.
            """
        )

    st.markdown(
        """
        We then applied **IQR-based outlier detection** and removed meaningless records (e.g., properties with 0 bedrooms or 0 bathrooms).

        Since the price distribution is heavily skewed, we applied a **logarithmic transformation** to the target `PRICE` to normalize its distribution.
        """
    )

    with st.expander("Price Distribution Before and After Cleaning", expanded=False):
        st.image(
            "./docs/figures/price_distribution_comparison.png",
            caption="Clean Price Distribution: Linear vs Log"
        )

    st.markdown(
        """
        We applied the same transformation to `PROPERTY SQFT`, which also exhibits a heavy-tailed distribution.

        Finally, we recomputed the pairplot and the correlation matrix to evaluate how relationships changed after cleaning.
        """
    )

    with st.expander("Correlation Matrix After Cleaning", expanded=False):
        st.image(
            "./docs/figures/correlation_matrix_after_cleaning.png",
            caption="Correlation Matrix (After Cleaning)"
        )

        st.markdown(
            """
            Now, the correlation between `PRICE_LOG` and other variables has increased—especially with `BATH`.

            A new pairplot confirms that data is better grouped and relationships are clearer.
            """
        )

        st.image(
            "./docs/figures/pairplot_after_cleaning.png",
            caption="Pairplot After Cleaning"
        )

        st.markdown(
            """
            **Summary:**
            - Outliers and invalid records removed.  
            - Log transformation applied to target and skewed numeric features.  
            - Data quality significantly improved for downstream analysis.
            """
        )

# --------------------------
# Feature Engineering
# --------------------------
with st.expander("Feature Engineering", expanded=False):
    st.markdown(
        """
        We engineered new features from existing ones to improve model performance.
        """
    )

    with st.expander("Structural Features", expanded=False):
        st.markdown(
            """
            Structural features help capture property quality with respect to its size and internal layout:

            - **PRICE_PER_SQFT_LOG**: Price per square foot in log scale — a better notion of relative price by size.  
            - **BEDS_PER_SQFT**: Bedrooms per square foot.  
            - **BATH_PER_SQFT**: Bathrooms per square foot.
            """
        )

    with st.expander("Geographical Features", expanded=False):
        st.markdown(
            """
            Geographical features help capture location effects and divide the city into zones:

            - **DISTANCE_TO_MANHATTAN**: Distance to Manhattan in miles.  
            - **LOCALITY**: City locality (zone-based grouping).  
            - **SUBLOCALITY**: Sub-locality (finer-grained zones).

            We then assessed the real relevance of these variables, since the dataset might be imbalanced or poorly labeled.
            To that end, we compared three clustering approaches: **Locality**, **SubLocality**, and **KMeans**.
            """
        )

        # Locality
        with st.expander("Locality Clustering", expanded=False):
            st.markdown(
                """
                The following map shows New York City with localities colored and boundaries marked.
                """
            )
            st.image("./docs/figures/localities_map.png", caption="Localities in New York")

            st.markdown(
                """
                Some locality labels (e.g., **“New York”**) cover the entire city, likely due to poor labeling.

                Price distribution by locality is heavily imbalanced:
                """
            )

            st.image(
                "./docs/figures/price_log_by_locality.png",
                caption="Locality PRICE Log Distribution"
            )

        # SubLocality
        with st.expander("SubLocality Clustering", expanded=False):
            st.markdown(
                """
                We then mapped sub-localities:
                """
            )
            st.image("./docs/figures/sublocalities_map.png", caption="Sublocalities in New York")

            st.markdown(
                """
                This produces more meaningful partitions, though some sublocalities (e.g., **“New York County”**) still span too broadly.

                Price distribution by sublocality:
                """
            )

            st.image(
                "./docs/figures/price_log_by_sublocality.png",
                caption="Locality Price LOG Distribution"
            )

            st.markdown(
                """
                Balance improves compared to locality clustering, but it remains uneven.
                """
            )

        # KMeans
        with st.expander("KMeans Clustering", expanded=False):
            st.markdown(
                """
                Finally, we applied **KMeans clustering** to find more relevant groupings based on property location.

                We used geographical coordinates (`LATITUDE`, `LONGITUDE`) and standardized features prior to clustering.
                """
            )

            st.image("./docs/figures/kmeans_clusters.png", caption="KMeans Clustering")

            st.markdown(
                """
                Clusters are much better defined and none of them spans the whole city.

                We can also relate prices to the clusters:
                """
            )

            st.image(
                "./docs/figures/kmeans_clusters_polygons.png",
                caption="Price by KMeans Clusters"
            )

            st.markdown(
                """
                Higher- and lower-price zones become clearer, and cluster balance improves (as shown below).
                """
            )

            st.image(
                "./docs/figures/boxplot_price_by_cluster.png",
                caption="KMeans Price LOG Distribution"
            )

st.markdown(
    """
    **Summary:**  
    - Engineered both structural and geographical features.  
    - Locality/SubLocality were not reliable due to poor labeling.  
    - **KMeans clustering** provided more meaningful and balanced segmentation.
    """
)
