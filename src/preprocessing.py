import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin    
 
class KMeansClusterEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.kmeans = joblib.load(model_path)
        self.model_path = model_path
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        
    def fit(self, X, y=None):
        clusters = self.kmeans.predict(X)
        self.encoder.fit(clusters.reshape(-1, 1))
        return self

    def transform(self, X):
        clusters = self.kmeans.predict(X)
        return self.encoder.transform(clusters.reshape(-1, 1)).toarray()

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out([ "ZONE" ])
 
 
class Preprocessor:
    def __init__(self, kmeans_model_path=None):
        self.fitted = False
        
        self.base_columns = ["BEDS", "BATH", "PROPERTYSQFT",
                             "LOCALITY", "SUBLOCALITY", "LATITUDE", "LONGITUDE"]

        # Log de SQFT
        log_sqft = Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=False))
        ])
        
        def bath_per_sqft(X: pd.DataFrame):
            return (X["BATH"] / (X["PROPERTYSQFT"] + 1e-5)).to_numpy().reshape(-1, 1)

        def beds_per_sqft(X: pd.DataFrame):
            return (X["BEDS"] / (X["PROPERTYSQFT"] + 1e-5)).to_numpy().reshape(-1, 1)

        def beds_per_bath(X: pd.DataFrame):
            return (X["BEDS"] / (X["BATH"] + 1e-5)).to_numpy().reshape(-1, 1)

        def baths_per_bed(X: pd.DataFrame):
            return (X["BATH"] / (X["BEDS"] + 1e-5)).to_numpy().reshape(-1, 1)

        # Cluster encoder si hay KMeans
        cluster_encoder = KMeansClusterEncoder(kmeans_model_path) if kmeans_model_path else "drop"

        # ColumnTransformer global
        self.pipeline = ColumnTransformer([
            ("log_sqft", log_sqft, ["PROPERTYSQFT"]),
            ("bath_per_sqft", FunctionTransformer(bath_per_sqft, validate=False), ["BATH", "PROPERTYSQFT"]),
            ("beds_per_sqft", FunctionTransformer(beds_per_sqft, validate=False), ["BEDS", "PROPERTYSQFT"]),
            ("beds_per_bath", FunctionTransformer(beds_per_bath, validate=False), ["BEDS", "BATH"]),
            ("baths_per_bed", FunctionTransformer(baths_per_bed, validate=False), ["BATH", "BEDS"]),
            ("imputer", SimpleImputer(strategy="median"),
             ["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE"]),
            ("zones", cluster_encoder, ["LATITUDE", "LONGITUDE"])
        ])

    def fit(self, X, y=None):
        # construyes pipeline interno (ColumnTransformer + tus features)
        self.pipeline.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet")
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        