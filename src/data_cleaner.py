from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 min_price=50_000, 
                 max_price=10_000_000, 
                 max_beds=10,
                 min_beds=1,
                 max_bath=10,
                 min_bath=1,
                 max_sqft=20_000,
                 min_sqft=100):
        
        self.min_price = min_price
        self.max_price = max_price
        self.max_beds = max_beds
        self.min_beds = min_beds
        self.max_bath = max_bath
        self.min_bath = min_bath
        self.max_sqft = max_sqft
        self.min_sqft = min_sqft

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        original_len = len(X)
        mask = (
            (X["PRICE"] >= self.min_price) &
            (X["PRICE"] <= self.max_price) &
            (X["BEDS"] <= self.max_beds) &
            (X["BEDS"] >= self.min_beds) &
            (X["BATH"] <= self.max_bath) &
            (X["BATH"] >= self.min_bath) &
            (X["PROPERTYSQFT"] <= self.max_sqft) &
            (X["PROPERTYSQFT"] >= self.min_sqft)
        )
        
        filtered_len = mask.sum()
        print(f"DataCleaner: Filtered {original_len - filtered_len} out of {original_len} rows.")
        
        return X.loc[mask].reset_index(drop=True)

    def flag_outliers(self, X):
        return (
            (X["PRICE"] < self.min_price) |
            (X["PRICE"] > self.max_price) |
            (X["BEDS"] > self.max_beds) |
            (X["BEDS"] < self.min_beds) |
            (X["BATH"] > self.max_bath) |
            (X["BATH"] < self.min_bath) |
            (X["PROPERTYSQFT"] > self.max_sqft) |
            (X["PROPERTYSQFT"] < self.min_sqft)
        )
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
