import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce  # Requires: pip install category_encoders

# --- Custom Transformers ---

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features from TransactionStartTime."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure datetime
        if not np.issubdtype(X_copy['TransactionStartTime'].dtype, np.datetime64):
            X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
            
        X_copy['Tx_Hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['Tx_Day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['Tx_Month'] = X_copy['TransactionStartTime'].dt.month
        return X_copy[['Tx_Hour', 'Tx_Day', 'Tx_Month']]

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numerical features at the 99th percentile (Winsorization)."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.caps_ = {}

    def fit(self, X, y=None):
        # Calculate caps based on IQR or Percentiles
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                upper_cap = X[col].quantile(0.99)
                self.caps_[col] = upper_cap
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, cap in self.caps_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(upper=cap)
        return X_copy

# --- Pipeline Construction ---

def get_training_pipeline(categorical_cols, numerical_cols):
    """
    Constructs the full ML pipeline.
    
    Args:
        categorical_cols: List of columns for WoE/OneHot encoding.
        numerical_cols: List of columns for Scaling/Imputation.
    """
    
    # 1. Numerical Pipeline: Impute -> Cap Outliers -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', OutlierCapper()),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline: Weight of Evidence (WoE)
    # WoE is supervised; it needs 'y' in fit(). It helps monotonic relationship with risk.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('woe', ce.WOEEncoder(regularization=1.0)) 
    ])

    # 3. Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

# --- Unit Tests for Processing (Add to tests/test_data_processing.py) ---
def test_time_extractor():
    df = pd.DataFrame({'TransactionStartTime': ['2023-01-01 10:00:00']})
    transformer = TimeFeatureExtractor()
    res = transformer.fit_transform(df)
    assert res['Tx_Hour'][0] == 10
    assert 'Tx_Month' in res.columns

def test_outlier_capper():
    df = pd.DataFrame({'Amount': [10, 20, 10000]}) # 10000 is outlier
    capper = OutlierCapper()
    res = capper.fit_transform(df)
    # The 10000 should be capped at 99th percentile (approx ~9800 depending on quantile logic)
    assert res['Amount'].max() < 10000