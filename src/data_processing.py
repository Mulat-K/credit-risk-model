import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce  # pip install category_encoders

# --- 1. Custom Feature Extractors ---

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features: Year, Month, Day, Hour.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure datetime format
        if not np.issubdtype(X_copy['TransactionStartTime'].dtype, np.datetime64):
            X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        
        # Extract features
        X_copy['Tx_Year'] = X_copy['TransactionStartTime'].dt.year
        X_copy['Tx_Month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['Tx_Day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['Tx_Hour'] = X_copy['TransactionStartTime'].dt.hour
        
        # Drop original timestamp column to prevent model errors
        return X_copy.drop(columns=['TransactionStartTime'])

# --- 2. Pipeline Construction ---

def get_preprocessing_pipeline(woe_cols, ohe_cols, num_cols):
    """
    Constructs a reproducible Scikit-Learn pipeline.
    
    Args:
        woe_cols (list): Columns to apply Weight of Evidence encoding (High cardinality).
        ohe_cols (list): Columns to apply One-Hot Encoding (Low cardinality).
        num_cols (list): Numerical columns for scaling.
    """
    
    # Numerical Pipeline: Impute -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline 1: One-Hot Encoding (for ChannelId, etc.)
    ohe_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Categorical Pipeline 2: Weight of Evidence (for ProviderId, ProductCategory)
    # WoE is excellent for credit scoring as it maps categories to risk monotonic trends.
    woe_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('woe', ce.WOEEncoder(regularization=1.0)) 
    ])

    # Combine all into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('ohe', ohe_transformer, ohe_cols),
            ('woe', woe_transformer, woe_cols)
        ],
        remainder='drop' # Drop columns not specified (like IDs)
    )
    
    return preprocessor

# --- 3. Wrapper for Data Loading & Initial Processing ---

def load_and_process_raw(filepath):
    """
    Loads raw data and performs initial aggregations (RFM) to create the training set.
    (This logic prepares the 'processed' CSV, while the pipeline above handles 'in-model' transformations)
    """
    df = pd.read_csv(filepath)
    
    # ... [Insert your RFM and Target Creation Logic from Task 4 here] ...
    # For brevity, assuming df now has 'is_high_risk' and features
    
    return df