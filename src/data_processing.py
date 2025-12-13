import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

def load_data(filepath):
    return pd.read_csv(filepath)

def create_aggregate_features(df):
    """Aggregates transaction data to customer level."""
    
    # Extract Date Features
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Tx_Hour'] = df['TransactionStartTime'].dt.hour
    df['Tx_Month'] = df['TransactionStartTime'].dt.month
    
    # Aggregations
    agg_funcs = {
        'Amount': ['sum', 'mean', 'std', 'count'],
        'Value': ['sum'],
        'FraudResult': ['max'] # Did they ever commit fraud?
    }
    
    # Group by CustomerId
    customer_df = df.groupby('CustomerId').agg(agg_funcs)
    
    # Flatten MultiIndex columns
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    customer_df.reset_index(inplace=True)
    
    # Fill NaN from std calculation (customers with 1 transaction have NaN std)
    customer_df.fillna(0, inplace=True)
    
    return customer_df

def create_rfm_clusters(df_raw, customer_df):
    """
    Task 4: Create Proxy Target Variable via RFM Analysis.
    """
    # 1. Calculate RFM
    snapshot_date = pd.to_datetime(df_raw['TransactionStartTime']).max() + pd.Timedelta(days=1)
    
    rfm = df_raw.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days, # Recency
        'TransactionId': 'count', # Frequency
        'Amount': 'sum' # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    # 2. Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # 3. KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)