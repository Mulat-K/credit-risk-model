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

    
    # 4. Define High Risk
    # Analyze clusters: Usually High Recency + Low Frequency/Monetary = High Risk (Churned/Low Value)
    # For this example, let's assume Cluster 0 is the "worst" performing one.
    # In a real scenario, you print rfm.groupby('Cluster').mean() to decide.
    cluster_means = rfm.groupby('Cluster').mean()
    high_risk_cluster = cluster_means['Recency'].idxmax() # Assuming high recency = bad
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    # Merge target back to customer features
    final_df = customer_df.merge(rfm[['is_high_risk']], on='CustomerId', how='left')
    
    return final_df

def process_pipeline(input_path, output_path):
    df = load_data(input_path)
    
    # Task 3: Aggregates
    cust_df = create_aggregate_features(df)
    
    # Task 4: Target Engineering
    final_df = create_rfm_clusters(df, cust_df)
    
    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_pipeline('data/raw/data.csv', 'data/processed/train_data.csv')