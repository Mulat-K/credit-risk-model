import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df):
    """
    Step 1: Calculate Recency, Frequency, and Monetary (RFM) metrics.
    """
    print("Calculating RFM metrics...")
    
    # Ensure datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Define snapshot date (usually the day after the last transaction in the dataset)
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Aggregate
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days, # Recency
        'TransactionId': 'count',                                         # Frequency
        'Amount': 'sum'                                                   # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    return rfm

def create_risk_label(rfm_df):
    """
    Step 2 & 3: Run K-Means and Label High-Risk Customers.
    High Risk is defined as the cluster with the highest Recency (inactive) 
    and lowest Frequency/Monetary (low engagement).
    """
    print("Running KMeans clustering for risk labeling...")
    
    # 1. Scaling (Crucial for KMeans)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    # 2. KMeans with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 3. Identify High-Risk Cluster Programmatically
    # We calculate the mean Recency for each cluster. 
    # The cluster with the HIGHEST Recency and LOWEST Frequency is usually 'High Risk' (Churned/Bad).
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    print("Cluster Summary:\n", cluster_summary)
    
    # Logic: High Risk = Max Recency
    high_risk_cluster_id = cluster_summary['Recency'].idxmax()
    print(f"Cluster {high_risk_cluster_id} identified as High Risk.")
    
    # 4. Assign Target Variable (1 = High Risk, 0 = Low Risk)
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)
    
    return rfm_df[['is_high_risk']] # Return only the target/index

def process_data(input_path, output_path):
    """
    Main pipeline to load, process, create target, and save.
    """
    # Load
    df = pd.read_csv(input_path)
    
    # 1. Feature Engineering (Basic Aggregations from previous steps)
    # (Simplified for brevity, ensure your create_aggregate_features logic is here)
    cust_features = df.groupby('CustomerId').agg({
        'Amount': ['mean', 'sum', 'std'],
        'FraudResult': 'max'
    })
    cust_features.columns = ['_'.join(col).strip() for col in cust_features.columns.values]
    cust_features.reset_index(inplace=True)
    cust_features.fillna(0, inplace=True)

    # 2. PROXY TARGET ENGINEERING (The core requirement)
    rfm_df = calculate_rfm(df)
    risk_labels = create_risk_label(rfm_df)
    
    # 3. Merge Labels back to Features
    final_df = cust_features.merge(risk_labels, on='CustomerId', how='left')
    
    # Drop CustomerId before training usually, or keep it to split later
    final_df.to_csv(output_path, index=False)
    print(f"Data processed and saved to {output_path}. Shape: {final_df.shape}")

if __name__ == "__main__":
    # Create dummy data if file doesn't exist for testing
    process_data('data/raw/data.csv', 'data/processed/train_data.csv')