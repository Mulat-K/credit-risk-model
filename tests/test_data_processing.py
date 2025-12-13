import pandas as pd
import pytest
from src.data_processing import create_aggregate_features

def test_aggregate_features():
    # Create dummy data
    data = {
        'CustomerId': [1, 1, 2],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-01 12:00:00'],
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50],
        'FraudResult': [0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    agg_df = create_aggregate_features(df)
    
    # Assertions
    assert agg_df.shape[0] == 2 # 2 unique customers
    assert 'Amount_sum' in agg_df.columns
    assert agg_df.loc[agg_df['CustomerId'] == 1, 'Amount_sum'].values[0] == 300