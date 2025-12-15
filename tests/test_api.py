import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import MagicMock, patch

client = TestClient(app)

# Dummy data matching Pydantic model
payload = {
    "Amount": 500.0,
    "Tx_Hour": 14,
    "Tx_Day": 1,
    "ProductCategory": "Airtime",
    "ChannelId": "Android",
    "PricingStrategy": "Tier_1"
}

def test_predict_endpoint_success():
    """Test the API with a mocked model to ensure logic works."""
    
    # Mock the MLflow model behavior
    mock_model = MagicMock()
    # Mock predict_proba to return [[0.8, 0.2]] (High risk 0.2)
    mock_model.predict_proba.return_value = [[0.8, 0.2]] 
    
    with patch("src.api.main.model", mock_model):
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "risk_probability" in data
        assert data["risk_probability"] == 0.2
        assert data["approved"] is True # 0.2 < 0.5

def test_predict_malformed_input():
    """Test that the API rejects invalid data types."""
    bad_payload = payload.copy()
    bad_payload["Amount"] = "Not a Number"
    
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422 # Unprocessable Entity