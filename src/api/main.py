from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from .pydantic_models import CustomerData, PredictionResponse

app = FastAPI(title="Bati Bank Credit Risk API")

# Load model (In production, load from Model Registry or artifact path)
# Assuming Random Forest was best and saved locally for this example
# MODEL_URI = "models:/Credit_Risk_Model_Bati/Production" 
# model = mlflow.sklearn.load_model(MODEL_URI)

# Mock model loading for the script to run without a live MLflow server
try:
    # This path depends on where you run it from or a specific path
    # For now, we simulate a loading
    model = None 
    print("Model loading logic goes here")
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerData):
    # Convert input to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make Prediction
    # probability = model.predict_proba(input_data)[:, 1][0]
    probability = 0.25 # Mock result
    
    # Define Threshold
    threshold = 0.5
    approved = probability < threshold
    
    return {
        "customer_id": "req_123", # Usually passed in
        "risk_probability": probability,
        "approved": approved
    }