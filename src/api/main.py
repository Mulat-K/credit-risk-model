import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.api.pydantic_models import CustomerData, PredictionResponse

# Global variables
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model on startup.
    In production, this URI points to the Model Registry (e.g., 'models:/CreditRiskModel/Production')
    For this implementation, we look for the latest local run or a fixed path.
    """
    global model
    try:
        # Example: Load from a specific run ID or local path
        # In a real CI/CD pipeline, the train step outputs the RUN_ID to an env var
        # model_uri = f"runs:/{os.getenv('MLFLOW_RUN_ID')}/model"
        
        # Fallback for local testing: Load from the 'mlruns' folder manually or assume a 'production_model' folder
        # For demonstration, we assume the user trained and saved to a local path or knows the URI
        model_uri = "./mlruns/0/latest_run_id/artifacts/model" # Update this logic dynamically
        
        # Simpler approach for assignment: Expect an environment variable or default to error
        if os.getenv("MODEL_URI"):
            model = mlflow.pyfunc.load_model(os.getenv("MODEL_URI"))
            print("Model loaded successfully.")
        else:
            print("Warning: MODEL_URI not set. API will fail on predict.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    yield
    # Clean up resources if needed

app = FastAPI(title="Credit Risk API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic object to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # The pipeline (WoE/Scaling) is inside the loaded model, 
        # so we pass raw features directly.
        probability = model.predict_proba(input_df)[:, 1][0]
        
        # Business Logic
        threshold = 0.5
        return {
            "customer_id": "cust_001", # Should pass this in request or generate
            "risk_probability": float(probability),
            "approved": bool(probability < threshold)
        }
    except Exception as e:
        # Log the error internally here
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")