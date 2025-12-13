from pydantic import BaseModel

class CustomerData(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_count: int
    Value_sum: float
    FraudResult_max: int
    # Add all other features expected by the model

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    approved: bool