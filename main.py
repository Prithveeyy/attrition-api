from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pickle
import pandas as pd

# ============================================
# CREATE FASTAPI APP
# ============================================

app = FastAPI()

# ============================================
# LOAD MODEL
# ============================================

with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================
# TRAINING COLUMNS
# ============================================

TRAINING_COLUMNS = [
    "EngagementScore",
    "TenureMonths",
    "StressScore",
    "PromotionDelay",
    "ExternalCompRatio",
    "InternalCompPosition",
    "SalaryStagnation",
    "EngagementTrend",
    "SentimentScore",
    "PerformanceScore",
    "ConflictScore"
]

# ============================================
# INPUT SCHEMA
# ============================================

class EmployeeData(BaseModel):

    EngagementScore: Optional[float] = None

    TenureMonths: Optional[int] = None

    StressScore: Optional[float] = None

    PromotionDelay: Optional[int] = None

    ExternalCompRatio: Optional[float] = None

    InternalCompPosition: Optional[float] = None

    SalaryStagnation: Optional[int] = None

    EngagementTrend: Optional[float] = None

    SentimentScore: Optional[float] = None

    PerformanceScore: Optional[float] = None

    ConflictScore: Optional[float] = None

# ============================================
# HOME ROUTE
# ============================================

@app.get("/")
def home():

    return {
        "message": "Attrition Prediction API is Running",
        "model_loaded": True
    }

# ============================================
# PREDICTION ROUTE
# ============================================

def predict(data: EmployeeData):

    replacements = {}

input_data = {

    "EngagementScore": data.EngagementScore,
    "TenureMonths": data.TenureMonths,
    "StressScore": data.StressScore,
    "PromotionDelay": data.PromotionDelay,
    "ExternalCompRatio": data.ExternalCompRatio,
    "InternalCompPosition": data.InternalCompPosition,
    "SalaryStagnation": data.SalaryStagnation,
    "EngagementTrend": data.EngagementTrend,
    "SentimentScore": data.SentimentScore,
    "PerformanceScore": data.PerformanceScore,
    "ConflictScore": data.ConflictScore
}

# ── FALLBACK / MEAN REPLACEMENTS ──

fallback_values = {

    "EngagementScore": 5,
    "TenureMonths": 24,
    "StressScore": 5,
    "PromotionDelay": 2,
    "ExternalCompRatio": 1.0,
    "InternalCompPosition": 5,
    "SalaryStagnation": 2,
    "EngagementTrend": 0.0,
    "SentimentScore": 0.68,
    "PerformanceScore": 5,
    "ConflictScore": 3
}

for key in input_data:

    value = input_data[key]

    if value is None or value == 0:

        input_data[key] = fallback_values[key]

        replacements[key] = fallback_values[key]

    input_df = pd.DataFrame(
        [input_data]
    )[TRAINING_COLUMNS]

    probability = model.predict_proba(input_df)[0][1]

    risk_score = probability * 100

    stability_score = 100 - risk_score

    if risk_score > 50:
        prediction = "LEAVE RISK"
    else:
        prediction = "STABLE"

return {

    "prediction": prediction,

    "risk_score": round(risk_score, 2),

    "stability_score": round(stability_score, 2),

    "replacements": replacements
}
