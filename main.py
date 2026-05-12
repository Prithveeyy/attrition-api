from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import pickle
import pandas as pd

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI()

# ============================================
# ENABLE CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# FALLBACK VALUES
# ============================================

FALLBACK_VALUES = {

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

@app.post("/predict")
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

    # ============================================
    # REPLACE NULL / ZERO VALUES
    # ============================================

    for key in input_data:

        value = input_data[key]

        if value is None or value == 0:

            input_data[key] = FALLBACK_VALUES[key]

            replacements[key] = FALLBACK_VALUES[key]

    # ============================================
    # DATAFRAME
    # ============================================

    input_df = pd.DataFrame(
        [input_data]
    )[TRAINING_COLUMNS]

    # ============================================
    # PREDICTION
    # ============================================

    probability = model.predict_proba(input_df)[0][1]

    risk_score = probability * 100

    stability_score = 100 - risk_score

    if risk_score > 50:
        prediction = "LEAVE RISK"
    else:
        prediction = "STABLE"

    # ============================================
    # RESPONSE
    # ============================================

    return {

        "prediction": prediction,

        "risk_score": round(risk_score, 2),

        "stability_score": round(stability_score, 2),

        "replacements": replacements
    }
