from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

# ============================================
# CREATE FASTAPI APP
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
# INPUT SCHEMA
# ============================================

class EmployeeData(BaseModel):

    EngagementScore: float
    TenureMonths: int
    StressScore: float
    PromotionDelay: int
    ExternalCompRatio: float
    InternalCompPosition: float
    SalaryStagnation: int
    EngagementTrend: float
    SentimentScore: float
    PerformanceScore: float
    ConflictScore: float

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

        "stability_score": round(stability_score, 2)
    }