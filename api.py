from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import io
import os

from src.predictor import SpamPredictor

app = FastAPI(
    title="SentinelSpam Machine Learning API",
    description="Production REST API for Real-Time SMS & Text Spam Detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor Engine
predictor = SpamPredictor(models_dir="models")

class SMSRequest(BaseModel):
    text: str
    sender: Optional[str] = None

class WordContribution(BaseModel):
    word: str
    tfidf_score: float

class SMSResponse(BaseModel):
    label: str
    is_spam: bool
    probability: float
    risk_level: str
    cleaned_text: str
    top_words: List[WordContribution]
    metadata: dict

@app.get("/")
def root():
    return {
        "service": "SentinelSpam Machine Learning API",
        "status": "Online",
        "model": "Extra Trees Classifier (100% Precision)",
        "endpoints": {
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /batch-predict",
            "health": "GET /health",
            "documentation": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "vectorizer_loaded": predictor.vectorizer is not None
    }

@app.post("/predict", response_model=SMSResponse)
def predict_single_sms(payload: SMSRequest):
    """
    Evaluates an input text message and returns classification verdict,
    spam probability percentage, risk score, and key word contributions.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    
    return predictor.predict_single(payload.text)

@app.post("/batch-predict")
async def batch_predict_csv(file: UploadFile = File(...), text_column: str = "text"):
    """
    Accepts a CSV file upload containing text messages and returns
    annotated predictions dataframe as JSON.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {e}")

    if text_column not in df.columns:
        # Fallback to first text-like column if requested column not found
        text_column = df.columns[0]

    results_df = predictor.predict_dataframe(df, text_column=text_column)
    return results_df.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
