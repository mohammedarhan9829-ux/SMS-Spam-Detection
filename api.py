from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from src.predictor import SpamPredictor

app = FastAPI(
    title="SentinelSpam REST API",
    description="Production REST API & Webhook Service for Real-time SMS Spam Detection",
    version="2.0.0"
)

# Enable CORS for mobile apps, web frontend, and extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor
predictor = SpamPredictor(models_dir="models")

class SMSRequest(BaseModel):
    text: str
    sender: Optional[str] = None

class SMSResponse(BaseModel):
    label: str
    is_spam: bool
    probability: float
    risk_level: str
    cleaned_text: str
    top_words: List[dict]

@app.get("/")
def root():
    return {
        "service": "SentinelSpam REST API",
        "status": "Online",
        "endpoints": {
            "single_prediction": "POST /predict",
            "twilio_webhook": "POST /webhook/sms",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}

@app.post("/predict", response_model=SMSResponse)
def predict_sms(payload: SMSRequest):
    """
    Evaluates an incoming SMS message and returns classification verdict with risk score.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    result = predictor.predict_single(payload.text)
    return result

@app.post("/webhook/sms")
async def twilio_or_android_webhook(request: Request):
    """
    Webhook endpoint for Android SMS forwarders (Tasker/MacroDroid/SMS Gateway) or Twilio.
    Listens for incoming SMS and returns spam determination.
    """
    form_data = await request.form()
    json_data = {}
    try:
        json_data = await request.json()
    except Exception:
        pass

    # Extract text from standard webhook parameter keys
    body_text = form_data.get("Body") or form_data.get("text") or form_data.get("message") or json_data.get("text") or json_data.get("message")

    if not body_text:
        return {"status": "ignored", "reason": "No message body found"}

    result = predictor.predict_single(body_text)

    return {
        "status": "processed",
        "incoming_text": body_text,
        "is_spam": result['is_spam'],
        "probability": result['probability'],
        "action": "BLOCK" if result['is_spam'] else "ALLOW",
        "verdict": result['label']
    }
