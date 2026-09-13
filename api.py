from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import json
import io
import os
import uvicorn

from src.predictor import SpamPredictor

app = FastAPI(
    title="SentinelSpam — Complete Web App & Mobile API",
    description="Unified Web Dashboard & Mobile SMS Anti-Spam API",
    version="2.0.0"
)

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

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}

@app.post("/predict")
def predict_sms(payload: SMSRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    return predictor.predict_single(payload.text)

@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    raw_body_bytes = await request.body()
    raw_body_str = raw_body_bytes.decode('utf-8', errors='ignore')

    form_data = {}
    try:
        form_data = await request.form()
    except Exception:
        pass

    json_data = {}
    try:
        json_data = await request.json()
    except Exception:
        pass

    # Universal extractor for text from any parameter key or raw body
    body_text = (
        form_data.get("text") or form_data.get("body") or form_data.get("message") or
        form_data.get("sms") or form_data.get("sms_body") or form_data.get("notification_text") or
        form_data.get("Body") or form_data.get("Text") or
        json_data.get("text") or json_data.get("body") or json_data.get("message") or
        raw_body_str
    )

    if not body_text or not body_text.strip():
        return Response(status_code=204)

    result = predictor.predict_single(body_text)

    # Return HTTP 200 ONLY for Spam, HTTP 204 for Ham
    if result['is_spam']:
        res_payload = {
            "status": "spam_detected",
            "is_spam": True,
            "verdict": "Spam",
            "probability": result['probability'],
            "risk_level": result['risk_level'],
            "incoming_text": body_text[:60]
        }
        return JSONResponse(content=res_payload, status_code=200)
    else:
        return Response(status_code=204) # 204 No Content for legitimate messages

# Complete HTML/JS Interactive Web Dashboard on Render
@app.get("/", response_class=HTMLResponse)
def dashboard_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SentinelSpam - AI SMS Spam Filter & Mobile Gateway</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
        }
        body {
            background: var(--bg-gradient);
            color: #f8fafc;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding-bottom: 50px;
        }
        .navbar-brand {
            font-weight: 800;
            font-size: 1.5rem;
            color: var(--accent-blue) !important;
        }
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 24px;
        }
        .badge-spam {
            background-color: var(--accent-red);
            color: white;
            font-size: 1.3rem;
            font-weight: 700;
            padding: 8px 20px;
            border-radius: 30px;
            display: inline-block;
        }
        .badge-ham {
            background-color: var(--accent-green);
            color: white;
            font-size: 1.3rem;
            font-weight: 700;
            padding: 8px 20px;
            border-radius: 30px;
            display: inline-block;
        }
        .btn-primary-custom {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            border: none;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .btn-primary-custom:hover {
            background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
            color: white;
        }
        .progress-bar-spam {
            background: linear-gradient(90deg, #eab308 0%, #ef4444 100%);
        }
        .nav-tabs .nav-link {
            color: #94a3b8;
            border: none;
            font-weight: 600;
            padding: 12px 20px;
        }
        .nav-tabs .nav-link.active {
            color: var(--accent-blue);
            background: transparent;
            border-bottom: 3px solid var(--accent-blue);
        }
        pre {
            background: #090d16;
            color: #38bdf8;
            padding: 14px;
            border-radius: 10px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<nav class="navbar navbar-dark bg-transparent border-bottom border-secondary mb-4">
    <div class="container">
        <a class="navbar-brand" href="#"><i class="fa-solid fa-shield-halved me-2"></i>SentinelSpam ML Engine</a>
        <span class="badge bg-primary px-3 py-2"><i class="fa-solid fa-cloud me-1"></i> Live on Render</span>
    </div>
</nav>

<div class="container">
    <div class="text-center mb-4">
        <h1 class="fw-bold">AI SMS Spam Detection & Mobile Gateway</h1>
        <p class="text-secondary">Classify text messages with Machine Learning & connect your mobile SMS app for automated spam reminders</p>
    </div>

    <!-- Navigation Tabs -->
    <ul class="nav nav-tabs mb-4 justify-content-center" id="myTab" role="tablist">
        <li class="nav-item">
            <button class="nav-link active" id="single-tab" data-bs-toggle="tab" data-bs-target="#single" type="button"><i class="fa-solid fa-magnifying-glass me-2"></i>Single SMS Analyzer</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" id="mobile-tab" data-bs-toggle="tab" data-bs-target="#mobile" type="button"><i class="fa-solid fa-mobile-screen-button me-2"></i>Mobile SMS Reminder Setup</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" id="api-tab" data-bs-toggle="tab" data-bs-target="#api" type="button"><i class="fa-solid fa-code me-2"></i>API Endpoint Docs</button>
        </li>
    </ul>

    <div class="tab-content" id="myTabContent">
        
        <!-- TAB 1: Single SMS Analyzer -->
        <div class="tab-pane fade show active" id="single">
            <div class="glass-card">
                <h4 class="mb-3"><i class="fa-solid fa-paper-plane me-2 text-info"></i>Test SMS Prediction</h4>
                
                <div class="mb-3">
                    <label class="form-label text-secondary">Quick Test Presets:</label>
                    <div class="d-flex flex-wrap gap-2 mb-3">
                        <button class="btn btn-sm btn-outline-warning" onclick="setPreset('WINNER!! Claim £900 cash reward now! Call 09061701461 to claim. Valid 12 hours.')">🎁 Scam Reward</button>
                        <button class="btn btn-sm btn-outline-danger" onclick="setPreset('URGENT: Your account has been temporarily locked. Visit http://secure-bank-login.com to restore access.')">🏦 Phishing Alert</button>
                        <button class="btn btn-sm btn-outline-success" onclick="setPreset('Hey mate, are we still meeting up for dinner tonight at 7pm?')">💬 Genuine Chat</button>
                        <button class="btn btn-sm btn-outline-info" onclick="setPreset('Your OTP for account login is 492810. Do not share with anyone.')">🔑 OTP Code</button>
                    </div>
                </div>

                <div class="mb-3">
                    <textarea id="smsInput" class="form-control bg-dark text-light border-secondary" rows="4" placeholder="Enter or paste SMS message text here..."></textarea>
                </div>

                <button class="btn btn-primary-custom w-100 mb-4" onclick="analyzeSMS()"><i class="fa-solid fa-bolt me-2"></i>Analyze Message Now</button>

                <!-- Result Box -->
                <div id="resultBox" class="d-none">
                    <hr class="border-secondary mb-4">
                    <div class="row align-items-center">
                        <div class="col-md-5 text-center mb-3">
                            <div id="verdictBadge"></div>
                            <h3 class="mt-3 text-light" id="probText">0%</h3>
                            <p class="text-secondary mb-1">Spam Risk Score</p>
                            <div class="progress bg-dark mb-2" style="height: 12px;">
                                <div id="probBar" class="progress-bar progress-bar-spam" style="width: 0%;"></div>
                            </div>
                            <span class="badge bg-secondary" id="riskBadge">Low Risk</span>
                        </div>
                        <div class="col-md-7">
                            <h5><i class="fa-solid fa-circle-info me-2 text-info"></i>Analysis Metadata</h5>
                            <ul class="list-group list-group-flush bg-transparent">
                                <li class="list-group-item bg-transparent text-light border-secondary d-flex justify-content-between">
                                    <span>Cleaned Tokens:</span> <strong id="cleanedText" class="text-info"></strong>
                                </li>
                                <li class="list-group-item bg-transparent text-light border-secondary d-flex justify-content-between">
                                    <span>Character Count:</span> <strong id="charCount"></strong>
                                </li>
                                <li class="list-group-item bg-transparent text-light border-secondary d-flex justify-content-between">
                                    <span>Word Count:</span> <strong id="wordCount"></strong>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Mobile SMS Reminder Setup -->
        <div class="tab-pane fade" id="mobile">
            <div class="glass-card">
                <h3><i class="fa-solid fa-bell me-2 text-warning"></i>Connect Android Mobile SMS for Auto-Spam Reminders</h3>
                <p class="text-secondary">Follow these 3 easy steps to make your Android phone automatically read incoming SMS and show a notification reminder if it's Spam or Not Spam.</p>

                <div class="row mt-4">
                    <div class="col-md-4 mb-3">
                        <div class="p-3 border border-secondary rounded bg-dark h-100">
                            <h5>1. Install MacroDroid</h5>
                            <p class="text-secondary small">Download <strong>MacroDroid</strong> (Free) from the Google Play Store on your Android phone.</p>
                        </div>
                    </div>
                    <div class="col-md-4 mb-3">
                        <div class="p-3 border border-secondary rounded bg-dark h-100">
                            <h5>2. Add Trigger</h5>
                            <p class="text-secondary small">Tap <strong>Add Macro</strong> ➔ Trigger: <strong>SMS Received</strong> (Select Any Number).</p>
                        </div>
                    </div>
                    <div class="col-md-4 mb-3">
                        <div class="p-3 border border-secondary rounded bg-dark h-100">
                            <h5>3. Add HTTP Action</h5>
                            <p class="text-secondary small">Action: <strong>HTTP Request (POST)</strong> to your Render URL: <br><code id="renderUrlPlaceholder"></code></p>
                        </div>
                    </div>
                </div>

                <div class="mt-3">
                    <h5>HTTP Action Configuration Payload:</h5>
                    <pre>
Method: POST
URL: <span id="apiUrlDisplay"></span>/predict
Headers: Content-Type: application/json
Body:
{
  "text": "[sms_body]"
}
                    </pre>
                </div>
            </div>
        </div>

        <!-- TAB 3: API Endpoint Docs -->
        <div class="tab-pane fade" id="api">
            <div class="glass-card">
                <h3><i class="fa-solid fa-code me-2 text-info"></i>REST API Documentation</h3>
                <p class="text-secondary">Interactive Swagger API documentation is accessible at <code>/docs</code>.</p>

                <div class="mb-3">
                    <h5>Endpoint: <code>POST /predict</code></h5>
                    <pre>curl -X POST "<span class="apiUrlDisplay"></span>/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER!! Claim £900 cash reward now"}'</pre>
                </div>

                <a href="/docs" target="_blank" class="btn btn-outline-info"><i class="fa-solid fa-arrow-up-right-from-square me-2"></i>Open Full Swagger Interactive Docs</a>
            </div>
        </div>

    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    const currentUrl = window.location.origin;
    document.getElementById('apiUrlDisplay').innerText = currentUrl;
    document.getElementById('renderUrlPlaceholder').innerText = currentUrl + '/predict';
    const displays = document.querySelectorAll('.apiUrlDisplay');
    displays.forEach(el => el.innerText = currentUrl);

    function setPreset(text) {
        document.getElementById('smsInput').value = text;
    }

    async function analyzeSMS() {
        const text = document.getElementById('smsInput').value.trim();
        if (!text) {
            alert('Please enter a message to analyze.');
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            
            document.getElementById('resultBox').classList.remove('d-none');
            
            const badge = document.getElementById('verdictBadge');
            if (data.is_spam) {
                badge.className = 'badge-spam';
                badge.innerHTML = '🚨 SPAM DETECTED';
            } else {
                badge.className = 'badge-ham';
                badge.innerHTML = '✅ LEGITIMATE (HAM)';
            }

            document.getElementById('probText').innerText = data.probability + '%';
            document.getElementById('probBar').style.width = data.probability + '%';
            document.getElementById('riskBadge').innerText = data.risk_level;

            document.getElementById('cleanedText').innerText = data.cleaned_text || 'N/A';
            document.getElementById('charCount').innerText = data.metadata.num_characters;
            document.getElementById('wordCount').innerText = data.metadata.num_words;

        } catch (e) {
            alert('Error analyzing SMS: ' + e);
        }
    }
</script>
</body>
</html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
