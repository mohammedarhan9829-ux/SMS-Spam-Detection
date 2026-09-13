# 📱 Automated Android SMS Spam Detector Setup Guide (MacroDroid / Tasker)

This guide explains how to connect your Android phone to the **SentinelSpam API** using **MacroDroid** (Free on Google Play Store). When a new SMS arrives on your phone, MacroDroid automatically sends it to your deployed API, checks if it is Spam, and displays an instant push notification on your phone!

---

## 🛠️ Step 1: Deploy your API Online

### Free Option 1: Render.com (Recommended)
1. Go to [Render.com](https://render.com/) and log in with GitHub (`mohammedarhan9829-ux`).
2. Click **New +** -> **Web Service**.
3. Select your repository `mohammedarhan9829-ux/SMS-Spam-Detection`.
4. Set settings:
   - **Name**: `sentinel-spam-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.
6. Render will build your app and give you a public URL (e.g. `https://sentinel-spam-api.onrender.com`).

---

## 📲 Step 2: Configure MacroDroid on Your Android Phone

1. Install **[MacroDroid](https://play.google.com/store/apps/details?id=com.arlosoft.macrodroid)** from Google Play Store.
2. Open MacroDroid and tap **Add Macro**.

### A. Trigger (When SMS Received)
1. Tap **+** under **Triggers**.
2. Select **Messaging** -> **SMS Received**.
3. Choose **Select Contact(s)** -> Select **Any Contact** (or Any Number).

### B. Action (Send to ML API)
1. Tap **+** under **Actions**.
2. Select **Applications** -> **HTTP Request** (or **Open Website / HTTP POST**).
3. Set properties:
   - **Method**: `POST`
   - **URL**: `https://sentinel-spam-api.onrender.com/predict`  *(replace with your Render API URL)*
   - **Content Type**: `application/json`
   - **Body text**:
     ```json
     {
       "text": "[sms_body]"
     }
     ```
4. Save Response to a MacroDroid variable named `api_response`.

### C. Notification Action (Show Alert if Spam)
1. Add an Action: **Notification** -> **Display Notification**.
2. Title: `🚨 SPAM ALERT DETECTED`
3. Text: `Message from [sms_number]: [sms_body]`
4. Set Condition: Only trigger if `api_response` contains `"is_spam": true`.

---

## 🧪 Step 3: Test It Out!

1. Send a test SMS to your phone from another mobile number:
   > *"WINNER!! Claim £900 cash reward now at http://win-claim.com"*
2. Your phone will intercept the message, send it to your SentinelSpam API, and show the **🚨 SPAM ALERT** notification on screen!
