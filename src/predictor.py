import os
import re
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import clean_text, extract_metadata_features

class SpamPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.model_path = os.path.join(models_dir, "spam_classifier.pkl")
        self.vectorizer = None
        self.model = None

        self._load_or_train()

    def _load_or_train(self):
        if not (os.path.exists(self.vectorizer_path) and os.path.exists(self.model_path)):
            print("[Predictor] Model files not found. Triggering automated model training...")
            from src.trainer import train_and_evaluate_models
            train_and_evaluate_models(models_dir=self.models_dir)

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        print("[Predictor] Vectorizer and Model successfully loaded.")

    def _calculate_spam_heuristics(self, text: str) -> float:
        """
        Calculates rule-based heuristic spam points for modern phishing, lottery,
        banking, and link scam patterns (e.g. Rs, PAN card, KYC, urgent, winning).
        """
        text_lower = text.lower()
        score = 0.0

        # Phishing / Scam Keywords
        scam_words = [
            'win', 'winner', 'won', 'congratulation', 'congrats', 'prize', 'reward', 'cash', 'lottery',
            'claim', 'urgent', 'blocked', 'suspend', 'kyc', 'pan card', 'bank account', 'update now',
            'free', 'lakh', 'crore', 'rs', 'rupees', 'loan approved', 'recharge', 'apk', 'click here',
            'limited time', 'selected', 'call immediately', 'txt', 'text to', 'customer service'
        ]

        for word in scam_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                score += 18.0

        # Presence of URLs or Links
        if re.search(r'https?://|www\.|bit\.ly|\.com|\.in|\.apk|\.xyz', text_lower):
            score += 25.0

        # Presence of Phone Numbers to Call
        if re.search(r'\b\d{10}\b|\b090\d{8}\b|\b0800\d{6,8}\b', text_lower):
            score += 15.0

        # Currency symbols or amounts (Rs 1000, £900, $500)
        if re.search(r'(rs\.?|rupees|£|\$|€|₹)\s*\d+', text_lower) or re.search(r'\d+\s*(rs|lakh|crore|points)', text_lower):
            score += 20.0

        return min(score, 85.0)

    def predict_single(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                'label': 'Invalid Input',
                'is_spam': False,
                'probability': 0.0,
                'risk_level': 'Low Risk',
                'cleaned_text': '',
                'top_words': [],
                'metadata': extract_metadata_features(text)
            }

        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.lower()

        vectorized = self.vectorizer.transform([cleaned])

        # Get ML Model Probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized)[0]
            ml_spam_prob = float(probs[1]) * 100.0
        else:
            pred_class = self.model.predict(vectorized)[0]
            ml_spam_prob = 100.0 if pred_class == 1 else 0.0

        # Combine ML Probability with Heuristic Phishing Score
        heuristic_score = self._calculate_spam_heuristics(text)
        combined_prob = max(ml_spam_prob, heuristic_score)

        # High sensitivity threshold for spam detection
        is_spam = bool(combined_prob >= 35.0 or ml_spam_prob >= 35.0)
        label = "Spam" if is_spam else "Ham"

        if combined_prob >= 70.0:
            risk_level = "High Risk"
        elif combined_prob >= 35.0:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        # Feature breakdown
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        vec_array = vectorized.toarray()[0]
        nonzero_indices = vec_array.nonzero()[0]
        
        top_words = []
        for idx in nonzero_indices:
            top_words.append({
                'word': feature_names[idx],
                'tfidf_score': round(float(vec_array[idx]), 4)
            })
        top_words.sort(key=lambda x: x['tfidf_score'], reverse=True)

        return {
            'label': label,
            'is_spam': is_spam,
            'probability': round(combined_prob, 2),
            'risk_level': risk_level,
            'cleaned_text': cleaned,
            'top_words': top_words[:8],
            'metadata': extract_metadata_features(text)
        }

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        df_out = df.copy()
        if text_column not in df_out.columns:
            raise KeyError(f"Column '{text_column}' not found in dataframe.")

        results = df_out[text_column].astype(str).apply(self.predict_single)
        df_out['Cleaned Text'] = [r['cleaned_text'] for r in results]
        df_out['Prediction'] = [r['label'] for r in results]
        df_out['Spam Probability (%)'] = [r['probability'] for r in results]
        df_out['Risk Level'] = [r['risk_level'] for r in results]

        return df_out
