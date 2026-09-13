import os
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

    def predict_single(self, text: str) -> dict:
        """
        Predicts whether a single input message is Spam or Ham with confidence score.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'label': 'Invalid Input',
                'is_spam': False,
                'probability': 0.0,
                'risk_level': 'Low',
                'cleaned_text': '',
                'top_words': [],
                'metadata': extract_metadata_features(text)
            }

        cleaned = clean_text(text)
        if not cleaned:
            # Fallback if text only contains symbols/stopwords
            cleaned = text.lower()

        vectorized = self.vectorizer.transform([cleaned])

        # Get prediction and probability
        pred_class = self.model.predict(vectorized)[0]
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized)[0]
            spam_prob = float(probs[1]) * 100.0
        else:
            spam_prob = 100.0 if pred_class == 1 else 0.0

        is_spam = bool(pred_class == 1)
        label = "Spam" if is_spam else "Ham"

        if spam_prob >= 75.0:
            risk_level = "High Risk"
        elif spam_prob >= 40.0:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        # Extract top feature words present in text
        words = cleaned.split()
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
            'probability': round(spam_prob, 2),
            'risk_level': risk_level,
            'cleaned_text': cleaned,
            'top_words': top_words[:8],
            'metadata': extract_metadata_features(text)
        }

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Batch predicts a dataframe containing a text column.
        """
        df_out = df.copy()
        if text_column not in df_out.columns:
            raise KeyError(f"Column '{text_column}' not found in dataframe.")

        cleaned_texts = df_out[text_column].astype(str).apply(clean_text)
        vectorized = self.vectorizer.transform(cleaned_texts)
        preds = self.model.predict(vectorized)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized)[:, 1] * 100.0
        else:
            probs = np.where(preds == 1, 100.0, 0.0)

        df_out['Cleaned Text'] = cleaned_texts
        df_out['Prediction'] = np.where(preds == 1, 'Spam', 'Ham')
        df_out['Spam Probability (%)'] = np.round(probs, 2)
        df_out['Risk Level'] = np.where(probs >= 75.0, 'High Risk',
                                np.where(probs >= 40.0, 'Moderate Risk', 'Low Risk'))

        return df_out
