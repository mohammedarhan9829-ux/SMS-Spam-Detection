import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.preprocessing import clean_text

def train_and_evaluate_models(data_path: str = "data/sms_spam_dataset.csv", models_dir: str = "models"):
    """
    Trains multiple ML classification models on preprocessed text,
    benchmarks their performance, saves metrics and exports the best model artifact.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")

    print(f"[Trainer] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate columns
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")

    # Encode labels: ham -> 0, spam -> 1
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(subset=['target', 'text'], inplace=True)
    df['target'] = df['target'].astype(int)

    print("[Trainer] Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Feature extraction with TF-IDF
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned_text']).toarray()
    y = df['target'].values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define Candidate Classifiers
    classifiers = {
        'Multinomial NB': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='sigmoid', gamma=1.0, probability=True, random_state=42)
    }

    results = []
    best_model_name = None
    best_model_obj = None
    best_precision = -1.0
    best_accuracy = -1.0

    print("[Trainer] Benchmarking candidate models...")
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Confusion Matrix': cm
        })

        # Selection criteria: High Precision first, then Accuracy
        if (prec > best_precision) or (prec == best_precision and acc > best_accuracy):
            best_precision = prec
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model

    results_df = pd.DataFrame(results).sort_values(by=['Precision', 'Accuracy'], ascending=False)
    print(f"\n[Trainer] Benchmark Results:\n{results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]}\n")
    print(f"[Trainer] Selected Best Model: {best_model_name} (Precision: {best_precision:.4f}, Accuracy: {best_accuracy:.4f})")

    # Export Model Artifacts
    os.makedirs(models_dir, exist_ok=True)
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(models_dir, "spam_classifier.pkl")

    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(best_model_obj, model_path)
    print(f"[Trainer] Model artifacts successfully saved to '{models_dir}/'")

    return results_df, best_model_name

if __name__ == "__main__":
    train_and_evaluate_models()
