# 🛡️ SentinelSpam: Intelligent SMS & Text Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Natural Language Processing (NLP) solution for identifying and filtering unsolicited SMS spam, phishing attempts, and fraudulent text messages.

---

## 🌟 Key Features

- **🔍 Real-Time Message Analysis**: Instant prediction with spam probability confidence scoring and risk categorization (High, Moderate, Low).
- **📁 Batch File Processing**: Bulk classify hundreds of text messages with interactive tabular results and downloadable annotated CSVs.
- **📊 Interactive EDA Dashboard**: Visual analysis displaying dataset distributions, character/word length metrics, and class ratios.
- **⚡ High-Precision Machine Learning Engine**: Benchmarked across 5 algorithms (Extra Trees, Random Forest, Multinomial Naive Bayes, Support Vector Machines, Logistic Regression) achieving **100% Precision** to prevent false positives on legitimate messages.
- **🌐 REST API Support**: Built-in FastAPI backend providing endpoints for single and batch predictions.

---

## 📊 Model Performance Benchmarks

All candidate algorithms were evaluated on 5,572 labeled messages (86.6% Ham, 13.4% Spam) using an 80-20 stratified train-test split:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Classifier** | **98.21%** | **100.0%** | **86.58%** | **0.9281** | **Selected Production Model** |
| **Random Forest Classifier** | 98.21% | 99.24% | 87.25% | 0.9286 | Evaluated |
| **Multinomial Naive Bayes** | 97.58% | 99.19% | 82.55% | 0.9011 | Evaluated |
| **Support Vector Machine (Sigmoid)** | 98.30% | 97.79% | 89.26% | 0.9333 | Evaluated |
| **Logistic Regression** | 96.95% | 98.32% | 78.52% | 0.8731 | Evaluated |

---

## 📁 Repository Structure

```
sms-spam-detection/
├── app.py                      # Main Streamlit Web Application
├── api.py                      # FastAPI REST API Backend
├── requirements.txt            # Package Dependencies
├── Dockerfile                  # Container Config for Cloud Deployment
├── render.yaml                 # Render Blueprint Config
├── README.md                   # Project Documentation
├── data/
│   └── sms_spam_dataset.csv    # Labeled Dataset (5,572 records)
├── models/
│   ├── tfidf_vectorizer.pkl    # Serialized TF-IDF Vectorizer
│   └── spam_classifier.pkl     # Production ML Classifier Model
├── notebooks/
│   └── Spam_Detection_Analysis.ipynb # Exploratory Data Analysis & Training Notebook
└── src/
    ├── __init__.py
    ├── preprocessing.py        # Text Normalization & Feature Extraction
    ├── trainer.py              # ML Benchmark & Training Pipeline
    └── predictor.py            # High-Level Single & Batch Inference Engine
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9 or higher installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/mohammedarhan9829-ux/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 4. Run the FastAPI REST API (Optional)
```bash
python api.py
```
Navigate to `http://localhost:8000/docs` for interactive Swagger API documentation.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
