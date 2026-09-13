# 🛡️ SentinelSpam: Intelligent SMS & Text Spam Detection Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Natural Language Processing (NLP) solution for identifying and filtering unsolicited SMS spam, phishing attempts, and fraudulent text messages.

---

## 🌟 Key Features

- **🔍 Real-Time Message Analysis**: Instant prediction with spam probability confidence scoring and risk categorization (High, Moderate, Low).
- **📁 Batch CSV File Processing**: Bulk classify hundreds of text messages with interactive tabular results and downloadable annotated CSVs.
- **📊 Interactive EDA Dashboard**: Built-in visual analysis displaying dataset distributions, character/word length metrics, and class ratios.
- **⚡ High-Precision Machine Learning Engine**: Benchmarked across 5 algorithms (Multinomial Naive Bayes, Support Vector Machines, Random Forest, Logistic Regression, Extra Trees) with a target of 100% precision to prevent false positives on legitimate messages.
- **🧩 Modular Architecture**: Clean, reusable Python package structure adhering to production software engineering standards.

---

## 📊 Model Performance Benchmarks

All candidate algorithms were evaluated on 5,572 labeled messages (86.6% Ham, 13.4% Spam) using a 80-20 stratified train-test split:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Classifier** | **98.21%** | **100.0%** | **86.58%** | **0.9281** | **Selected Production Model** |
| **Random Forest Classifier** | 98.21% | 99.24% | 87.25% | 0.9286 | Candidate |
| **Multinomial Naive Bayes** | 97.58% | 99.19% | 82.55% | 0.9011 | Candidate |
| **Support Vector Machine (Sigmoid)** | 98.30% | 97.79% | 89.26% | 0.9333 | Candidate |
| **Logistic Regression** | 96.95% | 98.32% | 78.52% | 0.8731 | Candidate |

---

## 📁 Repository Structure

```
sms-spam-detection/
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Package Dependencies
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
git clone https://github.com/YOUR_USERNAME/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train / Benchmark Models (Optional)
The pre-trained model artifacts are included under `models/`. To re-train or benchmark candidate algorithms:
```bash
python -m src.trainer
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud

Deploying your own version of SentinelSpam to Streamlit Cloud is free and takes less than 2 minutes:

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit - SentinelSpam ML Application"
   git remote add origin https://github.com/YOUR_USERNAME/sms-spam-detection.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy on Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **New app**.
   - Select your repository (`sms-spam-detection`), branch (`main`), and set **Main file path** to `app.py`.
   - Click **Deploy!**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
