import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.predictor import SpamPredictor
from src.preprocessing import extract_metadata_features

# Page Configuration
st.set_page_config(
    page_title="SentinelSpam - AI SMS Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Professional Dark/Light Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge-spam {
        background-color: #ef4444;
        color: white;
        padding: 8px 22px;
        border-radius: 30px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-ham {
        background-color: #22c55e;
        color: white;
        padding: 8px 22px;
        border-radius: 30px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor Engine (cached)
@st.cache_resource
def get_predictor():
    return SpamPredictor(models_dir="models")

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error initializing classification engine: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-header">🛡️ SentinelSpam — AI Text Spam Filter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production Machine Learning & Natural Language Processing Classification System</div>', unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.title("System Specs")
    st.info("""
    **SentinelSpam v2.0**
    - **Vectorization**: TF-IDF (Unigrams & Bigrams)
    - **Classifier**: Extra Trees Ensemble (100% Precision)
    - **Pre-processing**: Porter Stemming & Regex Normalization
    """)
    st.markdown("---")
    st.caption("🔒 Privacy First — Predictions evaluated strictly in-memory.")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-time Analyzer",
    "📁 Batch Predictor",
    "📊 Dataset & Model EDA",
    "ℹ️ Architecture & Metrics"
])

# ---------------------------------------------------------
# TAB 1: Real-time Single SMS Predictor
# ---------------------------------------------------------
with tab1:
    st.subheader("Analyze Message Content")
    st.write("Type or paste an SMS message below, or choose one of the preset samples to test.")

    # Preset Sample Selector
    sample_presets = {
        "Custom Input": "",
        "🎁 Scam Prize": "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Valid 12 hrs.",
        "🏦 Phishing Alert": "URGENT: Your bank account has been temporarily locked due to suspicious activity. Visit http://secure-bank-login.com to restore access immediately.",
        "💬 Genuine Chat": "Hey mate, are we still meeting up for dinner tonight at 7pm? Let me know!",
        "🔑 OTP Verification": "Your verification code is 839201. It will expire in 10 minutes. Do not share this code with anyone."
    }

    selected_preset = st.selectbox("Quick Test Presets:", list(sample_presets.keys()))
    default_text = sample_presets[selected_preset]

    input_text = st.text_area("SMS Message Input:", value=default_text, height=120, placeholder="Enter text message here...")

    analyze_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)

    if analyze_clicked or (input_text and selected_preset != "Custom Input"):
        if not input_text.strip():
            st.warning("Please enter a valid text message to analyze.")
        else:
            res = predictor.predict_single(input_text)
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1.5, 2])

            with res_col1:
                st.markdown("### Prediction Verdict")
                if res['is_spam']:
                    st.markdown('<div class="badge-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-ham">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)

                st.write("")
                st.metric(label="Spam Risk Probability", value=f"{res['probability']}%", delta=res['risk_level'], delta_color="inverse" if res['is_spam'] else "normal")
                
                # Progress Bar for probability
                st.progress(res['probability'] / 100.0)

            with res_col2:
                st.markdown("### Message Structural Features")
                meta = res['metadata']
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Characters", meta['num_characters'])
                m_col2.metric("Words", meta['num_words'])
                m_col3.metric("Uppercase Chars", meta['num_uppercase'])

                m_col4, m_col5, m_col6 = st.columns(3)
                m_col4.metric("Digits Count", meta['num_digits'])
                m_col5.metric("Currency Symbols", meta['num_currency_symbols'])
                m_col6.metric("Contains URL", "Yes" if meta['has_url'] else "No")

            # Top TF-IDF Terms breakdown
            if res['top_words']:
                st.markdown("---")
                st.markdown("### 🔑 Key Informative Terms Extracted")
                top_df = pd.DataFrame(res['top_words'])
                st.dataframe(top_df, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: Batch Predictor
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch File Processing")
    st.write("Upload a CSV or TXT file containing multiple SMS messages for automated classification.")

    uploaded_file = st.file_uploader("Upload File (CSV format recommended):", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file, encoding="latin-1")
            else:
                lines = [line.decode("utf-8", errors="ignore").strip() for line in uploaded_file.readlines()]
                df_upload = pd.DataFrame({"text": lines})

            st.write(f"📁 Loaded dataset containing **{len(df_upload)}** rows.")

            # Select column containing SMS text
            text_cols = list(df_upload.columns)
            selected_col = st.selectbox("Select column containing SMS text:", text_cols, index=0)

            if st.button("⚡ Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    results_df = predictor.predict_dataframe(df_upload, text_column=selected_col)
                
                st.success("Batch classification complete!")

                # Overview Statistics
                spam_cnt = (results_df['Prediction'] == 'Spam').sum()
                ham_cnt = (results_df['Prediction'] == 'Ham').sum()

                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Total Messages", len(results_df))
                stat_col2.metric("Spam Detected", spam_cnt)
                stat_col3.metric("Legitimate (Ham)", ham_cnt)

                st.dataframe(results_df, use_container_width=True)

                # Download Button
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=csv_data,
                    file_name="sentinel_spam_predictions.csv",
                    mime="text/csv"
                )

        except Exception as ex:
            st.error(f"Error processing file: {ex}")


# ---------------------------------------------------------
# TAB 3: Dataset & Model EDA
# ---------------------------------------------------------
with tab3:
    st.subheader("Exploratory Data Analysis & Benchmark Metrics")

    data_path = "data/sms_spam_dataset.csv"
    if os.path.exists(data_path):
        df_eda = pd.read_csv(data_path)
        
        col_eda1, col_eda2 = st.columns([1, 1])

        with col_eda1:
            st.markdown("#### Class Distribution (Ham vs Spam)")
            class_counts = df_eda['label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ['#22c55e', '#ef4444']
            ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.05, 0))
            ax.set_title("SMS Label Distribution")
            st.pyplot(fig)

        with col_eda2:
            st.markdown("#### Dataset Statistics")
            st.write(df_eda.groupby('label').describe())

        st.markdown("---")
        st.markdown("#### Model Architecture Benchmark Comparison")

        benchmark_data = pd.DataFrame([
            {"Model": "Extra Trees Classifier", "Accuracy": "98.2%", "Precision": "100.0%", "Recall": "86.6%", "F1-Score": "0.928"},
            {"Model": "Random Forest Classifier", "Accuracy": "98.2%", "Precision": "99.2%", "Recall": "87.3%", "F1-Score": "0.929"},
            {"Model": "Multinomial Naive Bayes", "Accuracy": "97.6%", "Precision": "99.2%", "Recall": "82.6%", "F1-Score": "0.901"},
            {"Model": "Support Vector Machine (Sigmoid)", "Accuracy": "98.3%", "Precision": "97.8%", "Recall": "89.3%", "F1-Score": "0.933"},
            {"Model": "Logistic Regression", "Accuracy": "97.0%", "Precision": "98.3%", "Recall": "78.5%", "F1-Score": "0.873"},
        ])
        st.table(benchmark_data)
    else:
        st.info("Dataset file `data/sms_spam_dataset.csv` not found for EDA preview.")


# ---------------------------------------------------------
# TAB 4: Architecture & Metrics
# ---------------------------------------------------------
with tab4:
    st.subheader("System Architecture & Implementation")

    st.markdown("""
    ### 🏗️ Technical Pipeline
    1. **Data Normalization**: Strips special noise, standardizes token formats, maps currency and URL patterns.
    2. **NLP Preprocessing**: Tokenization via NLTK, English Stopword elimination, and Porter Stemming normalization.
    3. **Vectorization**: TF-IDF Matrix representation using unigrams & bigrams (Max 3,000 features).
    4. **Inference Engine**: Evaluated via Extra Trees Classifier with 100% Precision target to avoid false positives on legitimate messages.
    """)

st.markdown("---")
st.caption("SentinelSpam System • Built with Python, Scikit-Learn & Streamlit")
