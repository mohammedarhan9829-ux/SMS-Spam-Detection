import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.predictor import SpamPredictor
from src.preprocessing import extract_metadata_features, clean_text

# Page Configuration
st.set_page_config(
    page_title="SentinelSpam - AI SMS Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Rich Dark Accent Palette & Custom Badges)
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
    }
    .metric-container {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .badge-spam {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 30px;
        font-size: 1.35rem;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    .badge-ham {
        background: linear-gradient(135deg, #22c55e 0%, #15803d 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 30px;
        font-size: 1.35rem;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.4);
    }
    .pipeline-step {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor Engine (cached for performance)
@st.cache_resource
def get_predictor():
    return SpamPredictor(models_dir="models")

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error initializing classification engine: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-title">🛡️ SentinelSpam — AI Text & SMS Spam Filter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production Machine Learning & Natural Language Processing Text Classifier</div>', unsafe_allow_html=True)

# Sidebar Configuration & Settings
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/security-checked.png", width=70)
    st.title("Model Controls")
    
    st.markdown("### Classification Sensitivity")
    sensitivity_threshold = st.slider(
        "Decision Threshold (%):",
        min_value=10,
        max_value=90,
        value=35,
        step=5,
        help="Messages with spam probability equal or higher than this threshold will be flagged as SPAM."
    )
    
    st.markdown("---")
    st.info("""
    **Engine Specifications**
    - **Vectorization**: TF-IDF (Unigrams & Bigrams)
    - **Model**: Extra Trees Ensemble
    - **Precision Target**: 100.0%
    """)
    st.markdown("---")
    st.caption("🔒 100% Privacy — Evaluated strictly in-memory.")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Live Detector",
    "📁 Batch Processing",
    "📊 Dataset & Model EDA",
    "🔬 NLP Pipeline Visualizer",
    "📜 Academic Overview"
])

# ---------------------------------------------------------
# TAB 1: Live Spam Detector
# ---------------------------------------------------------
with tab1:
    st.subheader("Analyze SMS Message Content")
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

    analyze_clicked = st.button("🚀 Analyze Message Now", type="primary", use_container_width=True)

    if analyze_clicked or (input_text and selected_preset != "Custom Input"):
        if not input_text.strip():
            st.warning("Please enter a valid text message to analyze.")
        else:
            res = predictor.predict_single(input_text)
            
            # Apply user custom sensitivity threshold from sidebar
            prob = res['probability']
            is_spam = prob >= sensitivity_threshold
            verdict_label = "Spam" if is_spam else "Ham"

            st.markdown("---")
            res_col1, res_col2 = st.columns([1.5, 2])

            with res_col1:
                st.markdown("### Classification Verdict")
                if is_spam:
                    st.markdown('<div class="badge-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-ham">✅ LEGITIMATE (HAM)</div>', unsafe_allow_html=True)

                st.write("")
                st.metric(
                    label="Spam Risk Score",
                    value=f"{prob}%",
                    delta=res['risk_level'],
                    delta_color="inverse" if is_spam else "normal"
                )
                
                # Progress Bar for probability
                st.progress(prob / 100.0)
                st.caption(f"Decision Threshold: {sensitivity_threshold}%")

            with res_col2:
                st.markdown("### Structural Metadata")
                meta = res['metadata']
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Characters", meta['num_characters'])
                m_col2.metric("Words", meta['num_words'])
                m_col3.metric("Uppercase Chars", meta['num_uppercase'])

                m_col4, m_col5, m_col6 = st.columns(3)
                m_col4.metric("Digits Count", meta['num_digits'])
                m_col5.metric("Currency Symbols", meta['num_currency_symbols'])
                m_col6.metric("Contains Link/URL", "Yes" if meta['has_url'] else "No")

            # Top TF-IDF Terms breakdown
            if res['top_words']:
                st.markdown("---")
                st.markdown("### 🔑 Key Informative Terms Extracted")
                top_df = pd.DataFrame(res['top_words'])
                
                col_chart, col_table = st.columns([1.5, 1])
                with col_chart:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.barplot(x='tfidf_score', y='word', data=top_df, palette='Reds_r' if is_spam else 'Greens_r', ax=ax)
                    ax.set_title("TF-IDF Term Weight Contributions")
                    ax.set_xlabel("TF-IDF Importance Score")
                    st.pyplot(fig)

                with col_table:
                    st.dataframe(top_df, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: Batch Processing
# ---------------------------------------------------------
with tab2:
    st.subheader("Bulk Batch Dataset Classification")
    st.write("Upload a CSV or TXT file containing multiple SMS messages for instant automated classification.")

    uploaded_file = st.file_uploader("Upload File (CSV format recommended):", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file, encoding="latin-1")
            else:
                lines = [line.decode("utf-8", errors="ignore").strip() for line in uploaded_file.readlines()]
                df_upload = pd.DataFrame({"text": lines})

            st.write(f"📁 Loaded dataset containing **{len(df_upload)}** rows.")

            text_cols = list(df_upload.columns)
            selected_col = st.selectbox("Select column containing SMS text:", text_cols, index=0)

            if st.button("⚡ Run Batch Prediction Engine"):
                with st.spinner("Processing batch classification..."):
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
            ax.set_title("SMS Label Distribution (5,572 Records)")
            st.pyplot(fig)

        with col_eda2:
            st.markdown("#### Dataset Statistical Summary")
            df_eda['length'] = df_eda['text'].astype(str).apply(len)
            st.write(df_eda.groupby('label')['length'].describe())

        st.markdown("---")
        st.markdown("#### Model Architecture Benchmark Comparison")

        benchmark_data = pd.DataFrame([
            {"Model": "Extra Trees Classifier", "Accuracy": "98.21%", "Precision": "100.0%", "Recall": "86.58%", "F1-Score": "0.9281"},
            {"Model": "Random Forest Classifier", "Accuracy": "98.21%", "Precision": "99.24%", "Recall": "87.25%", "F1-Score": "0.9286"},
            {"Model": "Multinomial Naive Bayes", "Accuracy": "97.58%", "Precision": "99.19%", "Recall": "82.55%", "F1-Score": "0.9011"},
            {"Model": "Support Vector Machine (Sigmoid)", "Accuracy": "98.30%", "Precision": "97.79%", "Recall": "89.26%", "F1-Score": "0.9333"},
            {"Model": "Logistic Regression", "Accuracy": "96.95%", "Precision": "98.32%", "Recall": "78.52%", "F1-Score": "0.8731"},
        ])
        st.table(benchmark_data)
    else:
        st.info("Dataset file `data/sms_spam_dataset.csv` not found for EDA preview.")


# ---------------------------------------------------------
# TAB 4: NLP Pipeline Visualizer
# ---------------------------------------------------------
with tab4:
    st.subheader("Step-by-Step NLP Transformation Pipeline")
    st.write("See how raw input text is converted into preprocessed tokens and numerical vector matrices.")

    viz_input = st.text_input("Enter text to visualize:", value="WINNER!! Claim £900 cash reward now at http://win-claim.com")
    
    if viz_input:
        st.markdown('<div class="pipeline-step"><strong>Step 1: Raw Input Text</strong><br><code>' + viz_input + '</code></div>', unsafe_allow_html=True)
        
        # Step 2: Preprocessing
        cleaned = clean_text(viz_input)
        st.markdown('<div class="pipeline-step"><strong>Step 2: Text Cleaning & Stemming (Lowercasing, Stopword Removal, Stemming)</strong><br><code>' + cleaned + '</code></div>', unsafe_allow_html=True)

        # Step 3: Tokens
        tokens = cleaned.split()
        st.markdown('<div class="pipeline-step"><strong>Step 3: Tokens Generated</strong><br><code>' + str(tokens) + '</code></div>', unsafe_allow_html=True)

        # Step 4: Vectorizer transformation
        vec = predictor.vectorizer.transform([cleaned])
        nonzero = vec.toarray()[0].nonzero()[0]
        st.markdown(f'<div class="pipeline-step"><strong>Step 4: Sparse TF-IDF Vector (Non-zero Features: {len(nonzero)})</strong></div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 5: Academic Overview
# ---------------------------------------------------------
with tab5:
    st.subheader("Academic Overview & Technical Specifications")

    st.markdown("""
    ### 🏗️ Technical Pipeline & Methodology
    1. **Data Preprocessing**: Tokenization via NLTK, lowercasing, English stopword removal, and Porter Stemming normalization.
    2. **Feature Extraction**: TF-IDF Matrix representation using unigrams & bigrams (Max 3,000 features).
    3. **Model Selection**: Evaluated across 5 candidate algorithms (Naive Bayes, Random Forest, SVM, Logistic Regression, Extra Trees). Extra Trees was selected due to achieving **100% Precision** to prevent false positives on legitimate messages.
    4. **Hybrid Detection Engine**: Combines ML probability with heuristic pattern rules for phishing URLs, currency symbols, and modern scam keywords.

    ---
    ### 📜 System Information
    - **Language**: Python 3.9+
    - **Libraries**: Scikit-Learn, Pandas, NumPy, NLTK, Streamlit, Matplotlib, Seaborn
    - **License**: MIT License
    """)

st.markdown("---")
st.caption("SentinelSpam Machine Learning Engine • Portfolio Edition")
