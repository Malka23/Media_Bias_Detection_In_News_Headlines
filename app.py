import streamlit as st
import joblib
import re
import numpy as np
from sentence_transformers import SentenceTransformer


# PAGE CONFIG
st.set_page_config(
    page_title="Media Bias Detection",
    page_icon="📰",
    layout="centered"
)

# MODERN CSS
st.markdown("""
<style>
.main-title {
    font-size: 100px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b4b, #4b6cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<p class="main-title">📰 Media Bias Detection</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Left or Right or Neutral bias</p>',
    unsafe_allow_html=True
)

# LOAD MODELS
@st.cache_resource
def load_models():
    clf = joblib.load("models/bias_model.pkl") 
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return clf, sbert

clf, sbert_model = load_models()

# TEXT CLEANING

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# INPUT

headline = st.text_area(
    "✍️ Enter News Headline",
    placeholder="Example: Government announces major tax reform...",
    height=120
)

analyze_btn = st.button("🔍 Analyze Bias", use_container_width=True)

# PREDICTION
if analyze_btn:

    if headline.strip() == "":
        st.warning("⚠️ Please enter a headline.")
        st.stop()

    with st.spinner("Analyzing political bias..."):

        emb = sbert_model.encode([headline])

        pred_label = clf.predict(emb)[0]

        decision_scores = clf.decision_function(emb)[0]
        scores = np.abs(decision_scores)
        confidence = float(np.max(scores) / np.sum(scores))

    st.divider()

    # RESULT DISPLAY
    if pred_label == "left":
        st.error(f"🔴 LEFT BIAS detected")
    elif pred_label == "right":
        st.success(f"🔵 RIGHT BIAS detected")
    else:
        st.info(f"⚪ CENTER bias detected")

    
    # CONFIDENCE METER (approximate)
    
    st.subheader("🧠 Model Confidence")

    if confidence > 0.75:
        st.success("High confidence prediction")
    elif confidence > 0.55:
        st.warning("Moderate confidence prediction")
    else:
        st.error("Low confidence — headline may be ambiguous")


# SIDEBAR

with st.sidebar:
    st.header("ℹ️ About")
    st.write(""" 

**Classes:** Left / Center / Right  
             
**Project:** Media Bias Detection
""")
    st.write("👩‍💻 Built by Malka Naaz")